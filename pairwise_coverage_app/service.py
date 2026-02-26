from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from evaluate_scores import calc_scores

from pairwise_coverage_app.models import CSV_COLUMNS, VoteRecord


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


@dataclass
class PairSelection:
    image_a: str
    image_b: str


class PairwiseCoverageError(Exception):
    pass


class PairwiseCoverageService:
    def __init__(
        self,
        data_dir: Path,
        state_dir: Path,
        target_n: int,
    ) -> None:
        if target_n < 0:
            raise PairwiseCoverageError("n must be >= 0")

        self.data_dir = data_dir.resolve()
        self.state_dir = state_dir.resolve()
        self.state_path = self.state_dir / "global_state.json"
        self.csv_path = self.state_dir / "labels.csv"
        self.target_n = target_n

        self.lock = Lock()
        self.image_paths: dict[str, Path] = {}
        self.image_ids: list[str] = []
        self.appearance_counts: dict[str, int] = {}
        self.used_pairs: set[tuple[str, str]] = set()
        self.used_degree: dict[str, int] = {}
        self.round_seen: set[str] = set()
        self.total_votes = 0
        self.status = "active"
        self.status_reason = ""
        self.dataset_signature = ""
        self._rng = random.SystemRandom()
        self.undo_available = False

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._load_images()
        self._ensure_csv()
        self._load_existing_votes()
        self._reconcile_or_create_state()
        self._refresh_status()
        self._persist_state()

    @property
    def _max_unique_pairs(self) -> int:
        image_count = len(self.image_ids)
        return (image_count * (image_count - 1)) // 2

    def _load_images(self) -> None:
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            raise PairwiseCoverageError(f"Data directory not found: {self.data_dir}")

        image_paths = sorted(
            [
                path
                for path in self.data_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
        )

        if not image_paths:
            raise PairwiseCoverageError(f"No images found in {self.data_dir}")

        ids: list[str] = []
        mapping: dict[str, Path] = {}
        for path in image_paths:
            image_id = path.relative_to(self.data_dir).as_posix()
            ids.append(image_id)
            mapping[image_id] = path

        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()

        self.image_ids = ids
        self.image_paths = mapping
        self.dataset_signature = digest

    def _ensure_csv(self) -> None:
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()

    def _read_vote_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            return []

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != CSV_COLUMNS:
                raise PairwiseCoverageError(
                    f"Unexpected CSV columns in {self.csv_path}. Expected {CSV_COLUMNS}, got {reader.fieldnames}"
                )
            return list(reader)

    def _write_vote_rows(self, rows: list[dict[str, str]]) -> None:
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def _load_existing_votes(self) -> None:
        self.appearance_counts = {image_id: 0 for image_id in self.image_ids}
        self.used_degree = {image_id: 0 for image_id in self.image_ids}
        self.used_pairs = set()
        self.round_seen = set()
        self.total_votes = 0

        cycle_pairs: set[tuple[str, str]] = set()
        cycle_degree = {image_id: 0 for image_id in self.image_ids}

        if not self.csv_path.exists():
            return

        for row in self._read_vote_rows():
            left = row["image_a"]
            right = row["image_b"]
            if left not in self.appearance_counts or right not in self.appearance_counts:
                raise PairwiseCoverageError(
                    "CSV contains image IDs not present in current dataset. "
                    "Use a separate state directory or clear old state files."
                )

            key = _pair_key(left, right)

            if self._max_unique_pairs > 0 and len(cycle_pairs) >= self._max_unique_pairs:
                cycle_pairs = set()
                cycle_degree = {image_id: 0 for image_id in self.image_ids}

            if key in cycle_pairs:
                cycle_pairs = set()
                cycle_degree = {image_id: 0 for image_id in self.image_ids}

            cycle_pairs.add(key)
            cycle_degree[left] += 1
            cycle_degree[right] += 1

            self.appearance_counts[left] += 1
            self.appearance_counts[right] += 1
            self.total_votes += 1

            self._apply_round_seen_after_vote(left, right)

        self.used_pairs = cycle_pairs
        self.used_degree = cycle_degree

    def _reconcile_or_create_state(self) -> None:
        if not self.state_path.exists():
            return

        data = json.loads(self.state_path.read_text(encoding="utf-8"))

        stored_signature = data.get("dataset_signature")
        if stored_signature and stored_signature != self.dataset_signature:
            raise PairwiseCoverageError(
                "Dataset has changed from existing state. Use a new state directory or clear prior state files."
            )

        stored_data_dir = data.get("data_dir")
        if stored_data_dir and Path(stored_data_dir).resolve() != self.data_dir:
            raise PairwiseCoverageError(
                "State directory is linked to a different data directory. Use a new state directory."
            )

    def _deficit(self, image_id: str) -> int:
        return max(0, self.target_n - self.appearance_counts[image_id])

    def _remaining_neighbors(self, image_id: str) -> int:
        return (len(self.image_ids) - 1) - self.used_degree[image_id]

    def _check_impossible_reason(self) -> str:
        image_count = len(self.image_ids)
        if image_count < 2 and self.target_n > 0:
            return "At least two images are required to continue comparisons"

        return ""

    def _reset_seen_pairs_cycle(self) -> None:
        self.used_pairs = set()
        self.used_degree = {image_id: 0 for image_id in self.image_ids}

    def _apply_round_seen_after_vote(self, image_a: str, image_b: str) -> None:
        self.round_seen.add(image_a)
        self.round_seen.add(image_b)
        if len(self.round_seen) >= len(self.image_ids):
            self.round_seen = set()

    def _round_candidates(self, candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if not candidates:
            return []

        unseen = {image_id for image_id in self.image_ids if image_id not in self.round_seen}

        if not unseen:
            self.round_seen = set()
            unseen = set(self.image_ids)

        if len(unseen) >= 2:
            filtered = [pair for pair in candidates if pair[0] in unseen and pair[1] in unseen]
            if filtered:
                return filtered

        if len(unseen) == 1:
            only_unseen = next(iter(unseen))
            filtered = [pair for pair in candidates if only_unseen in pair]
            if filtered:
                return filtered

        return candidates

    def _refresh_status(self) -> None:
        if all(self.appearance_counts[image_id] >= self.target_n for image_id in self.image_ids):
            self.status = "complete"
            self.status_reason = "All images have reached the target coverage"
            return

        reason = self._check_impossible_reason()
        if reason:
            self.status = "impossible"
            self.status_reason = reason
            return

        self.status = "active"
        self.status_reason = ""

    def _persist_state(self) -> None:
        payload = {
            "data_dir": str(self.data_dir),
            "dataset_signature": self.dataset_signature,
            "target_n": self.target_n,
            "total_images": len(self.image_ids),
            "total_votes": self.total_votes,
            "used_pairs_count": len(self.used_pairs),
            "round_seen_count": len(self.round_seen),
            "undo_available": self.undo_available,
            "status": self.status,
            "status_reason": self.status_reason,
            "appearance_counts": self.appearance_counts,
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _candidate_pairs(self) -> list[tuple[str, str]]:
        deficits = {image_id: self._deficit(image_id) for image_id in self.image_ids}
        needers = [image_id for image_id, deficit in deficits.items() if deficit > 0]
        if not needers:
            return []

        candidates: list[tuple[str, str]] = []
        for left in needers:
            for right in self.image_ids:
                if left == right:
                    continue
                key = _pair_key(left, right)
                if key in self.used_pairs:
                    continue
                candidates.append(key)

        return list(dict.fromkeys(candidates))

    def next_pair(self) -> PairSelection | None:
        with self.lock:
            self._refresh_status()
            if self.status != "active":
                self._persist_state()
                return None

            candidates = self._candidate_pairs()
            if not candidates:
                self._reset_seen_pairs_cycle()
                candidates = self._candidate_pairs()
                if not candidates:
                    self.status = "impossible"
                    self.status_reason = "No candidate pairs are available"
                    self._persist_state()
                    return None

            round_candidates = self._round_candidates(candidates)
            if not round_candidates:
                self._reset_seen_pairs_cycle()
                candidates = self._candidate_pairs()
                round_candidates = self._round_candidates(candidates)
                if not round_candidates:
                    self.status = "impossible"
                    self.status_reason = "No candidates available under round constraints"
                    self._persist_state()
                    return None

            selected_left, selected_right = self._rng.choice(round_candidates)
            return PairSelection(image_a=selected_left, image_b=selected_right)

    def record_vote(self, session_id: str, image_a: str, image_b: str, outcome: str) -> VoteRecord:
        with self.lock:
            if image_a not in self.image_paths or image_b not in self.image_paths:
                raise PairwiseCoverageError("Vote contains unknown image ID")
            if image_a == image_b:
                raise PairwiseCoverageError("Vote must compare two distinct images")

            key = _pair_key(image_a, image_b)
            if key in self.used_pairs:
                raise PairwiseCoverageError("This pair has already been used")

            vote = VoteRecord.create(session_id=session_id, image_a=image_a, image_b=image_b, outcome=outcome)

            with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writerow(vote.to_row())

            self.used_pairs.add(key)
            self.used_degree[image_a] += 1
            self.used_degree[image_b] += 1
            self.appearance_counts[image_a] += 1
            self.appearance_counts[image_b] += 1
            self.total_votes += 1

            self._apply_round_seen_after_vote(image_a, image_b)
            self.undo_available = True

            self._refresh_status()
            self._persist_state()

            return vote

    def undo_last_vote(self, session_id: str | None = None) -> VoteRecord:
        with self.lock:
            if not self.undo_available:
                raise PairwiseCoverageError("Undo is only available for the most recent vote")

            rows = self._read_vote_rows()
            if not rows:
                raise PairwiseCoverageError("No votes available to undo")

            last_row = rows[-1]
            if session_id and last_row.get("session_id") != session_id:
                raise PairwiseCoverageError("The most recent vote was made by a different session")

            rows.pop()
            self._write_vote_rows(rows)

            self._load_existing_votes()
            self.undo_available = False
            self._refresh_status()
            self._persist_state()

            return VoteRecord(
                timestamp=last_row["timestamp"],
                session_id=last_row["session_id"],
                image_a=last_row["image_a"],
                image_b=last_row["image_b"],
                outcome=last_row["outcome"],
            )

    def resolve_image_path(self, image_id: str) -> Path:
        path = self.image_paths.get(image_id)
        if path is None:
            raise PairwiseCoverageError("Unknown image")
        return path

    def calculate_progress_metrics(self) -> tuple[float, float, float] | None:

        scores = calc_scores(self.csv_path.as_posix())

        max_deviation = round(max(scores, key=lambda p: p.deviation).deviation, 2)
        average_deviation = round(sum(p.deviation for p in scores) / len(scores), 2)

        estimated_left = (self.target_n * len(self.image_ids)) // 2 - self.total_votes

        return estimated_left, average_deviation, max_deviation

    def status_payload(self) -> dict:
        with self.lock:
            self._refresh_status()
            remaining = {
                image_id: self._deficit(image_id)
                for image_id in self.image_ids
            }
            return {
                "status": self.status,
                "reason": self.status_reason,
                "target_n": self.target_n,
                "total_images": len(self.image_ids),
                "total_votes": self.total_votes,
                "used_pairs": len(self.used_pairs),
                "can_undo": self.undo_available,
                "csv_path": str(self.csv_path),
                "state_path": str(self.state_path),
                "remaining_deficit": remaining,
            }
