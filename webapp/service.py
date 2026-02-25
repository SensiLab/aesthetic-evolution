from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import yaml

from aesthetic_evolution.generate_designs import aesthetic_evolution
from webapp.models import JobRecord, WebJobConfig


class ValidationError(ValueError):
    pass


class JobService:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.experiments_root = workspace_root / "Experiments"
        self.jobs_root = workspace_root / "webapp" / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.experiments_root.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, JobRecord] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def submit(self, payload: dict[str, Any]) -> JobRecord:
        config = self._parse_config(payload)
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(job_id=job_id, config=config, log_path=str(self.jobs_root / f"{job_id}.log"))

        with self._lock:
            self._jobs[job_id] = record
            self._queue.put(job_id)

        self._write_job_metadata(record)
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobRecord]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def list_experiments(self) -> list[str]:
        if not self.experiments_root.exists():
            return []

        return sorted([item.name for item in self.experiments_root.iterdir() if item.is_dir()])

    def experiment_runs(self, experiment_name: str) -> list[str]:
        exp_path = self.experiments_root / experiment_name
        if not exp_path.exists() or not exp_path.is_dir():
            raise ValidationError(f"Experiment '{experiment_name}' not found.")

        runs = [item.name for item in exp_path.iterdir() if item.is_dir() and item.name.startswith("run")]
        return sorted(runs, key=lambda item: int(item.removeprefix("run")) if item.removeprefix("run").isdigit() else 9999)

    def run_artifacts(self, experiment_name: str, run_name: str) -> dict[str, Any]:
        run_path = self.experiments_root / experiment_name / run_name
        images_path = run_path / "Images"
        params_path = run_path / "Params"

        if not run_path.exists() or not run_path.is_dir():
            raise ValidationError(f"Run '{run_name}' not found for experiment '{experiment_name}'.")

        plot_files = []
        design_files = []
        if images_path.exists():
            for file in sorted(images_path.iterdir()):
                if not file.is_file() or file.suffix.lower() != ".png":
                    continue
                if file.name.startswith("Population"):
                    plot_files.append(file.name)
                else:
                    design_files.append(file.name)

        param_files = []
        if params_path.exists():
            param_files = [file.name for file in sorted(params_path.iterdir()) if file.is_file() and file.suffix.lower() == ".json"]

        return {
            "plot_files": plot_files,
            "design_files": design_files,
            "param_files": param_files,
        }

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        job = self._jobs[job_id]
        config = job.config
        log_path = Path(job.log_path) if job.log_path else self.jobs_root / f"{job_id}.log"
        exp_dir = self.experiments_root / config.experiment_name

        job.status = "running"
        job.started_at = datetime.now(UTC).isoformat()
        self._write_job_metadata(job)

        try:
            if exp_dir.exists():
                if not config.overwrite:
                    raise ValidationError(
                        f"Experiment '{config.experiment_name}' already exists. Enable overwrite to replace it."
                    )
                shutil.rmtree(exp_dir)

            prompt = self._resolve_prompt(config)

            with open(log_path, "w", encoding="utf-8") as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
                print(f"Starting experiment: {config.experiment_name}")
                aesthetic_evolution(
                    experiment_name=config.experiment_name,
                    runs=config.runs,
                    param_spec_filepath=config.param_spec_file,
                    sketch_dir=config.sketch_dir,
                    prompt=prompt,
                    alpha_mode=config.alpha_mode,
                    alpha=config.alpha,
                    mutation_rate=config.mutation_rate,
                    mutation_sigma=config.mutation_sigma,
                    k=config.k,
                    ranking_method=config.ranking_method,
                    population_size=int(self._raw(config, "population_size")),
                    processing=config.processing,
                    screen=config.screen,
                    workers=config.workers,
                    parents_compete=config.parents_compete,
                    competing_parents_rate=config.competing_parents_rate,
                )

                exp_dir.mkdir(parents=True, exist_ok=True)
                with open(exp_dir / "experiment_config.yaml", "w", encoding="utf-8") as output_file:
                    yaml.safe_dump(config.snapshot(), output_file, sort_keys=False)

                if config.prompt_text:
                    with open(exp_dir / "web_prompt.txt", "w", encoding="utf-8") as prompt_file:
                        prompt_file.write(config.prompt_text)

                print("Experiment completed.")

            job.status = "completed"
            job.finished_at = datetime.now(UTC).isoformat()
            job.error = None
            self._write_job_metadata(job)

        except Exception as exc:
            job.status = "failed"
            job.finished_at = datetime.now(UTC).isoformat()
            job.error = str(exc)
            self._write_job_metadata(job)

            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write("\n\n--- ERROR ---\n")
                log_file.write(f"{exc}\n")
                log_file.write(traceback.format_exc())

    def _raw(self, config: WebJobConfig, field_name: str) -> Any:
        return getattr(config, field_name)

    def _resolve_prompt(self, config: WebJobConfig) -> str:
        return config.prompt_text

    def _parse_config(self, payload: dict[str, Any]) -> WebJobConfig:
        alpha_mode = str(payload.get("alpha_mode", "")).strip()
        if alpha_mode not in {"random", "fixed", "biased"}:
            raise ValidationError("alpha_mode must be one of: random, fixed, biased")

        ranking_method = str(payload.get("ranking_method", "")).strip()
        if ranking_method not in {"glicko", "simple", "CLIP-IQA"}:
            raise ValidationError("ranking_method must be one of: glicko, simple, CLIP-IQA")

        processing = str(payload.get("processing", "")).strip()
        if processing not in {"serial", "parallel"}:
            raise ValidationError("processing must be either serial or parallel")

        experiment_name = str(payload.get("experiment_name", "")).strip()
        if not experiment_name:
            raise ValidationError("experiment_name is required")
        if any(char in experiment_name for char in "\\/:*?\"<>|"):
            raise ValidationError("experiment_name contains invalid filesystem characters")

        prompt_filepath = str(payload.get("prompt_filepath", "")).strip() or None
        if prompt_filepath is not None:
            raise ValidationError("prompt_filepath is not supported; provide prompt_text only")

        prompt_text = str(payload.get("prompt_text", "")).strip()
        if not prompt_text:
            raise ValidationError("prompt_text is required")

        alpha = payload.get("alpha")
        if alpha in ("", None):
            alpha_value = None
        else:
            alpha_value = float(alpha)

        if alpha_mode == "fixed":
            if alpha_value is None:
                raise ValidationError("alpha is required when alpha_mode is fixed")
            if not (0 <= alpha_value <= 1):
                raise ValidationError("alpha must be between 0 and 1")

        mutation_rate = float(payload.get("mutation_rate", 0.1))
        if not (0 <= mutation_rate <= 1):
            raise ValidationError("mutation_rate must be between 0 and 1")

        mutation_sigma = float(payload.get("mutation_sigma", 0.1))
        if mutation_sigma <= 0:
            raise ValidationError("mutation_sigma must be positive")

        parents_compete = self._to_bool(payload.get("parents_compete", False))
        competing_parents_rate = payload.get("competing_parents_rate")
        if competing_parents_rate in ("", None):
            competing_rate_value = None
        else:
            competing_rate_value = float(competing_parents_rate)

        if parents_compete:
            if competing_rate_value is None:
                raise ValidationError("competing_parents_rate required when parents_compete is true")
            if not (0 <= competing_rate_value <= 1):
                raise ValidationError("competing_parents_rate must be between 0 and 1")

        runs = int(payload.get("runs", 0))
        population_size = int(payload.get("population_size", 0))
        workers = int(payload.get("workers", 8))
        k = float(payload.get("k", 0))

        if runs <= 0:
            raise ValidationError("runs must be > 0")
        if population_size <= 0 or population_size % 2 != 0:
            raise ValidationError("population_size must be a positive even integer")
        if workers <= 0:
            raise ValidationError("workers must be > 0")
        if not (0 <= k <= 1):
            raise ValidationError("k must be between 0 and 1")

        param_spec_file = str(payload.get("param_spec_file", "")).strip()
        sketch_dir = str(payload.get("sketch_dir", "")).strip()
        if not param_spec_file:
            raise ValidationError("param_spec_file is required")
        if not sketch_dir:
            raise ValidationError("sketch_dir is required")

        spec_path = self._resolve_path(param_spec_file)
        if not spec_path.exists() or not spec_path.is_file():
            raise ValidationError(f"param_spec_file '{param_spec_file}' not found")

        sketch_path = self._resolve_path(sketch_dir)
        if not sketch_path.exists() or not sketch_path.is_dir():
            raise ValidationError(f"sketch_dir '{sketch_dir}' not found")

        config = WebJobConfig(
            experiment_name=experiment_name,
            runs=runs,
            population_size=population_size,
            param_spec_file=str(spec_path),
            sketch_dir=str(sketch_path),
            prompt_text=prompt_text,
            processing=processing,
            screen=self._to_bool(payload.get("screen", False)),
            workers=workers,
            alpha_mode=alpha_mode,
            alpha=alpha_value,
            mutation_rate=mutation_rate,
            mutation_sigma=mutation_sigma,
            parents_compete=parents_compete,
            competing_parents_rate=competing_rate_value,
            k=k,
            ranking_method=ranking_method,
            overwrite=self._to_bool(payload.get("overwrite", False)),
        )
        return config

    def _resolve_path(self, value: str) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        return candidate.resolve()

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return bool(value)

    def _write_job_metadata(self, job: JobRecord) -> None:
        metadata_file = self.jobs_root / f"{job.job_id}.json"
        with open(metadata_file, "w", encoding="utf-8") as file:
            json.dump(job.to_dict(), file, indent=2)
