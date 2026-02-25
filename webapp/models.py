from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WebJobConfig:
    experiment_name: str
    runs: int
    population_size: int
    param_spec_file: str
    sketch_dir: str
    prompt_text: str
    processing: str
    screen: bool
    workers: int
    alpha_mode: str
    alpha: float | None
    mutation_rate: float
    mutation_sigma: float
    parents_compete: bool
    competing_parents_rate: float | None
    k: float
    ranking_method: str
    overwrite: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "job": {
                "experiment_name": self.experiment_name,
                "param_spec_file": self.param_spec_file,
                "sketch_dir": self.sketch_dir,
                "prompt_source": "web_input",
                "processing": self.processing,
                "screen": self.screen,
                "workers": self.workers,
            },
            "evo": {
                "runs": self.runs,
                "population_size": self.population_size,
                "alpha_mode": self.alpha_mode,
                "alpha": self.alpha,
                "mutation_rate": self.mutation_rate,
                "mutation_sigma": self.mutation_sigma,
                "parents_compete": self.parents_compete,
                "competing_parents_rate": self.competing_parents_rate,
                "k": self.k,
                "ranking_method": self.ranking_method,
            },
        }


@dataclass(slots=True)
class JobRecord:
    job_id: str
    config: WebJobConfig
    status: str = "queued"
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"] = self.config.snapshot()
        return payload


@dataclass(slots=True)
class ExperimentView:
    name: str
    path: Path
    runs: list[str]
