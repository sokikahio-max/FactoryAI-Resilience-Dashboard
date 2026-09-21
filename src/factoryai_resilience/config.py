from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    seeds: list[int]
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    target: str
    primary_metric: str
    max_models_total: int
    initial_models_recovery_methods: int
    nfolds: int
    exclude_deep_learning_for_reproducibility: bool = True


@dataclass(frozen=True)
class CostPolicy:
    false_negative_cost: float = 5.0
    false_positive_cost: float = 1.0
    # Target fraction of predictions retained after validation-calibrated
    # decision-boundary abstention. The abstention band is learned only
    # from the validation set and then frozen for the untouched test set.
    target_coverage: float = 0.90


@dataclass(frozen=True)
class RecoveryPolicy:
    min_primary_metric_retention: float = 0.85
    max_cost_inflation: float = 1.25
    max_retries: int = 1


@dataclass(frozen=True)
class QualityPolicy:
    moderate_missing_rate: float = 0.10
    severe_missing_rate: float = 0.25
    moderate_standardized_mean_shift: float = 0.50
    severe_standardized_mean_shift: float = 1.00
    moderate_ood_rate: float = 0.20
    severe_ood_rate: float = 0.50
    ood_training_quantile: float = 0.99


@dataclass(frozen=True)
class PlannerSettings:
    mode: str = "mock"
    model: str = "gpt-5.6-luna"
    temperature: float = 0.0
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0


@dataclass(frozen=True)
class BackendSettings:
    name: str = "sklearn_smoke"
    h2o_port: int = 54321
    h2o_max_mem_size: str = "4G"


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    severity: str
    magnitude: float = 0.0
    features: list[str] = field(default_factory=list)

    @property
    def condition_id(self) -> str:
        return self.name if self.severity == "none" else f"{self.name}__{self.severity}"


@dataclass(frozen=True)
class AppConfig:
    experiment: ExperimentSettings
    cost_policy: CostPolicy
    recovery_policy: RecoveryPolicy
    quality_policy: QualityPolicy
    planner: PlannerSettings
    backend: BackendSettings
    perturbations: list[PerturbationSpec]


def _build(cls, payload: dict[str, Any]):
    return cls(**payload)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig(
        experiment=_build(ExperimentSettings, payload["experiment"]),
        cost_policy=_build(CostPolicy, payload["cost_policy"]),
        recovery_policy=_build(RecoveryPolicy, payload["recovery_policy"]),
        quality_policy=_build(QualityPolicy, payload["quality_policy"]),
        planner=_build(PlannerSettings, payload["planner"]),
        backend=_build(BackendSettings, payload["backend"]),
        perturbations=[PerturbationSpec(**item) for item in payload["perturbations"]],
    )
