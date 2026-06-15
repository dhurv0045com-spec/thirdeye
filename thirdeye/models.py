from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


SCHEMA_VERSION = 1


class FeatureCategory(StrEnum):
    ARCHITECTURE = "architecture"
    TRAINING = "training"
    DATA = "data"
    RUNTIME = "runtime"
    AGENT = "agent"
    MEMORY = "memory"
    PRODUCT = "product"


class ProtocolKind(StrEnum):
    SYSTEM_AUDIT = "system_audit"
    RUNTIME_ABLATION = "runtime_ablation"
    MATCHED_RETRAINING = "matched_retraining"
    MECHANISM_ABLATION = "mechanism_ablation"
    FACTORIAL_SCREENING = "factorial_screening"
    EFFICIENCY_PROFILE = "efficiency_profile"
    REGRESSION_SUITE = "regression_suite"
    OBSERVATIONAL = "observational"

    @property
    def controlled(self) -> bool:
        return self in {
            self.RUNTIME_ABLATION,
            self.MATCHED_RETRAINING,
            self.MECHANISM_ABLATION,
            self.FACTORIAL_SCREENING,
            self.REGRESSION_SUITE,
        }


class EvidenceGrade(StrEnum):
    REGISTERED = "registered"
    ACTIVATION_VERIFIED = "activation_verified"
    OBSERVED = "observed"
    CONTROLLED_SINGLE_RUN = "controlled_single_run"
    REPLICATED = "replicated"
    INTERACTION_TESTED = "interaction_tested"
    PROMOTION_GRADE = "promotion_grade"
    INCONCLUSIVE = "inconclusive"
    REGRESSED = "regressed"


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"
    TARGET = "target"


@dataclass(frozen=True)
class ProjectSpec:
    project_id: str
    name: str
    description: str = ""
    privacy_mode: str = "aggregate"
    compute_budget_minutes: float = 0.0
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureVariant:
    variant_id: str
    label: str
    config: dict[str, Any] = field(default_factory=dict)
    is_control: bool = False


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    name: str
    owner: str
    version: str
    intended_behavior: str
    category: FeatureCategory
    variants: tuple[FeatureVariant, ...]
    requires_retraining: bool
    parent_feature_id: str | None = None
    expected_benefits: tuple[str, ...] = ()
    possible_regressions: tuple[str, ...] = ()
    protected_metrics: tuple[str, ...] = ()
    required_scenarios: tuple[str, ...] = ()
    resource_metrics: tuple[str, ...] = ()
    incompatible_features: tuple[str, ...] = ()
    activation_probe: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError("A feature must declare at least one variant.")
        controls = sum(variant.is_control for variant in self.variants)
        if controls != 1:
            raise ValueError("A feature must declare exactly one control variant.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    kind: ProtocolKind
    feature_id: str | None = None
    control_variant: str | None = None
    treatment_variants: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    token_budget: int = 0
    paired: bool = False
    protected_thresholds: dict[str, float] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    project_id: str
    protocol_id: str
    feature_variants: dict[str, str]
    git_commit: str
    dirty_patch_hash: str
    config_hash: str
    dependency_hash: str
    dataset_hashes: dict[str, str]
    tokenizer_hash: str | None
    checkpoint_hash: str | None
    seed: int
    hardware: dict[str, Any]
    runtime: dict[str, Any]
    precision: str
    reproduction_command: str
    created_at: float
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricObservation:
    metric_id: str
    value: float
    direction: MetricDirection
    sample_count: int
    scorer: str
    scorer_version: str
    deterministic: bool
    uncertainty_low: float | None = None
    uncertainty_high: float | None = None
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    project_id: str
    feature_id: str
    protocol_id: str
    run_ids: tuple[str, ...]
    grade: EvidenceGrade
    summary: str
    causal: bool
    effect_sizes: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, tuple[float, float]] = field(default_factory=dict)
    missing_requirements: tuple[str, ...] = ()
    created_at: float = 0.0
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

