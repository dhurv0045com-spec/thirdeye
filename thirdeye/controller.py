from __future__ import annotations

from dataclasses import dataclass

from thirdeye.models import FeatureSpec, ProtocolKind, ProtocolSpec


@dataclass(frozen=True)
class PlannedExperiment:
    priority: int
    reason: str
    protocol: ProtocolSpec


def plan_missing_evidence(
    *,
    features: list[FeatureSpec],
    latest_evidence: dict[str, dict],
    profile: str = "auto",
) -> list[PlannedExperiment]:
    if profile not in {"quick", "standard", "exhaustive", "auto"}:
        raise ValueError(f"Unknown evaluation profile: {profile}")
    plans: list[PlannedExperiment] = []
    seeds = (1301,) if profile == "quick" else (1301, 1302, 1303)
    for feature in features:
        evidence = latest_evidence.get(feature.feature_id)
        grade = str((evidence or {}).get("grade", ""))
        if grade in {"promotion_grade", "replicated"} and profile != "exhaustive":
            continue
        kind = (
            ProtocolKind.MATCHED_RETRAINING
            if feature.requires_retraining
            else ProtocolKind.RUNTIME_ABLATION
        )
        control = next(v.variant_id for v in feature.variants if v.is_control)
        treatments = tuple(v.variant_id for v in feature.variants if not v.is_control)
        priority = 10 if grade == "regressed" else (20 if not evidence else 40)
        plans.append(
            PlannedExperiment(
                priority=priority,
                reason="regression requires confirmation" if grade == "regressed" else
                "feature has no controlled evidence" if not evidence else
                "feature evidence requires replication",
                protocol=ProtocolSpec(
                    protocol_id=f"auto:{feature.feature_id}:{kind.value}",
                    kind=kind,
                    feature_id=feature.feature_id,
                    control_variant=control,
                    treatment_variants=treatments,
                    seeds=seeds,
                    token_budget=10_000 if profile == "quick" else 100_000,
                    paired=not feature.requires_retraining,
                ),
            )
        )
    return sorted(plans, key=lambda plan: (plan.priority, plan.protocol.protocol_id))

