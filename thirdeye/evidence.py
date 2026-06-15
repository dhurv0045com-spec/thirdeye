from __future__ import annotations

import math
import time
import uuid
from statistics import mean, stdev

from thirdeye.models import EvidenceGrade, EvidenceRecord, ProtocolKind, ProtocolSpec


def mean_confidence_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    center = mean(values)
    if len(values) < 2:
        return (center, center)
    half_width = 1.96 * stdev(values) / math.sqrt(len(values))
    return (center - half_width, center + half_width)


def grade_evidence(
    *,
    project_id: str,
    feature_id: str,
    protocol: ProtocolSpec,
    run_ids: list[str],
    activation_verified: bool,
    effects: dict[str, list[float]] | None = None,
    protected_regression: bool = False,
    interaction_tested: bool = False,
    promotion_checks_passed: bool = False,
) -> EvidenceRecord:
    effects = effects or {}
    missing: list[str] = []
    causal = protocol.kind.controlled

    if not activation_verified:
        grade = EvidenceGrade.REGISTERED
        missing.append("activation proof")
    elif protected_regression:
        grade = EvidenceGrade.REGRESSED
    elif protocol.kind == ProtocolKind.OBSERVATIONAL:
        grade = EvidenceGrade.OBSERVED
        causal = False
    elif not protocol.kind.controlled:
        grade = EvidenceGrade.ACTIVATION_VERIFIED
        causal = False
    elif interaction_tested:
        grade = EvidenceGrade.INTERACTION_TESTED
    elif len(set(protocol.seeds)) >= 3 and len(run_ids) >= 3:
        grade = (
            EvidenceGrade.PROMOTION_GRADE
            if promotion_checks_passed
            else EvidenceGrade.REPLICATED
        )
    elif run_ids:
        grade = EvidenceGrade.CONTROLLED_SINGLE_RUN
        missing.append("three-seed replication")
    else:
        grade = EvidenceGrade.INCONCLUSIVE
        missing.append("completed controlled run")

    effect_sizes = {metric: mean(values) for metric, values in effects.items() if values}
    confidence = {
        metric: mean_confidence_interval(values)
        for metric, values in effects.items()
        if values
    }
    return EvidenceRecord(
        evidence_id=str(uuid.uuid4()),
        project_id=project_id,
        feature_id=feature_id,
        protocol_id=protocol.protocol_id,
        run_ids=tuple(run_ids),
        grade=grade,
        summary=f"{feature_id}: {grade.value}",
        causal=causal,
        effect_sizes=effect_sizes,
        confidence=confidence,
        missing_requirements=tuple(missing),
        created_at=time.time(),
    )

