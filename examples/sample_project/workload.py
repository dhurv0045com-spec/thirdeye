from thirdeye import MetricDirection, emit_metric


def main() -> None:
    emit_metric(
        "software.tests_passed",
        12,
        direction=MetricDirection.HIGHER,
        sample_count=12,
        scorer="sample.verifier",
        deterministic=True,
        unit="tests",
    )
    emit_metric(
        "software.failure_rate",
        0.0,
        direction=MetricDirection.LOWER,
        sample_count=12,
        scorer="sample.verifier",
        deterministic=True,
    )


if __name__ == "__main__":
    main()

