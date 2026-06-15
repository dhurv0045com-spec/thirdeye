# ThirdEye

ThirdEye is an evidence and experiment control plane for models, agents, training
systems, and software products.

It answers:

> What is working, how well is it working, what caused the result, what did it
> cost, and what experiment should run next?

ThirdEye separates observation from causal evidence. Telemetry may identify a
hypothesis, but only controlled protocols can earn causal evidence grades.

## V0.1

- Versioned project, feature, protocol, run, metric, evidence, and decision contracts.
- Git, environment, hardware, input, checkpoint, and reproduction lineage.
- Local SQLite evidence graph and content-addressed artifacts.
- System audit, runtime ablation, matched retraining, mechanism ablation,
  factorial screening, efficiency, and regression protocol types.
- Deterministic evidence grading and autonomous missing-evidence planning.
- Decision scorecard, scientific Markdown report, and machine-readable bundle.
- CLI and optional FastAPI control plane.

## Quick Start

```powershell
python -m pip install -e ".[dev,server]"
thirdeye init --project demo
thirdeye register-feature --project demo --spec examples/feature.json
thirdeye evaluate --project demo --profile auto
thirdeye serve
```

By default, state is written under `.thirdeye/`. Set `THIRDEYE_HOME` to place it
elsewhere.

## Scientific Rule

An observational run can never be reported as causal. ThirdEye records the
protocol, control, treatment, seeds, lineage, activation proof, uncertainty,
and scorer version behind every claim.

