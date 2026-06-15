# ThirdEye

ThirdEye is an evidence and experiment control plane for models, agents, training
systems, and software products.

It answers:

> What is working, how well is it working, what caused the result, what did it
> cost, and what experiment should run next?

ThirdEye separates observation from causal evidence. Telemetry may identify a
hypothesis, but only controlled protocols can earn causal evidence grades.

## V0.2

- Versioned project, feature, protocol, run, metric, evidence, and decision contracts.
- Portable `thirdeye.json` onboarding manifest for software, AI, ML, and agent projects.
- Build, test, train, evaluate, inference, serving, and runtime lifecycle execution.
- Generic CLI adapter, Python instrumentation, optional PyTorch profiling, and plugin API.
- Git, environment, hardware, input, checkpoint, and reproduction lineage.
- Local SQLite evidence graph and content-addressed artifacts.
- System audit, runtime ablation, matched retraining, mechanism ablation,
  factorial screening, efficiency, and regression protocol types.
- Deterministic evidence grading and autonomous missing-evidence planning.
- Decision scorecard, scientific report, interactive HTML dashboard, and evidence bundle.
- CLI, one-command assessment, and optional FastAPI control plane.
- Privacy-first aggregate capture; raw logs require explicit opt-in.

## Quick Start

```powershell
python -m pip install -e ".[dev,server]"
thirdeye onboard C:\path\to\your-project
thirdeye assess --manifest C:\path\to\your-project\thirdeye.json
thirdeye serve
```

By default, state is written under `.thirdeye/`. Set `THIRDEYE_HOME` to place it
elsewhere.

## Scientific Rule

An observational run can never be reported as causal. ThirdEye records the
protocol, control, treatment, seeds, lineage, activation proof, uncertainty,
and scorer version behind every claim.

See [the integration guide](docs/integration-guide.md) for command, Python,
PyTorch, feature-ablation, and third-party adapter examples.
