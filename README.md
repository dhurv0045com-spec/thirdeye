# ThirdEye

ThirdEye is an evidence and experiment control plane for models, agents, training
systems, and software products.

It answers:

> What is working, how well is it working, what caused the result, what did it
> cost, and what experiment should run next?

ThirdEye separates observation from causal evidence. Telemetry may identify a
hypothesis, but only controlled protocols can earn causal evidence grades.

## V0.3

ThirdEye now includes Model Intelligence Telemetry: a framework-neutral way to
measure optimization health, subsystem behavior, representation dynamics,
efficiency, reliability, and held-out capability during and after training.
It never labels raw loss or activation statistics as intelligence. An overall
estimate appears only after calibration against versioned held-out evaluations.

- Versioned project, feature, protocol, run, metric, evidence, and decision contracts.
- Registered model subsystem graph with sampled, non-mutating PyTorch hooks.
- Intelligence Signal Vectors and calibrated checkpoint capability estimates.
- Per-subsystem predictive contributions, uncertainty, and explicit limitations.
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

Read [the complete walkthrough](walkthrough.md) for the repository tree,
scientific model, intelligence telemetry, AN-RA training flow, report
interpretation, readiness boundary, and recommended tuning process.

## Intelligence Telemetry

```python
from thirdeye.intelligence import IntelligenceMonitor, PyTorchSubsystemCollector
from thirdeye.models import SubsystemSpec

subsystems = [
    SubsystemSpec(
        subsystem_id="encoder",
        name="Encoder",
        owner="research",
        kind="architecture",
        module_patterns=("encoder.*",),
    )
]
monitor = IntelligenceMonitor(subsystems)
hooks = PyTorchSubsystemCollector(model, subsystems, monitor.collector, sample_every=10)

hooks.begin_step(step)
loss = training_step()
loss.backward()
hooks.capture_gradients(learning_rate=optimizer.param_groups[0]["lr"])
monitor.collector.record_training_step(
    step=step,
    loss=loss.item(),
    learning_rate=optimizer.param_groups[0]["lr"],
    gradient_norm=global_gradient_norm,
)
```

Call `monitor.checkpoint(checkpoint_id)` during training. Record held-out
`CapabilityTarget` results for several checkpoints to calibrate the estimate.
Telemetry remains diagnostic by default; it is not added to the training
objective unless a project explicitly validates and opts into a proxy loss.
