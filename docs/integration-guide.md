# Integration Guide

ThirdEye supports progressively deeper integration. A project can begin with a
command and later add model hooks, features, controlled protocols, and custom
adapters without replacing its evidence history.

## Level 1: Any Software

Discover a project:

```powershell
thirdeye onboard C:\path\to\project
thirdeye register-project --manifest C:\path\to\project\thirdeye.json
thirdeye assess --manifest C:\path\to\project\thirdeye.json
```

Edit `thirdeye.json` to declare build, test, evaluation, inference, serving, or
training commands. ThirdEye captures lineage, duration, status, aggregate
metrics, and optionally content-addressed logs.

Emit metrics from any Python command:

```python
from thirdeye import MetricDirection, emit_metric

emit_metric(
    "quality.accuracy",
    0.91,
    direction=MetricDirection.HIGHER,
    sample_count=1000,
    scorer="heldout-v2",
    deterministic=True,
)
```

Other languages can print one JSON object per line prefixed with
`THIRDEYE_METRIC `.

## Level 2: Python Runtime

Instrument a function at any lifecycle point:

```python
from thirdeye import LifecyclePhase, lifecycle

with lifecycle(
    eye,
    project_id="my-project",
    run_id=manifest.run_id,
    phase=LifecyclePhase.INFERENCE,
    name="generate",
) as run:
    output = model.generate(inputs)
    run.metric("quality.accepted", 1, deterministic=True)
```

This produces OpenTelemetry-shaped trace and span identifiers plus metrics.

## Level 3: PyTorch

```python
from thirdeye.adapters import PyTorchAdapter

eye.emit_metrics(run_id, PyTorchAdapter.model_metrics(model))
output, metrics = PyTorchAdapter.profile_callable(
    lambda: model(batch),
    tokens=batch.numel(),
)
eye.emit_metrics(run_id, metrics)
```

PyTorch remains optional; software projects do not install it.

## Level 4: Feature Causality

Register feature variants, activation probes, expected benefits, protected
metrics, and whether a change requires retraining. Use runtime ablations for
same-checkpoint systems and matched retraining for learned features.

Telemetry may suggest a hypothesis. Only a controlled protocol can earn a
causal evidence grade.

## Level 5: Framework Plugin

Publish an adapter under the `thirdeye.adapters` entry-point group:

```toml
[project.entry-points."thirdeye.adapters"]
my-framework = "my_package:build_adapter"
```

Adapters receive a project-root-bounded execution context, run manifest, state
store, and private work directory.

## Privacy

Raw command output is not retained by default. Set `"retain_output": true`
only when project policy permits it. Retained artifacts are content-addressed
and linked to their exact run.

