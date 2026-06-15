# ThirdEye Walkthrough

This document explains ThirdEye from first principles: what problem it solves,
how the repository is organized, what happens during training, how evidence is
stored, how the intelligence estimate is produced, how AN-RA uses it, and what
ThirdEye can and cannot conclude.

ThirdEye asks:

> What is working, how well is it working, what changed, what did it cost, and
> what experiment should run next?

The short mental model is:

```text
your product or model
        |
        v
instrumentation and evaluation
        |
        v
versioned observations with lineage
        |
        v
evidence grades and calibrated estimates
        |
        v
reports, comparisons, and next experiments
```

ThirdEye is the mask over the face of the product: it can observe the product at
build, test, training, evaluation, inference, serving, and runtime boundaries.
For ML systems it can also run inside the model, sampling registered subsystems
without changing their outputs or gradients.

## The Scientific Contract

ThirdEye deliberately separates three questions.

1. **What happened?**

   Telemetry answers this. Examples include loss, gradient norm, activation
   saturation, throughput, benchmark score, and memory usage.

2. **What predicts capability?**

   Calibration answers this. ThirdEye learns a relationship between internal
   telemetry and a held-out behavioral evaluation across multiple checkpoints.

3. **What caused the improvement?**

   Controlled experiments answer this. A feature only receives causal evidence
   after an appropriate runtime ablation, matched retraining experiment,
   mechanism ablation, factorial screen, or regression protocol.

These are not interchangeable. A subsystem can correlate with a better model
without causing the improvement. A lower training loss can coexist with worse
held-out capability. A feature can be configured but never actually execute.
ThirdEye keeps those distinctions visible.

## Readiness Flag

### Green: ready for the next AN-RA tuning run

The current integration can:

- Observe AN-RA during training with sampled, non-mutating PyTorch hooks.
- Record loss, learning rate, global gradient norm, estimated update ratio, and
  tokens per second.
- Observe embeddings, attention, MLPs, normalization, ESV, RIM, MoD, HAL,
  cognition, and the language-model head when those modules are present.
- Record activation mean, RMS, standard deviation, sparsity, and saturation.
- Record subsystem parameter norm, gradient norm, and estimated update ratio.
- Attach telemetry to a stable checkpoint ID.
- Run AN-RA's compact held-out evaluation after training.
- Store checkpoint telemetry and capability targets across training sessions.
- Produce an overall calibrated capability estimate after at least five
  comparable evaluated checkpoints.
- Produce subsystem contribution estimates and clearly label them predictive,
  not causal.
- Generate a scorecard, scientific report, evidence bundle, HTML dashboard, and
  intelligence JSON report.
- Inventory AN-RA features and identify missing controlled evidence.

### Yellow: requires an experiment design

ThirdEye does not call a feature causal merely because its subsystem signals
look strong. To answer "did HAL cause this improvement?" run HAL off/on with
matched initialization, data order, token budget, seed, precision, and
evaluation protocol. Replicate with at least three seeds for a replicated
evidence grade.

### Not yet part of v0.3

The following parts of the long-term platform vision are not complete:

- Distributed worker leasing and remote campaign scheduling.
- PostgreSQL and S3 server mode.
- ASHA, Hyperband, and multi-objective search.
- Automatic execution of every recommended experiment.
- Inspect AI and LM Evaluation Harness adapters.
- A production React application.
- A validated differentiable "intelligence loss" used for backpropagation.

ThirdEye v0.3 is ready as the measurement and evidence foundation for tuning.
It is not yet the complete autonomous research laboratory described by the
full roadmap.

## Why This Is Not One More Loss Function

Cross-entropy loss measures next-token prediction error on a particular batch.
It does not directly measure reasoning, identity retention, verification,
memory, calibration, safety, latency, or generalization.

ThirdEye therefore builds an **Intelligence Signal Vector** instead of declaring
one raw number to be intelligence.

For each registered signal it can summarize:

```text
latest value
mean value
stability = 1 / (1 + population standard deviation)
linear trend over sampled observations
```

Lower-is-better signals are sign-oriented before entering the vector. For
example, loss `1.0` becomes oriented value `-1.0`, so an improving loss trend
moves in the positive direction.

The vector is diagnostic by default. It is not added to the training objective.
This avoids teaching the model to optimize a weak proxy while appearing to
become more intelligent.

After enough checkpoints have both telemetry and a held-out capability score,
ThirdEye fits a ridge regression:

```text
held-out capability ~= intercept + telemetry_vector * coefficients
```

The result is a **Calibrated Capability Estimate**, not universal intelligence.
Its meaning is exactly:

> Given previous checkpoints from this project, how well does this checkpoint's
> telemetry predict the configured held-out evaluation target?

## Repository Tree

```text
thirdeye/
|-- README.md                    Short project introduction
|-- walkthrough.md               This complete operating guide
|-- pyproject.toml               Package metadata and CLI entry points
|-- docs/
|   |-- architecture.md          Concise architecture summary
|   `-- integration-guide.md     Progressive integration examples
|-- examples/
|   |-- feature.json             Example feature contract
|   `-- sample_project/          Generic command-based project example
|-- tests/
|   |-- test_core.py             Evidence, lineage, and report tests
|   |-- test_integration.py      Runner, API, and lifecycle tests
|   `-- test_intelligence.py     Calibration and hook-invariance tests
`-- thirdeye/
    |-- __init__.py              Public SDK exports
    |-- models.py                Versioned data contracts
    |-- sdk.py                   Main ThirdEye facade
    |-- store.py                 SQLite records and artifact storage
    |-- hashing.py               Stable content hashing
    |-- lineage.py               Code, config, data, and environment lineage
    |-- evidence.py              Evidence grading rules
    |-- controller.py            Missing-evidence experiment planner
    |-- reports.py               JSON, Markdown, and HTML reports
    |-- cli.py                   thirdeye/evidence commands
    |-- api.py                   Optional FastAPI control plane
    |-- discovery.py             Project discovery
    |-- manifest.py              thirdeye.json loading and saving
    |-- runner.py                Lifecycle command execution
    |-- runtime.py               Python lifecycle instrumentation
    |-- integration.py           Metric emission over stdout
    |-- plugins.py               Third-party adapter discovery
    |-- adapters/
    |   |-- base.py              Adapter contracts and registry
    |   |-- command.py           Language-neutral command adapter
    |   `-- pytorch.py           Basic model and runtime profiling
    `-- intelligence/
        |-- signals.py           Signal capture and vector construction
        |-- pytorch.py           Deep subsystem hooks
        |-- calibration.py       Capability calibration model
        `-- monitor.py           Checkpoints, targets, persistence, reports
```

## The Evidence Graph

The central product model is:

```text
Project
  -> Feature
    -> Feature Variant
      -> Protocol
        -> Run
          -> Metric
            -> Evidence
              -> Decision
```

Model intelligence telemetry adds:

```text
Project
  -> Subsystem
    -> Intelligence Signal
      -> Checkpoint Signal Vector
        -> Capability Target
          -> Calibrated Estimate
```

### Project

A `ProjectSpec` identifies the product being measured. It includes its stable
project ID, name, root, kind, privacy mode, and optional compute budget.

### Feature

A `FeatureSpec` declares something whose effect matters:

- Stable ID and owner.
- Version and intended behavior.
- Architecture, training, data, runtime, agent, memory, or product category.
- Exactly one control variant.
- One or more treatment variants.
- Whether changing it requires retraining.
- Expected benefits and possible regressions.
- Protected metrics.
- Required scenarios and resource measurements.
- Incompatibilities.
- An activation probe.

Features answer product and experiment questions. Subsystems answer internal
model-diagnostic questions. One concept may appear in both forms. HAL is a
feature when comparing HAL off versus on, and a subsystem when observing its
internal behavior during a run.

### Protocol

A `ProtocolSpec` says how evidence was produced:

- `system_audit`
- `runtime_ablation`
- `matched_retraining`
- `mechanism_ablation`
- `factorial_screening`
- `efficiency_profile`
- `regression_suite`
- `observational`

Only controlled protocol kinds can produce causal evidence.

### Run and Lineage

Every lifecycle run can record:

- Git commit.
- Dirty patch hash.
- Configuration hash.
- Dependency hash.
- Dataset hashes.
- Tokenizer hash.
- Checkpoint hash.
- Seed.
- Hardware and runtime.
- Precision.
- Exact reproduction command.

This is what lets two results be compared without quietly comparing different
data, code, tokenizer, or checkpoints.

### Metric

A `MetricObservation` includes:

- Metric ID and value.
- Higher, lower, or target direction.
- Sample count.
- Scorer and scorer version.
- Determinism.
- Optional uncertainty interval.
- Unit and metadata.

### Evidence

Evidence grades progress through:

```text
registered
activation_verified
observed
controlled_single_run
replicated
interaction_tested
promotion_grade
```

`inconclusive` and `regressed` represent important negative outcomes.

An observational protocol never receives a causal grade. Three independent
seeds and three completed controlled runs are required for `replicated`.

## What Happens During an AN-RA Training Step

AN-RA creates an `ANRAIntelligenceSession` after moving the model to its device.
Telemetry is enabled by default.

At the start of a step:

```python
intelligence_session.begin_step(global_step)
```

The default sampling interval is every 25 optimizer steps. Configure it with:

```powershell
$env:ANRA_THIRDEYE_SAMPLE_EVERY = "10"
```

Disable telemetry only when intentionally measuring a no-instrumentation
baseline:

```powershell
$env:ANRA_THIRDEYE_INTELLIGENCE = "0"
```

During a sampled forward pass, registered module hooks:

1. Find the first tensor in the module output.
2. Detach it from autograd.
3. Convert the detached view to float for stable summary statistics.
4. Record aggregate statistics.
5. Discard the tensor view.

Raw activations, prompts, and outputs are not retained.

After backward and before the optimizer update, AN-RA records:

- Batch loss.
- Learning rate.
- Global gradient norm.
- Subsystem parameter norms.
- Subsystem gradient norms.
- Estimated update ratios.
- Token throughput.

The current subsystem update ratio is:

```text
abs(learning_rate) * gradient_norm / max(parameter_norm, epsilon)
```

This is an optimization-scale diagnostic. For adaptive optimizers such as AdamW
it is not the exact realized parameter delta.

At session completion:

1. AN-RA saves the checkpoint.
2. AN-RA runs its compact held-out evaluation.
3. ThirdEye snapshots the session signal vector under a checkpoint ID containing
   the checkpoint filename and global step.
4. The compact evaluation overall score becomes the capability target.
5. Telemetry, target, and estimate are persisted.
6. ThirdEye attempts cross-checkpoint calibration.
7. Reports are regenerated.

## AN-RA Subsystem Map

| Subsystem ID | Model area | Main diagnostic purpose |
| --- | --- | --- |
| `anra.embeddings` | Token embeddings | Representation scale and learning |
| `anra.attention` | Transformer attention modules | Saturation, scale, gradient flow |
| `anra.mlp` | SwiGLU feed-forward modules | Activity, sparsity, update scale |
| `anra.normalization` | Block and final norms | Representation stability |
| `anra.esv` | Emotional State Vector | Identity-state behavior |
| `anra.rim` | Residual Identity Modulators | Identity-path learning |
| `anra.mod` | Mixture of Depth routers | Routing activity and learning |
| `anra.hal` | Hormonal Analog Layer | Adaptive-state behavior |
| `anra.cognition` | Cognitive extension | Cognitive-path learning |
| `anra.output` | Language-model head | Output scale and saturation |

ESV, RIM, and HAL are marked as protected subsystems. That label is preserved
in the subsystem contract so reports and future policy layers can apply stricter
gates to them.

Signals retain module names in metadata. The subsystem vector aggregates by
subsystem and signal type, while the evidence bundle preserves the raw
module-level observations for deeper inspection.

## First Run Versus Fifth Run

### After the first evaluated checkpoint

ThirdEye provides:

- Complete telemetry collected for that session.
- Training and subsystem signal vectors.
- Held-out compact evaluation score.
- An uncalibrated estimate with `overall: null`.
- Explicit limitation: no fitted post-training calibration.

This is correct behavior. ThirdEye does not invent an intelligence score from
one checkpoint.

### After five comparable evaluated checkpoints

ThirdEye fits the calibration model using signals shared across those
checkpoints. It then provides:

- Predicted held-out capability for the newest checkpoint.
- Calibration mean absolute error.
- Confidence based on calibration sample count and error.
- Per-subsystem predictive contributions.
- Limitations stating that this is not universal intelligence and not causal
  attribution.

More checkpoints and a stronger held-out suite generally make this estimate
more useful. Five is the minimum, not the ideal scientific sample size.

## Running AN-RA With ThirdEye

Install or update the standalone package:

```powershell
python -m pip install -U "thirdeye-evidence @ git+https://github.com/dhurv0045com-spec/thirdeye.git@v0.3.0"
```

Run the AN-RA inventory and evidence-gap report before training:

```powershell
python scripts/evaluate_with_thirdeye.py --profile auto
```

Run the 27M tuning model through the canonical trainer:

```powershell
python scripts/build_brain.py `
  --data_path training_data/anra_training.txt `
  --checkpoint_path anra_v2_brain.pt `
  --model-size 25m `
  --max_minutes 30
```

ThirdEye telemetry is automatically active for this trainer unless
`ANRA_THIRDEYE_INTELLIGENCE=0`.

Inspect the stored intelligence records:

```powershell
thirdeye --home output/v2/thirdeye intelligence --project anra
```

Re-run calibration manually:

```powershell
thirdeye --home output/v2/thirdeye calibrate-intelligence `
  --project anra `
  --target anra.compact_eval.overall `
  --minimum-checkpoints 5
```

Regenerate the full report bundle:

```powershell
thirdeye --home output/v2/thirdeye evaluate --project anra --profile auto
```

## Output Locations

AN-RA writes ThirdEye state beneath:

```text
output/v2/thirdeye/
```

Important outputs are:

```text
output/v2/thirdeye/evidence.sqlite3
output/v2/thirdeye/reports/anra/evidence-bundle.json
output/v2/thirdeye/reports/anra/decision-scorecard.md
output/v2/thirdeye/reports/anra/scientific-report.md
output/v2/thirdeye/reports/anra/decision-dashboard.html
output/v2/thirdeye/reports/anra/intelligence.json
```

### Decision scorecard

Use this first. It summarizes:

- Operational status.
- Latest metrics.
- Feature evidence grades.
- Missing evidence.
- Overall calibrated capability when available.
- Subsystem predictive contributions.
- Number of recommended experiments.

### Scientific report

Use this to understand methods and trace conclusions back to evidence. It
contains feature evidence, runs, metrics, intelligence telemetry, and proposed
experiments.

### Evidence bundle

Use this for programmatic analysis. It is the complete versioned JSON payload
behind the human-readable reports.

### Dashboard

Open `decision-dashboard.html` locally. It displays feature status, metrics,
runs, subsystem count, signal count, capability estimate, and confidence.

### Intelligence JSON

Use this for detailed training diagnostics. It contains subsystem definitions,
all sampled signals from the current session, capability targets, estimates,
and scientific-status labels.

## How To Read Signals

No signal should be interpreted alone.

### Loss decreases, held-out capability increases

This is the desired broad pattern. Check protected metrics, throughput, and
subsystem anomalies before accepting the checkpoint.

### Loss decreases, held-out capability decreases

Possible causes include overfitting, data-distribution mismatch, shortcut
learning, identity degradation, or benchmark instability. Do not promote based
on loss.

### Gradient norm becomes very large

Inspect learning rate, precision, clipping, activation saturation, and the
subsystem gradient distribution. A global spike can originate from one path.

### Gradient norm collapses toward zero

Inspect dead or bypassed modules, saturation, routing behavior, frozen
parameters, optimizer state, and whether the treatment actually activated.

### Activation RMS drifts steadily upward

Inspect normalization, residual scaling, precision, attention temperature, and
learning rate. Compare the same subsystem across matched runs.

### Saturation rises

The subsystem is producing more values beyond the configured absolute
threshold. This may indicate instability, but the meaning depends on the
activation family and architecture.

### Sparsity rises

This can be healthy for routing or sparse mechanisms and unhealthy for a
collapsed dense pathway. Interpret it using subsystem intent.

### One subsystem has a strong calibrated contribution

It means that subsystem's telemetry helped predict the configured held-out
score in the calibration data. It does not prove that changing that subsystem
will improve capability. Use it to generate an ablation hypothesis.

## The Recommended Tuning Loop

1. **Freeze the protocol.**

   Choose dataset version, tokenizer, benchmark version, seeds, token budget,
   precision, hardware class, and protected thresholds.

2. **Run a baseline.**

   Record the current configuration as the accepted comparison point.

3. **Change one feature or a declared factorial set.**

   Avoid changing architecture, data mixture, optimizer, and evaluation at once
   unless the experiment explicitly models those interactions.

4. **Verify activation.**

   A configured feature that never executes is not a treatment.

5. **Train with ThirdEye enabled.**

   Keep the sampling interval fixed across compared runs.

6. **Run held-out evaluations.**

   Use deterministic scoring where possible and version every scorer.

7. **Inspect regressions before aggregate improvement.**

   Identity, safety, verification, memory, latency, and reliability can regress
   while an overall average rises.

8. **Replicate promising changes.**

   Use at least three independent seeds for replicated evidence.

9. **Test mechanisms and interactions.**

   A parent feature can work for a different reason than expected. Disable its
   internal mechanisms separately.

10. **Promote manually.**

    ThirdEye does not automatically promote a production model.

## Generic Integration With Another PyTorch Model

```python
from thirdeye.intelligence import IntelligenceMonitor, PyTorchSubsystemCollector
from thirdeye.models import CapabilityTarget, SubsystemSpec

subsystems = [
    SubsystemSpec(
        subsystem_id="encoder",
        name="Encoder",
        owner="research",
        kind="architecture",
        module_patterns=("encoder.*",),
        expected_signals=("activation.rms", "gradients.norm"),
    ),
    SubsystemSpec(
        subsystem_id="head",
        name="Task Head",
        owner="research",
        kind="behavior",
        module_patterns=("head",),
    ),
]

monitor = IntelligenceMonitor(subsystems)
hooks = PyTorchSubsystemCollector(
    model,
    subsystems,
    monitor.collector,
    sample_every=10,
)

for step, batch in enumerate(loader):
    hooks.begin_step(step)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["input"])
    loss = criterion(output, batch["target"])
    loss.backward()

    subsystem_stats = hooks.capture_gradients(
        learning_rate=optimizer.param_groups[0]["lr"]
    )
    monitor.collector.record_training_step(
        step=step,
        loss=loss.item(),
        learning_rate=optimizer.param_groups[0]["lr"],
        gradient_norm=global_gradient_norm,
    )
    optimizer.step()

estimate = monitor.checkpoint("checkpoint-0001")
monitor.record_capability(
    CapabilityTarget(
        target_id="heldout.accuracy",
        value=heldout_accuracy,
        checkpoint_id="checkpoint-0001",
        evaluator="my-evaluator-v2",
        sample_count=heldout_examples,
    )
)
```

For TensorFlow, JAX, agents, services, or non-Python systems, implement an
adapter that emits the same versioned contracts. The evidence and reporting
layers do not depend on PyTorch.

## Generic Integration With Any Software Product

Create a manifest:

```powershell
thirdeye onboard C:\path\to\product
```

Review `thirdeye.json`, then run:

```powershell
thirdeye register-project --manifest thirdeye.json
thirdeye assess --manifest thirdeye.json --profile auto
```

Any command can emit:

```text
THIRDEYE_METRIC {"metric_id":"quality.score","value":0.91}
```

The command adapter captures status, duration, declared metrics, lineage, and
optional content-addressed logs.

## Storage and Privacy

Local mode uses SQLite and content-addressed files.

The record types include:

- Projects and manifests.
- Features.
- Runs and run results.
- Lifecycle events.
- Evidence.
- Artifacts.
- Subsystems.
- Intelligence signals.
- Capability targets.
- Intelligence estimates.

Raw command output is not retained unless `retain_output` is explicitly enabled.
Deep model telemetry stores numeric aggregates and tensor shapes, not raw
activations. Dataset, tokenizer, checkpoint, and artifact hashes allow lineage
checks without copying their contents into the report.

## Failure Modes and Safeguards

### No overall estimate appears

This is expected before the minimum number of comparable checkpoints have both
telemetry and capability targets. Inspect the raw signal vector and benchmark
result, then continue accumulating checkpoints.

### Calibration says signals are missing

The compared checkpoints do not share the same signal schema. Check subsystem
registration, model structure, sampling configuration, and whether a feature
was absent in some runs.

### A subsystem has no signals

Its module pattern matched no executed module. Confirm the registered pattern,
the model wrapper hierarchy, and whether the subsystem participated in that
forward pass.

### Duplicate-looking activation observations

Gradient checkpointing can execute forward modules again during backward.
Those observations remain non-mutating, but analysis should account for the
recomputation policy when comparing runs.

### The score improves but protected behavior regresses

Treat the run as a regression. A capability average does not override protected
thresholds.

### The dashboard shows "Uncalibrated"

The dashboard is refusing to invent a number. Run enough comparable evaluated
checkpoints and then call `calibrate-intelligence`.

## Validation Performed for v0.3

The test suite verifies:

- Lower-is-better signal orientation.
- Refusal to produce an uncalibrated overall score.
- Recovery of known capability ordering in synthetic experiments.
- Persistence of subsystems, signals, targets, and estimates.
- Calibration across independent training sessions.
- PyTorch hooks do not change model outputs.
- PyTorch hooks do not change gradients.
- Intelligence records appear in API and report flows.
- SQLite connections close correctly on Windows.
- AN-RA's exact 27,019,999-parameter reference model remains recognized.
- AN-RA core subsystems emit deep telemetry.

The v0.3 release passed all ThirdEye tests, AN-RA integration tests, lint, and
Python compilation checks before publication.

## Final Interpretation Rule

Use ThirdEye outputs in this order:

```text
lineage
  -> activation proof
    -> raw metrics and uncertainty
      -> protected regressions
        -> calibrated prediction
          -> controlled causal evidence
            -> human decision
```

Never reverse that order. A polished score without lineage is not evidence. A
correlation without a control is not causality. An average improvement with a
protected regression is not a successful model.

With that discipline, ThirdEye is ready to be the measurement layer around the
next AN-RA tuning campaign.
