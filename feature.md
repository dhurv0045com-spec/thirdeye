# ThirdEye Feature Map and Remaining Vision

This document is the product checklist for ThirdEye. It answers two questions:

1. What can ThirdEye do today?
2. What remains before it fully reaches the original vision?

The current release is `v0.4.0`.

## Product Goal

ThirdEye exists to make research iteration trustworthy.

It should let a team ask:

```text
What is working?
How well is it working?
What changed?
What did it cost?
Which subsystem moved?
Which feature regressed?
What evidence is missing?
What experiment should run next?
```

For ML, the deeper goal is to observe a model as a system, not just as one loss
curve. ThirdEye therefore measures features, protocols, runs, metrics,
subsystems, training dynamics, held-out capability, uncertainty, and lineage.

## Features Available Now

### 1. Universal Project Evidence

Status: available.

ThirdEye can onboard software, AI, ML, agent, and hybrid projects through a
portable `thirdeye.json` manifest.

Current capabilities:

- Project registration.
- Build, test, train, evaluate, inference, serve, and runtime lifecycle phases.
- Command execution through a generic adapter.
- Python metric emission through stdout.
- Python lifecycle instrumentation.
- Optional FastAPI control plane.
- Local SQLite evidence store.
- Content-addressed artifact storage.

Why this matters:

ThirdEye is not locked to AN-RA or PyTorch. Any product can start by declaring
commands and emitting metrics.

### 2. Evidence Graph

Status: available.

ThirdEye stores versioned records for:

- Project.
- Feature.
- Feature variant.
- Protocol.
- Run.
- Metric.
- Evidence.
- Artifact.
- Subsystem.
- Intelligence signal.
- Capability target.
- Intelligence estimate.

Why this matters:

The graph makes every displayed number traceable. A score without lineage is
not treated as real evidence.

### 3. Feature Contracts

Status: available.

A feature can declare:

- Stable ID.
- Owner.
- Version.
- Intended behavior.
- Category.
- Control variant.
- Treatment variants.
- Parent feature.
- Whether changing it requires retraining.
- Expected benefits.
- Possible regressions.
- Protected metrics.
- Required scenarios.
- Resource metrics.
- Incompatibilities.
- Activation probe.

Why this matters:

ThirdEye can distinguish "feature exists in the codebase" from "feature was
actually tested with a valid control."

### 4. Scientific Protocol Types

Status: available as contracts and grading logic.

Supported protocol kinds:

- `system_audit`
- `runtime_ablation`
- `matched_retraining`
- `mechanism_ablation`
- `factorial_screening`
- `efficiency_profile`
- `regression_suite`
- `observational`

Why this matters:

Telemetry can suggest a hypothesis, but only controlled protocols can produce
causal evidence.

### 5. Evidence Grades

Status: available.

ThirdEye grades feature evidence as:

- `registered`
- `activation_verified`
- `observed`
- `controlled_single_run`
- `replicated`
- `interaction_tested`
- `promotion_grade`
- `inconclusive`
- `regressed`

Important rule:

Observational data never receives a causal grade.

### 6. Lineage Capture

Status: available.

Each run can record:

- Git commit.
- Dirty patch hash.
- Config hash.
- Dependency hash.
- Dataset hashes.
- Tokenizer hash.
- Checkpoint hash.
- Seed.
- Hardware.
- Runtime.
- Precision.
- Reproduction command.

Why this matters:

Two experiments cannot be compared honestly if code, data, checkpoint,
tokenizer, or scoring changed silently.

### 7. Model Intelligence Telemetry

Status: available.

ThirdEye can observe model subsystems during training with sampled,
non-mutating PyTorch hooks.

Current signal families:

- Optimization.
- Representation.
- Behavior.
- Efficiency.
- Reliability.

Current sampled subsystem signals:

- Activation mean.
- Activation RMS.
- Activation standard deviation.
- Activation sparsity.
- Activation saturation.
- Parameter norm.
- Gradient norm.
- Estimated update ratio.
- Training loss.
- Learning rate.
- Tokens per second.

Important boundary:

These are diagnostic signals. They are not automatically used as a training
loss.

### 8. Intelligence Signal Vector

Status: available.

ThirdEye converts raw signals into checkpoint vectors with:

- Latest value.
- Mean value.
- Stability.
- Trend.

Lower-is-better signals are sign-oriented before entering the vector, so
improvement points in a consistent direction.

### 9. Calibrated Capability Estimate

Status: available after enough comparable checkpoints.

ThirdEye fits telemetry to held-out capability targets after the project has
enough evaluated checkpoints.

Current minimum:

```text
5 comparable checkpoints with telemetry and held-out target scores
```

Output:

- Overall predicted capability for the configured target.
- Confidence.
- Calibration error.
- Subsystem predictive contributions.
- Explicit limitations.

Important boundary:

This predicts the configured held-out capability target. It is not universal
intelligence and not causal attribution.

### 10. Training Insight Engine

Status: available.

ThirdEye turns raw telemetry into a bounded explanation of what is visible in
the training process. It can surface:

- Loss improvement, rise, stability, or plateau.
- Training-versus-held-out divergence.
- Gradient health and instability.
- Learning-rate transitions.
- Throughput regressions.
- Token utilization and padding waste.
- Subsystem saturation and representation/update drift.
- Missing calibration for a capability estimate.

Every finding includes a measured basis, confidence, limitations, and a
recommended next measurement or controlled experiment. Findings are always
observational; they never claim that a signal caused the result.

### 11. Reports

Status: available.

Every evaluation can produce:

- `decision-scorecard.md`
- `scientific-report.md`
- `evidence-bundle.json`
- `decision-dashboard.html`
- `intelligence.json` for deep training telemetry.

Why this matters:

Researchers get both human-readable reports and machine-readable evidence.

### 12. CLI

Status: available.

Current commands include:

- `thirdeye init`
- `thirdeye onboard`
- `thirdeye register-project`
- `thirdeye register-feature`
- `thirdeye run`
- `thirdeye assess`
- `thirdeye evaluate`
- `thirdeye inspect`
- `thirdeye intelligence`
- `thirdeye explain`
- `thirdeye calibrate-intelligence`
- `thirdeye serve`

### 13. API and Dashboard

Status: basic available.

Current FastAPI endpoints include:

- Health check.
- Project creation.
- Project listing.
- Project snapshot.
- Feature listing.
- Project evaluation.
- Project assessment when explicitly enabled.
- Intelligence snapshot.
- Training insight endpoint.

Current dashboard:

- Lists projects.
- Shows runs and reports.
- Shows latest metrics.
- Shows feature evidence.
- Shows intelligence telemetry summary.
- Shows the measured training explanation, confidence, and next action.

### 14. AN-RA Reference Integration

Status: available locally in AN-RA integration.

AN-RA currently registers:

- ESV.
- RIM.
- DSTP.
- MoD.
- HAL.
- HAL attention temperature.
- HAL RLVR feedback.
- HAL memory threshold.
- HAL Ouroboros weighting.
- Optimizer selection.
- Data mixture.
- RLVR.
- Memory.
- Verification.
- Cognition.
- Inference efficiency.

AN-RA subsystem telemetry covers:

- Embeddings.
- Attention.
- MLP.
- Normalization.
- ESV.
- RIM.
- MoD.
- HAL.
- Cognition.
- LM head.

The 27,019,999-parameter AN-RA reference model is verified by tests.

## What Remains To Fully Achieve The Vision

The foundation is real, but the complete vision is larger. The remaining work is
listed by product layer.

## Layer 1: Stronger Experiment Execution

Status: partially built.

What exists:

- Protocol contracts.
- Evidence grading.
- Missing-evidence planning.
- Local command execution.
- AN-RA one-click evidence report.

Remaining:

- Automatic execution of matched retraining campaigns.
- Automatic runtime ablation runner.
- Mechanism-ablation runner.
- Factorial-screening runner.
- Regression-suite runner with protected thresholds.
- Experiment budget scheduler.
- Resume-aware trial orchestration.
- Trial deduplication.
- Campaign state machine.

Definition of done:

ThirdEye can take a feature contract and a project budget, generate a valid
campaign, run the trials, resume interrupted jobs, and write evidence records
without manual bookkeeping.

## Layer 2: Remote Workers

Status: not built for v0.4.

Remaining:

- Worker process.
- Database-backed leases.
- Heartbeats.
- Expired-lease recovery.
- Idempotent job claims.
- Remote artifact upload.
- Per-project resource budgets.
- T4 and larger GPU worker profiles.
- Retry policies.
- Interrupted-run resume.

Definition of done:

A local controller can dispatch trials to one or more machines and recover from
worker interruption without corrupting evidence.

## Layer 3: Server Storage

Status: local SQLite built.

Remaining:

- PostgreSQL backend.
- S3-compatible artifact store.
- Alembic migrations.
- Server-side retention policy.
- Backup and restore.
- Multi-user project isolation.
- Access control.

Definition of done:

ThirdEye can run as a team service while preserving the same public contracts as
local mode.

## Layer 4: Professional Dashboard

Status: basic HTML dashboard built.

Remaining:

- Full React dashboard.
- Campaign timeline.
- Feature evidence graph view.
- Subsystem heatmaps.
- Calibration curve view.
- Metric trend charts.
- Pareto frontier.
- Run comparison tables.
- Report drill-down from every number to run, metric, scorer, lineage, and
  artifact.
- Human review queues.

Definition of done:

A researcher can open the dashboard and understand the state of a campaign
without reading JSON.

## Layer 5: Evaluation Ecosystem Adapters

Status: generic command, Python, and PyTorch support built.

Remaining:

- Inspect AI adapter.
- EleutherAI LM Evaluation Harness adapter.
- Pytest adapter with richer test metadata.
- Human-review adapter.
- Audited model-judge adapter.
- MLflow import/export.
- Weights and Biases import/export.
- Hugging Face evaluation metadata export.

Definition of done:

ThirdEye can reuse established benchmark ecosystems instead of forcing teams to
rewrite evaluations.

## Layer 6: Better Statistics

Status: basic uncertainty and ridge calibration built.

Remaining:

- Bootstrap confidence intervals for benchmark metrics.
- Paired-test analysis.
- Multiple-comparison warnings.
- Effect-size reporting by task family.
- Calibration validation split.
- Calibration drift detection.
- fANOVA or PED-ANOVA factor importance.
- Clear labeling of model-based importance versus causal proof.
- Power analysis for recommended replication.

Definition of done:

ThirdEye tells researchers not only what won, but how much confidence the
experiment deserves.

## Layer 7: Autonomous Controller

Status: deterministic missing-evidence planner exists.

Remaining:

- Regression-first campaign policy.
- Low-budget screening policy.
- Early stopping for protected-metric violations.
- Promotion through increasing token budgets.
- Replication-before-confidence policy.
- ASHA or Hyperband allocation.
- Multi-objective search.
- Pareto frontier reporting.
- Manual approval gates for production promotion.

Definition of done:

ThirdEye can propose and run the smallest safe experiment campaign needed to
reduce uncertainty, while never automatically promoting a production model.

## Layer 8: Intelligence-Loss Research

Status: diagnostic telemetry built.

Remaining:

- Validate which signal vectors predict downstream capability across many
  checkpoints.
- Compare telemetry-derived proxies against held-out benchmarks.
- Identify proxy failure modes.
- Build optional auxiliary losses from validated signals only.
- Prevent reward hacking and proxy overfitting.
- Test whether subsystem-level auxiliary objectives improve generalization.
- Keep protected metrics as hard constraints.

Definition of done:

ThirdEye can offer optional, validated auxiliary objectives that improve
training without replacing held-out evaluation.

Important warning:

Do not backpropagate a new "intelligence loss" merely because it looks elegant.
First prove that it predicts and improves held-out behavior without damaging
protected metrics.

## Layer 9: Privacy and Security

Status: privacy-first local defaults built.

Remaining:

- Encrypted raw artifact retention.
- Prompt and output redaction policies.
- Secret scanning.
- Per-field retention classes.
- Project-level privacy profiles.
- Audit logs.
- Team access control.

Definition of done:

Sensitive projects can use ThirdEye without accidentally retaining prompts,
outputs, credentials, or private benchmarks.

## Layer 10: Documentation and Examples

Status: README, integration guide, walkthrough, and sample project exist.

Remaining:

- Generic PyTorch example project.
- Generic CLI-only software example.
- AN-RA public reference report.
- HAL off/on matched experiment example.
- Mechanism-ablation tutorial.
- Dashboard tutorial.
- API client examples.
- Contribution guide for third-party adapters.

Definition of done:

A new research team can integrate ThirdEye in one hour and understand how to
interpret the output.

## Roadmap

### V0.4: Campaign Execution

Focus:

- Local campaign runner.
- Runtime ablation execution.
- Matched retraining campaign templates.
- Regression-suite runner.
- Budget and resume policy.
- Richer AN-RA HAL experiment scripts.

Goal:

Make "run the next valid experiment" more automatic.

### V0.5: Remote Workers and Server Mode

Focus:

- Worker leases.
- Heartbeats.
- PostgreSQL.
- S3-compatible storage.
- Remote T4 worker profile.
- Artifact integrity checks.

Goal:

Move from local evidence tracking to team-scale experiment execution.

### V0.6: Evaluation Adapters

Focus:

- Inspect AI.
- LM Evaluation Harness.
- Pytest metadata.
- Human review queues.
- MLflow and W&B import/export.

Goal:

Connect ThirdEye to the benchmark tools researchers already trust.

### V0.7: Advanced Intelligence Analysis

Focus:

- Calibration validation.
- Drift detection.
- Task-family contribution analysis.
- Factor importance.
- Interaction analysis.
- Optional telemetry-derived auxiliary losses for research only.

Goal:

Turn subsystem telemetry into stronger scientific insight.

### V1.0: Stable Public Platform

Focus:

- Stable plugin API.
- Stable database schema.
- Production dashboard.
- Security and privacy hardening.
- Public reproducible AN-RA reference study.

Goal:

ThirdEye becomes a reliable open-source evidence and experiment platform.

## Priority For AN-RA Right Now

For the next AN-RA tuning cycle, the most important remaining work is:

1. Run several comparable 27M smoke sessions to verify telemetry and reports.
2. Run several comparable frontier-scale sessions to calibrate against compact
   eval.
3. Create a HAL off/on matched retraining protocol.
4. Create HAL mechanism ablations.
5. Add protected thresholds for identity, safety, verification, memory, latency,
   and stability.
6. Publish the first complete AN-RA evidence report.

## Final Status

ThirdEye is ready to measure the next AN-RA tuning run.

ThirdEye is not yet finished as the full autonomous research platform. The next
big leap is campaign execution: automatically running the controlled experiments
that the evidence planner already knows are missing.
