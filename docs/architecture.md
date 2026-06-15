# Architecture

ThirdEye uses an append-oriented evidence graph:

```text
Project -> Feature -> Variant -> Protocol -> Run -> Metric -> Evidence -> Decision
```

The Python SDK owns stable contracts. Storage, APIs, dashboards, benchmark
adapters, and remote workers consume those contracts rather than redefining
them.

## Trust Boundaries

- Telemetry is observational.
- Activation probes prove that a treatment executed.
- Controlled protocols support causal claims.
- Replication requires independent seeds.
- Promotion requires project-defined protected gates and human approval.

## Local First

V0.1 uses SQLite and content-addressed local artifacts. Server storage and
remote worker scheduling will preserve the same public contracts.

