# expression: context-conditioned perturbation search

Rank supported compounds by how well their **predicted single-cell response distributions**
reach a specified transcriptional goal in a fixed cell-line context, using
[Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M). A conditional distribution
model (JAX/Equinox, FlowJAX spline flow) is the engine; retrospective inverse search against
strong baselines is the benchmark.

- Design brief: [`cellular_response_prototype_brief.md`](cellular_response_prototype_brief.md)
- Plan and milestones: [`ROADMAP.md`](ROADMAP.md)
- Decisions and departures from the brief: [`DECISIONS.md`](DECISIONS.md)
- M0 audit: [`reports/data_audit.md`](reports/data_audit.md), [`reports/design_audit.md`](reports/design_audit.md)

## Quick start

```bash
uv sync
uv run pytest
uv run cellresponse audit --config configs/audit.yaml            # downloads ~2.4 GB of metadata
uv run cellresponse audit --config configs/audit.yaml --offline  # re-render from cached artifacts
```

Downloaded and derived data live in `data/` (git-ignored). Reports and manifests are committed.
Set `HF_TOKEN` to avoid anonymous rate limits.

## Status

M0 (data and design audit) is complete. Next: freeze and download the smoke cohort, then the
prototype cohort (see `ROADMAP.md`).
