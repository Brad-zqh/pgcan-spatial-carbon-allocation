# Public repository scope

This repository distributes the research implementation, figure-generation code,
anonymous author-generated numerical results, and scientific verification records.
It does not distribute manuscript Word files, Word-generation scripts, submission
documents, private collaborator handoff archives, exact evaluation-site lists,
commercial basemaps, or licensed fonts. Software and result-data licences do not
grant redistribution rights for excluded third-party inputs.

The report-material assembly script previously included under `analysis/` has
been removed from the current public release. Local manuscript production remains
separate; the numerical experiment and figure-generation entry points remain in
`analysis/`.

## Version 0.1.1 — 22 October 2026

- Figures 4b and 6b use solid row colours progressing from pink to green.
- Enlarged count-based intersection panels and category-error annotations are
  included in the figure code; heatmap colour bars align with their matrices.
- Figure 13 retains actual per-seed predictions, shared axis transforms, density
  percentiles, and three-decimal metric labels. No artificial scatter is added.
- Saved targets, predictions, benchmark metrics, model equations, and training
  protocols are unchanged by this release.

## Verify public results

```console
python -m pip install -e .
python scripts/recompute_metrics.py
python -m unittest discover -s tests
```

This recomputes 15 configurations from actual anonymous released results. Full
retraining and map regeneration require separately authorized spatial inputs;
the public repository alone does not contain them. See `data/README.md` and
`docs/FRAMEWORK_CODE_AUDIT.md` for evidence and input boundaries.
