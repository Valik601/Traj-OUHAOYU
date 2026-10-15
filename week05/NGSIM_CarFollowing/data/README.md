# Data placement

The pipeline treats the supplied dataset as read-only. By default, `config.yaml`
points to the sibling directory:

```text
../datasets/NGSIM-I80/NGSIM_I80_20050413_1600-1615_SI.csv
../datasets/NGSIM-I80/source_and_checks.json
```

No raw data are copied into or overwritten by this project. To use another SI
file with the same eleven-column schema, update `paths.si_csv` in
`config.yaml`. Values in the SI file are already metres, seconds, metres per
second, and metres per second squared; the pipeline performs no unit conversion.

## What is actually present locally

The locally supplied classroom package contains only the SI file
(`NGSIM_I80_20050413_1600-1615_SI.csv`, 99 340 985 bytes, sha256
`df86a98e…695255`), `使用说明.txt`, and `source_and_checks.json`. There is **no**
imperial `*_raw.csv` in this workspace, so the pipeline neither requires nor
references one. Keeping a single copy of the 99 MB input outside `data/` avoids
duplicating the dataset; this file exists so that `data/` is still self-explanatory.

`source_and_checks.json` declares a `global_time_ms` lower bound of
`1000000000`. That value does **not** appear anywhere in the CSV: the real range
is `1113433200000 … 1113434099900` (2005-04-13 16:00–16:15 PDT). Stage 00 reports
this discrepancy explicitly instead of trusting the metadata; see
`outputs/01_data_quality/dataset_summary.json → source_and_checks_comparison`.
