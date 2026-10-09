# Data location

The reproducible archive does not duplicate the 99 MB source CSV. Place an authorized NGSIM CSV here or pass its path explicitly:

```bash
python main.py --input data/i80.csv --lane 3
```

The completed run in `results/` was generated from the adjacent classroom dataset `NGSIM_I80_20050413_1600-1615_SI.csv`. Its byte size and SHA-256 are recorded in `results/processed/preprocessing_summary.json`.
