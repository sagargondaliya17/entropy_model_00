# Entropy Model Pipeline

## Large `options_daily.csv` workflow (without uploading to GitHub)

If your options dataset is too large for GitHub, keep it local and run simulation with an explicit path.

### 1) Keep the file local
- Put it at `data/options_daily.csv` (already gitignored), **or** any local path.

### 2) Run phase-1 simulation
```bash
python run_entropy_simulation.py --options-path /absolute/path/to/options_daily.csv
```

If `--options-path` is missing/not found, the script still runs in fallback mode (`market+macro`) and prints a note.

### 3) Train phase-2 model
```bash
python train_regime_model.py
```

## Expected options columns
The options CSV must include:
- `Date`
- `iv_atm`
- `iv_change`
- `iv_skew`
- `pcr_oi`
- `oi_concentration`
