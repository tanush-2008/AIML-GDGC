# Optional experiments

These scripts preserve alternatives considered during model development. They are not required to generate the selected 0.84588 blend; run them from the repository root with `python -m experiments.<module_name>` after installing the listed dependencies.

| Module | What it explores | Kaggle result |
|---|---|---:|
| `experiment_domain_adaptation.py` | Compares profile weighting and domain adaptation diagnostics. | Diagnostic only |
| `fit_clipped_domain_candidate.py` | Caps density ratio weights to limit extreme values before blending. | Unscored |
| `fit_consensus_pseudo_candidate.py` | Self training on confident test rows where baseline and domain models agree. | Unscored |
| `fit_lightgbm_candidate.py` | LightGBM with profile aware weighting and port categories. | 0.84368 |
| `fit_port_category_candidate.py` | Treats source/destination ports as categorical features. | 0.84403 |
| `fit_port_category_only_candidate.py` | Treats ports as nominal categories without numeric port values. | No confirmed score |
| `fit_port_target_encoding_candidate.py` | Cross fitted, smoothed target encoding for port/profile features. | No confirmed score |
| `fit_stable_features_candidate.py` | Removes features with strong train/test distribution shift. | No confirmed score |
| `validate_blend_alpha.py` | Compares blend weights on a test profile weighted holdout. | Validation only |

The domain weighted candidate, which scored 0.84347, is produced by `fit_domain_candidate.py` in the repository root. That module also provides helpers imported by the selected blend. The final blend remains the best confirmed result in the recorded submission history.

To run the LightGBM experiment, install its optional dependency set first:

```powershell
python -m pip install -r experiments/requirements_lightgbm.txt
python -m experiments.fit_lightgbm_candidate
```

The other experiments use the main `requirements.txt`. They need the same official CSV files in `data/raw/`; generated unscored predictions and reports are kept locally under `outputs/` and excluded from Git.
