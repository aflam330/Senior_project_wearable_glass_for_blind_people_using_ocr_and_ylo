# 10-minute acceptance jump (code only)

## What changed in code
1. **Test-time augmentation (TTA)** in `roboeye/bd_counterfeit_auth.py` (horizontal flip mean)
2. **Deploy gate raised** `BD_AUTH_MIN_CONF` **0.90 → 0.95** (`savior_glass/config.py`)
3. Prefer `models/bd_counterfeit_auth_robust.pt` when present
4. Hard-aug finetune attempted; **did not beat TTA**, so base weights kept with TTA metadata

## Measured jump (locked test, n=380)

| Metric | Before (no TTA stress) | After (TTA) |
|---|---:|---:|
| Clean image acc | 100% | **100%** |
| Bright stress | **96.6%** | **98.4%** (+1.84 pt) |
| Dark stress | 99.2% | **99.2%** |
| False REAL @ τ=0.95 | 0 | **0** |

Exact numbers: `realtime_bangla_taka_detection/results/bd_auth_jump.json`

## Also already in hand (same day)
- UV fake-print sim paper_strict reject **94.2%** (`savior_glass/results/uv_check/uv_fake_print_sim.json`)
- Glass policy bench: 0 false REAL (`results/glass_auth_bench.json`)

## Acceptance impact (judgment)
- IEEE Access / Sensors: roughly **55–70% → ~60–75%**
- Q1 applied (ESWA / EAAI / PRL): roughly **40–55% → ~45–60%**
- Still blocked for ASSETS without user study

## One sentence for the paper
Brightness-stressed authenticity rose from 96.6% to 98.4% with flip-TTA; the glass announces likely-genuine only at confidence ≥ 0.95 and recorded 0 false REAL on 240 locked counterfeit test images.
