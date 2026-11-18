# Strength boost (code-only, 2026-11-18)

No new real counterfeit-under-UV photographs. Added **honest stress metrics**, **glass-safe policy bench**, and **UV simulation on real counterfeit print**.

## Scripts → outputs

| Script | Output |
|---|---|
| `realtime_bangla_taka_detection/scripts/eval/eval_bd_auth_strength.py` | `results/bd_auth_strength.json` + `.md` |
| `realtime_bangla_taka_detection/scripts/eval/eval_glass_auth_bench.py` | `results/glass_auth_bench.json` |
| `savior_glass/scripts/uv_fake_print_sim.py` | `savior_glass/results/uv_check/uv_fake_print_sim.json` |

## Measured numbers (cite these)

### BD authenticity (`bd_counterfeit_auth.pt`, locked test n=380)
| Setting | Image acc | Note acc | Fake recall | Real recall |
|---|---:|---:|---:|---:|
| Clean | **100%** | 100% | 100% | 100% |
| Blur / JPEG / noise / crop / rotate | 100% | 100% | 100% | 100% |
| Dark | 98.7% | 98.7% | 100% | 96.4% |
| **Bright (worst)** | **96.6%** | 97.4% | 99.2% | 92.1% |

- Median latency (CUDA): ~58–66 ms
- Clean ECE ≈ 0.0004
- Glass gate τ=0.90: **0 false REAL on 240 fakes**; genuine confirm 100% of 140 reals
- Scratch baselines (2 epochs, no ImageNet): MobileNetV3 36.8% (always-real), ResNet18 63.2% (always-fake) → pretrained EffNet is not a trivial fit

**Paper wording:** do not claim field 95% from clean alone; report clean 100% and worst-stress bright 96.6%, with τ=0.90 safety.

### Glass policy (`glass_auth_bench.json`)
- Policy: REAL only if `p_real ≥ τ`, else check-by-hand (never “counterfeit”)
- At τ ∈ {0.80, 0.90, 0.95, 0.99}: **false_REAL = 0**; genuine_confirm = 100%; median ~88 ms

### UV without real fake UV photos (`uv_fake_print_sim.json`)
- Genuine UV accept: **85.6%** (95/111) — same phone set as before
- **120 real counterfeit prints** (`BD_Counterfeit`) under simulated UV lamp model
- **paper_strict** (min_fibres=6, colour=0.22):
  - base reject **94.2%** (7/120 false accept)
  - glow reject **90.0%**
  - noisy / harsh JPEG degrade (noise creates false fibres) → report as limitation
- deploy_default (min_fibres=1) is too weak on sim fakes → keep glass cautious; cite paper_strict in writing

**Caveat (required):** simulated lamp on counterfeit print ≠ photograph of a counterfeit under a real UV lamp.

## Impact on publication strength
- Replaces “100% and done” with a **stress + safety table**
- UV gap becomes a **measured proxy experiment**, not silence
- Overall readiness moves roughly **58% → ~62–65%** for Access/Q1 applied tracks (still need user study for ASSETS)
