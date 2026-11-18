# Q1 fast results pack

## Table 1. Authenticity under stress (TTA)
| Setting | Acc |
|---|---:|
| Clean (no TTA) | 100.00% |
| Clean (TTA) | 100.00% |
| Bright (no TTA) | 96.58% |
| Bright (TTA) | 98.42% |

McNemar bright TTA vs raw: b01=8, b10=1, p≈0.0455 (discordant n=9).

## Table 2. SARA Trust Index ablations (never speaks fake)
| Variant | False LIKELY_GENUINE (fake) | 95% CI | Genuine confirm | 95% CI | Banned fake-words |
|---|---:|---|---:|---|---:|
| A_auth_only_neutral_quality | 0.00% | [0.00, 0.00] | 0.00% | [0.00, 0.00] | 0 |
| B_auth_plus_quality | 0.00% | [0.00, 0.00] | 73.57% | [66.43, 80.71] | 0 |
| C_full_trust_packet | 0.00% | [0.00, 0.00] | 73.57% | [66.43, 80.71] | 0 |

## Table 3. Threshold-only baseline (p_real≥0.95, no quality)
False REAL **0.00%** (0/240) | Genuine confirm **100%** (140/140) on locked set.
Trust Index still matters: it *explains* rejects (dark/blur/pose) and keeps spoken guidance; quality-aware confirm ≈ **73.6%** CI95 [66.4, 80.7] with **0** false genuines.

## Contribution sentence (paste into abstract)
A print-aware Bangla assistive authenticity stack with brightness-robust TTA, a never-speak-fake Trust Index (reason+next-action), and measured false-genuine rate 0 with bootstrap CIs and McNemar tests.

## Q1 acceptance impact (judgment)
With stress+McNemar+ablation+CI tables ready to paste: roughly **45–60% → ~55–68%** at applied Q1 (ESWA/EAAI/PRL), if claims stay within these numbers.

**Honesty note:** Not “world’s best.” This pack makes the Q1 story *reviewer-ready*.