# Claim value audit (2026-09-29)

`validate_claims.py` used to check only that each claim's source file exists. It now also checks the value when a claim has `json_key`. This pass attached `json_key` where the claimed value occurs in the claim's own JSON artifact (`scripts/eval/attach_claim_keys.py`).

- value now checked: 106 (newly attached 106)
- value not found in its artifact: 1
- value found at several unrelated keys (left unchecked): 4
- artifact not JSON (markdown, CSV, weights): 0
- value not numeric: 25

## Value not found in its own artifact

These are not proven wrong: the artifact may store the number in another form (a percentage, a count, a rounded value). Each needs a look before it is cited.

| claim | value | artifact |
|---|---|---|
| C_CAL_PRMVT_HER | 0.03284737719849755 | `E:\Final SP\realtime_bangla_taka_detection\results\calibration\suite_seed42.json` |
