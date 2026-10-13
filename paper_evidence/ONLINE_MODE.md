# Online description mode: status (2026-10-13)

The mode is not measured. Both routes that were tried on 2026-10-13 are blocked, for reasons outside the code.

## Route A: cloud model with the key in `savior_glass/.env`

Command: `python scripts/eval_online_mode.py --limit 12` (12 text-reading images, same prompt and scoring as the offline pipelines).

Result: the first call was refused.

```
HTTP 429  "You have no credits remaining. Add credits to continue using the API"  code: credit_balance_exhausted
```

Nothing was written. The code path itself reached the provider and got a well-formed answer, so the request format and the key are accepted; the account has no credit.

**Needed:** credit on the account. Then `python scripts/eval_online_mode.py --limit 12`, and the full run (144 images) without `--limit`.

## Route B: a small vision-language model run on the laptop

Script written: `savior_glass/scripts/eval_local_vlm.py` (SmolVLM-500M-Instruct by default; 12 validation images with `--check`, then the 144 test images once).

Result: it could not start. Windows Application Control on this laptop blocks a compiled file of SciPy, which the model-loading library imports:

```
ImportError: DLL load failed while importing _min_spanning_tree: An Application Control policy has blocked this file.
```

The same policy blocks scikit-learn at the moment. It is a security setting of the laptop and was not worked around.

**Needed:** the policy to allow the Python packages (it has cleared by itself before), or another machine. Then `python scripts/eval_local_vlm.py --check`.

## What is not claimed

No latency, accuracy or hallucination rate exists for the online mode or for a local model. A 500-million-parameter model is unlikely to read Bangla; that is an expectation, not a measurement.
