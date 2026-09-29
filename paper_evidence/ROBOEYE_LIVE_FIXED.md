# roboeye_live.py fixes (2026-09-29)

Desktop HUD: `realtime_bangla_taka_detection/scripts/roboeye_live.py`.

## A. Forward hook + Ultralytics (crash on the first frame)

**Symptom.** The first `DualHeadYOLO.predict()` raised `TypeError: cannot pickle '_thread.lock' object` inside `ultralytics ... setup_model`. Installed Ultralytics is 8.4.164.

**Cause.** `DualHeadYOLO.__init__` registered a forward hook on `detector.model.model[9]` (SPPF) with the bound method `self._save_feat`. Ultralytics deep-copies the model when it sets up its predictor. The copy followed the hook into the `DualHeadYOLO` object and reached a thread lock. Even without the crash, a hook on the original model would not fire on the copy that actually runs, so the authenticity head would have received no features.

**Fix** (`roboeye/dual_head.py`). The hook is installed lazily on the predictor's own network (`detector.predictor.model.model.model[9]`) after the first predict. `close()` removes it.

**Check.** Three calls on `samples/sample_note_10_taka.jpg`. The hook is present and SPPF features of shape (1, 512, 10, 20) are captured on every call. Detection is unchanged: 10_taka at 0.903.

## Grad-CAM (`roboeye/gradcam_live.py`)

**Symptom.** The `g` overlay did nothing: `pytorch_grad_cam` is not installed (`requirements.txt` lists it as optional), so `LiveGradCAM.ok` was False. The old design also hooked the shared live model permanently, turned on gradients for its weights, and swallowed every exception.

**Fix.** A self-contained Grad-CAM on a private copy of the network, with the hook attached only during one call. The live model's weights stay frozen (`requires_grad` False after use), and its predictions are identical before and after a heat-map call (0.903).

**Throttling.** The HUD recomputes the heat map every 15 frames (`CAM_EVERY`), because one call takes about 1 s here.

**Check.** A heat map was produced for class 10_taka. Example overlay: `realtime_bangla_taka_detection/results/gradcam_check.jpg`. Heat concentrates on the emblem, the "TEN TAKA" lettering and the "10" numeral.

## B. CLIP vs MobileNet prototype mismatch

**Symptom.** `scripts/test_roboeye_modules.py` failed with `inconsistent tensor size ... 576 and 512`.

**Cause.** `models/authenticity_prototypes.pt` had been built with CLIP (512-d). CLIP (transformers) is not installed now, so the encoder fell back to MobileNet (576-d), and `load()` accepted the prototypes anyway.

**Fix** (`roboeye/clip_zero_shot.py`). `load()` rejects prototypes whose backend or dimension differs from the active encoder, and records the reason in `load_error`.

**Consequence of the fix.** The smoke test then rebuilt the prototypes with MobileNet, which overwrote the tracked file `models/authenticity_prototypes.pt`. `git checkout` restores the CLIP version.

**Caveat.** `build()` takes the first 80 JaalTaka notes per class without consulting the split, so these prototypes may include test notes. They must not be used for any reported accuracy.

**Check.** `scripts/test_roboeye_modules.py` prints `ALL MODULE SMOKE TESTS PASSED`.

## Verdict gating

The HUD, speech, haptics and the VLM sentence used the dual-head genuine/counterfeit vote, which, like the glass verdict, is not validated on whole-note webcam crops (`JAAL_VERDICT_FIXED.md`).

`AUTH_VERDICT_ENABLED = False`:
- labels become `unverified`
- SPACE says "authenticity not checked"
- the VLM template says "Authenticity was not checked." with no percentage
- haptics play the `unknown` pattern

## C. Live camera loop

Command: `scripts/roboeye_live.py --max-frames 150` (the new flag stops the loop after N frames). Run on the laptop's built-in webcam while a training job shared the RTX 3050:

`frames=150 seconds=13.6 fps=11.02 gradcam=ok`, exit code 0.

No note was held in front of the camera, so this checks that the loop runs. It does not measure accuracy.
