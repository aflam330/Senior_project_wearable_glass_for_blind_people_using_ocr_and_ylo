# Occlusion with a different box on each view

The early-stop runs already occluded a different region on every view. The draw is independent per view, with area fraction uniform between 0.35 and 0.55, then median fill. Details and the validation table are in `OCCLUSION_EARLYSTOP.md`.

No epoch improved occluded validation without either collapsing clean validation to 0.5769230769230769 or leaving occluded validation at 0.8413461538461539. The test set was not used. Target 0.90 was not reached. The standing test number remains median-fill 0.875 on the original occlusion checkpoint.
