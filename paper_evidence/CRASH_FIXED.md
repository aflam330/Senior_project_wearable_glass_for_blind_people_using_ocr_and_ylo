# The hand-held fine-tune crash: cause and fix (14 October 2026)

## What happened

1. **First run** (`workers=4`): stopped in its first epoch with `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 9.38 MiB for an array with shape (640, 640, 3)` inside Ultralytics' mixup augmentation. At that moment a Whisper study (`scripts/bangla_commands_study.py`, model "small") was running beside it.
2. **Second run**: stopped silently when the session restarted; its processes were children of the session and were ended with it. Nothing had been saved yet, because checkpoints were written to a folder the selection script did not look in (`runs/detect/runs/closeup_ft/...`, where Ultralytics puts a relative `project`).

## Measured memory (third run, Windows process list)

| Process | Memory |
|---|---:|
| Trainer (YOLOv8s, batch 16, 640 px, GPU) | about 3.0 GB |
| Each data-loading worker | 0.8 to 1.0 GB |
| Workers alive at once (training + validation loaders) | up to 6 |
| Total | about 8.3 GB of the laptop's 16 GB; 1.5 to 1.7 GB left free |

The bottleneck is not the GPU (3.4 GB of 4 GB, steady) but system memory: every worker holds its own decoded images and augmentation buffers. A Whisper model of 1 to 2 GB started beside it leaves nothing.

## Fix

| Problem | Fix |
|---|---|
| Too many loader processes | `workers=2` (was 4) |
| Two heavy jobs at once | one heavy job at a time; light jobs (online speech synthesis, file copies) may run beside training; model evaluations wait for training to end when less than 2 GB is free |
| Jobs ended with the session | long jobs are started as detached processes (`Start-Process`), not as children of the session |
| Progress lost on a crash | a checkpoint after every epoch (`save_period=1`); `python scripts/train_closeups.py` now resumes an unfinished run from its last epoch (`--fresh` starts again) |
| Checkpoints not found | the selection script reads the folder Ultralytics actually uses |

Mixed precision is already on (Ultralytics trains with automatic mixed precision on the GPU). Gradient checkpointing and smaller batches would lower GPU memory, which was not the limit.

## Result

The third run has trained without interruption, with system memory steady, writing one checkpoint per epoch; validation mAP@0.5 stays at 0.992 (the deployed detector: 0.993). Its accuracy on hand-held notes is reported in `HANDHELD_FINETUNE.md` once selection and the single test read are done.
