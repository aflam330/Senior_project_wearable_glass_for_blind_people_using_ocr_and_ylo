# A second counterfeit evaluation from JaalTaka (2026-09-30)

JaalTaka has no session or camera IDs (`camera_id` and `session_id` are empty), so a session-disjoint split cannot be built. Two other independent evaluations were made instead.

| Evaluation | What makes it a second test | Result | Source |
|---|---|---|---|
| Serial-disjoint split | Test counterfeits come from prints absent from training | Prefix network 90.1 % (1 view) / 87.8 % (6); + watermark 93.2 / 95.5 % | `SERIAL_FIX.md` |
| Per-note cross-validation | Every note is tested once; folds note-disjoint or serial-grouped | Probe 98.3 / 95.8 % at 1 view; 98.9 % at 6 views both ways | `EXPANDED_TEST.md` |
| Whole-note counterfeit set | Different photos, whole notes, other counterfeits | Policy E passed 0 / 25; AUC up to 0.966 | `JAAL_VERDICT_FIXED.md` |

The serial-disjoint split is the one that answers "does it catch a counterfeit print it has never seen". Its answer, about 88–90 % for one network and 95.5 % with the watermark, is the number to lead with.
