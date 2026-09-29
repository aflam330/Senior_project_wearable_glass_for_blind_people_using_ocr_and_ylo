# Operating points for policy E, fixed before any whole-note score was read

Written 2026-09-29 13:47 UTC, while `jaal_whole_note.py` was still scoring photos. Only JaalTaka VAL notes are used.

- Primary: tau = highest p(genuine) of any JaalTaka VAL counterfeit note (0 VAL counterfeit passed).
- Secondary: tau = 99th percentile of JaalTaka VAL counterfeit scores (at most 1 % of VAL counterfeit passed).
- Checker for the app: the one with the lowest share of VAL genuine notes sent to "check by hand" at the primary tau; ties go to fewer views.
- Whole-note photos (counterfeit set, Bangla Money, BanglaTaka) are scored once with these tau and are not used to choose tau or the checker.
