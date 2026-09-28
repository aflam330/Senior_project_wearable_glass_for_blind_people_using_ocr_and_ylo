# Occlusion: security-feature prior (attempt E)

**Not attempted (2026-09-28).** The idea was: if the covered region holds a security feature, flag
the note as uncertain; otherwise classify from what is visible. That needs to know where the
security features are. JaalTaka has no such labels, and there is no ground truth to check a proxy
against, so any "security-feature region" would be invented. The rejection rule in
`OCCLUSION_REJECTION.md` covers the same safety need (answer only when confident) without it.

Doing this properly means annotating the watermark, security thread, hologram and serial
regions for each denomination and view, which is data collection, not a code change.
