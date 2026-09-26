"""Write the severity table from the saved JSON. No new measurements."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
payload = json.loads((ROOT / "results" / "robustness" / "severity_seed42.json").read_text(encoding="utf-8"))
clean = payload["clean_6view"]
lines = [
    "# Severity curves",
    "",
    "Six views, seed 42, test n=208. Drop is clean accuracy minus corrupted accuracy.",
    "The severity column is the parameter passed to `apply_corruption`.",
    "For low light, JPEG quality, and scale, a larger parameter is not a stronger corruption.",
    "",
    f"Clean 6-view: PRMVT {clean['prmvt']:.4f}, NDAL {clean['ndal']:.4f}, baseline {clean['baseline']:.4f}.",
    "",
    "Source: `realtime_bangla_taka_detection/results/robustness/severity_seed42.json`.",
    "",
    "| corruption | severity | PRMVT | drop | NDAL | drop | baseline | drop |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in payload["rows"]:
    lines.append(
        f"| {row['corruption']} | {row['severity']} | {row['prmvt']:.4f} | {row['prmvt_drop']:.4f} | "
        f"{row['ndal']:.4f} | {row['ndal_drop']:.4f} | {row['baseline']:.4f} | {row['baseline_drop']:.4f} |"
    )
out = ROOT.parent / "paper_evidence" / "SEVERITY_CURVES.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out, len(payload["rows"]))
