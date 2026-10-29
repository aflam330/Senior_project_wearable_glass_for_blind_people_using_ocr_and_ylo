# Replaces slides 3-4 of build4.js (Problem Statement, Proposed Solution) with the
# picture flowcharts in slides_3_4_pictures.js (rich-picture style).
import pathlib

d = pathlib.Path(__file__).parent
js = (d / "build4.js").read_text(encoding="utf-8")
new = (d / "slides_3_4_pictures.js").read_text(encoding="utf-8")

m = "  // picture-flowchart helpers"
a = js.index(m) if m in js else js.index("  // ================================================================ 3. problem statement: flowchart")
b = js.index("  // ================================================================ 5. SARA at a glance")
js = js[:a] + new + js[b:]
(d / "build4.js").write_text(js, encoding="utf-8")
print("ok")
