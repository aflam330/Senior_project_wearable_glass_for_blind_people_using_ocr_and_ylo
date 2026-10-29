# Replaces the Hardware Design slide of build4.js with the wiring diagram in slide_hw_wiring.js.
import pathlib

d = pathlib.Path(__file__).parent
js = (d / "build4.js").read_text(encoding="utf-8")
new = (d / "slide_hw_wiring.js").read_text(encoding="utf-8")
m = "  // ================================================================ 8b. hardware design (wiring diagram)"
a = js.index(m) if m in js else js.index("  // ================================================================ 8b. hardware design")
b = js.index("  // ================================================================ 9. prototype evolution")
js = js[:a] + new + js[b:]
(d / "build4.js").write_text(js, encoding="utf-8")
print("ok")
