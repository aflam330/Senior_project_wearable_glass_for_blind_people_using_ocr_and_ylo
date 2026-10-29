# Replaces the System Architecture slide of build4.js with the stream diagram in slide_arch_streams.js.
import pathlib

d = pathlib.Path(__file__).parent
js = (d / "build4.js").read_text(encoding="utf-8")
new = (d / "slide_arch_streams.js").read_text(encoding="utf-8")
m = "  // ================================================================ 8. architecture (stream diagram)"
a = js.index(m) if m in js else js.index("  // ================================================================ 8. architecture")
b = js.index("  // ================================================================ 8b. hardware design")
js = js[:a] + new + js[b:]
(d / "build4.js").write_text(js, encoding="utf-8")
print("ok")
