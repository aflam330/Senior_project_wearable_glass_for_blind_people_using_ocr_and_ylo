# Replaces slides 2-4 of build4.js (Introduction, Problem Statement, Proposed Solution)
# with the bullet introduction and the two flowcharts in slides_2_4.js.
import pathlib

d = pathlib.Path(__file__).parent
js = (d / "build4.js").read_text(encoding="utf-8")
new = (d / "slides_2_4.js").read_text(encoding="utf-8")

a = js.index("  // ================================================================ 2. problem: scenario storyboard")
b = js.index("  // ================================================================ 5. SARA at a glance")
js = js[:a] + new + js[b:]

ref = '  ["uddin",'
add = ('  ["rpiaid", "Adusumilli, V. et al. Reading aid and translator with Raspberry Pi for blind people. ICACCS, 2023, pp. 327-331.", '
       '"https://doi.org/10.1109/ICACCS57279.2023.10113042"],\n')
assert '["rpiaid"' not in js
js = js.replace(ref, add + ref, 1)

(d / "build4.js").write_text(js, encoding="utf-8")
print("ok")
