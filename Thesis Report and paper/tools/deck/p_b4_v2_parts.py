# Adds the "Second Prototype: Its Parts" slide (team's labelled photo, assets/v2_labelled.jpg) after
# "Evolution of the Prototype" in build4.js. Running it again replaces it.
import pathlib
import re

d = pathlib.Path(__file__).parent
js = (d / "build4.js").read_text(encoding="utf-8")
js = re.sub(r"  // =+ 9b\. second prototype, labelled parts.*?(?=  // =+ 10\. Taka detection)", "", js, flags=re.S)
a = js.index("  // ================================================================ 10. Taka detection")
js = js[:a] + (d / "slide_v2_parts.js").read_text(encoding="utf-8") + js[a:]
(d / "build4.js").write_text(js, encoding="utf-8")
print("ok")
