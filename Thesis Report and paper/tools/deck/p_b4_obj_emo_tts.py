# Adds three slides to build4.js: Object Detection and Facial Expression Recognition (new section "Objects and
# emotion", before "Reading and voice") from slides_obj_emo.js, and Text-to-Speech (before Voice Interaction) from
# slide_tts.js. Running it again replaces them.
import pathlib
import re

d = pathlib.Path(__file__).parent
js = (d / "build4.js").read_text(encoding="utf-8")
js = re.sub(r"  // =+ 17b\. object detection.*?(?=  // =+ 18\. text reading)", "", js, flags=re.S)
js = re.sub(r"  // =+ 18b\. text-to-speech.*?(?=  // =+ 19\. voice)", "", js, flags=re.S)
a = js.index("  // ================================================================ 18. text reading")
js = js[:a] + (d / "slides_obj_emo.js").read_text(encoding="utf-8") + js[a:]
b = js.index("  // ================================================================ 19. voice")
js = js[:b] + (d / "slide_tts.js").read_text(encoding="utf-8") + js[b:]
(d / "build4.js").write_text(js, encoding="utf-8")
print("ok")
