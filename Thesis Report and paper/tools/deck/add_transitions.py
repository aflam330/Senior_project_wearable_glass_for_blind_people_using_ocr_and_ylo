"""Add a subtle fade transition to every slide of a .pptx (pptxgenjs cannot write transitions).
python add_transitions.py <deck.pptx>"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

src = Path(sys.argv[1])
tmp = src.with_suffix(".tmp.pptx")
FADE = '<p:transition spd="med"><p:fade/></p:transition>'
n = 0
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if re.fullmatch(r"ppt/slides/slide\d+\.xml", item.filename):
            x = data.decode("utf-8")
            if "<p:transition" not in x:
                assert x.count("</p:clrMapOvr>") == 1, item.filename
                x = x.replace("</p:clrMapOvr>", "</p:clrMapOvr>" + FADE)
                n += 1
            data = x.encode("utf-8")
        zout.writestr(item, data)
shutil.move(tmp, src)
print("fade added to", n, "slides")
