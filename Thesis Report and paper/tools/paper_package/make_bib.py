"""Write references.bib for each paper: the entries it cites, taken from the thesis bibliography (already checked),
plus three entries the thesis does not have (Vosk, ESC-50, Silero VAD)."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
THESIS_BIB = HERE.parents[1] / "thesis report" / "My report" / "ref.bib"
EXTRA = r"""
@misc{vosk,
  author       = {{Alpha Cephei}},
  title        = {Vosk Offline Speech Recognition Toolkit},
  howpublished = {\url{https://alphacephei.com/vosk/}}
}

@inproceedings{esc50,
  author    = {Piczak, Karol J.},
  title     = {{ESC}: Dataset for Environmental Sound Classification},
  booktitle = {Proceedings of the 23rd ACM International Conference on Multimedia},
  pages     = {1015--1018},
  year      = {2015},
  doi       = {10.1145/2733373.2806390}
}

@misc{silero,
  author       = {{Silero Team}},
  title        = {Silero {VAD}: Pre-trained Enterprise-Grade Voice Activity Detector},
  howpublished = {\url{https://github.com/snakers4/silero-vad}}
}
"""


def entries(text):
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        start = m.start()
        depth, i = 0, text.index("{", start)
        for j in range(i, len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                out[m.group(2)] = text[start:j + 1]
                break
    return out


def main():
    pool = entries(THESIS_BIB.read_text(encoding="utf-8"))
    pool.update(entries(EXTRA))
    for name in ["ieee_conference", "q1_journal"]:
        tex = (HERE / "src" / "latex" / name / "main.tex").read_text(encoding="utf-8")
        keys = []
        for group in re.findall(r"\\cite\{([^}]+)\}", tex):
            for k in group.split(","):
                k = k.strip()
                if k and k not in keys:
                    keys.append(k)
        missing = [k for k in keys if k not in pool]
        assert not missing, (name, missing)
        body = "% Entries taken from the SARA thesis bibliography (ref.bib) unless marked otherwise.\n\n"
        # keep the capitalisation of titles (dataset and model names) that the styles would otherwise lower-case
        protect = lambda e: re.sub(r"(\btitle\s*=\s*)\{(?!\{)(.*?)\}(\s*,?\s*\n)", r"\1{{\2}}\3", e, count=1, flags=re.S)
        body += "\n\n".join(protect(pool[k]) for k in keys) + "\n"
        (HERE / "src" / "latex" / name / "references.bib").write_text(body, encoding="utf-8")
        print(name, len(keys), "references")


if __name__ == "__main__":
    main()
