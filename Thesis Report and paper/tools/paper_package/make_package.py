"""Build SARA_Paper_Package/ and SARA_Paper_Package.zip: results data, figures, evidence notes, documents and the
two LaTeX papers (IEEE conference, Q1 journal). Text result files only (no weights, images of data, audio or pickles);
nothing private (.env, field logs, participant data) is copied.

python make_package.py            # copies everything, compiles both papers, writes the zip
"""
import csv
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]                      # E:/Final SP
HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
OUT = ROOT / "SARA_Paper_Package"
REP = ROOT / "Thesis Report and paper" / "thesis report" / "My report"
SLIDES = ROOT / "Thesis Report and paper" / "thesis report" / "slides"
TEX = Path("C:/Users/LAM/tools/TinyTeX/bin/windows")
TEXT = {".json", ".csv", ".md", ".txt", ".tex"}
MAXB = 5_000_000
SKIP_PARTS = {"__pycache__", "venv", ".git", "Unused"}
PRIVATE = re.compile(r"(\.env$|field_log|participant_data(?!_template)|consent_signed)", re.I)


def copy_text_tree(src: Path, dst: Path):
    n = 0
    for p in src.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT or p.stat().st_size > MAXB:
            continue
        if SKIP_PARTS & set(p.parts) or PRIVATE.search(p.name):
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        n += 1
    return n


def copy_files(files, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / Path(f).name)
    return len(files)


def figure_index():
    """One row per figure file: where it is in the package, and the thesis caption when the thesis uses it."""
    caps = {}
    for t in REP.glob("*.tex"):
        s = t.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"\\begin\{figure\}(.*?)\\end\{figure\}", s, re.S):
            body = m.group(1)
            c = re.search(r"\\caption\{(.*)\}\s*\\label", body, re.S)
            for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
                if c:
                    caps.setdefault(Path(g).name, re.sub(r"\s+", " ", c.group(1)).strip())
    rows = []
    for p in sorted((OUT / "figures").rglob("*")):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
            rows.append((str(p.relative_to(OUT)).replace("\\", "/"), caps.get(p.name, "")))
    with open(OUT / "figures" / "FIGURE_INDEX.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "thesis caption (empty = not used in the thesis; see the notes in FIGURE_INDEX.md)"])
        w.writerows(rows)
    return len(rows)


def compile_paper(d: Path, engine="pdflatex"):
    run = lambda *a: subprocess.run([str(TEX / a[0]), *a[1:]], cwd=d, capture_output=True, text=True)
    run(engine, "-interaction=nonstopmode", "main.tex")
    run("bibtex", "main")
    run(engine, "-interaction=nonstopmode", "main.tex")
    r = run(engine, "-interaction=nonstopmode", "main.tex")
    log = (d / "main.log").read_text(encoding="utf-8", errors="ignore")
    errs = [l for l in log.splitlines() if l.startswith("!")]
    undef = len(re.findall(r"undefined", log))
    over = re.findall(r"Overfull \\hbox \(([\d.]+)pt", log)
    print(d.name, "overfull boxes:", len(over), "largest", max(map(float, over), default=0), "pt")
    pages = re.search(r"Output written on main\.pdf \((\d+) page", log)
    for junk in ["main.aux", "main.log", "main.out", "main.blg", "main.spl"]:
        (d / junk).unlink(missing_ok=True)
    return (pages.group(1) if pages else "?"), errs, undef


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    # authored documents
    for f in ["README.md", "01_PROJECT_FULL_INFORMATION.md", "02_RESULTS_MASTER_TABLE.md", "02_RESULTS_MASTER_TABLE.csv",
              "03_WRITING_GUIDE.md"]:
        shutil.copy2(SRC / f, OUT / f)
    shutil.copy2(SRC / "latex" / "ieee_conference" / "references_verified.bib", OUT / "IEEE_references_verified.bib")
    # results data
    counts = {}
    counts["results_data/glass_app"] = copy_text_tree(ROOT / "savior_glass" / "results", OUT / "results_data" / "glass_app")
    counts["results_data/taka_and_counterfeit"] = copy_text_tree(ROOT / "realtime_bangla_taka_detection" / "results",
                                                                 OUT / "results_data" / "taka_and_counterfeit")
    counts["results_data/research_results"] = copy_text_tree(ROOT / "research_results", OUT / "results_data" / "research_results")
    shutil.copytree(ROOT / "benchmark_report", OUT / "results_data" / "pi5_benchmark_report", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.pkl", "__pycache__"))
    # evidence notes (dated experiment notes, claim registry, tables, model cards, study protocol without data)
    counts["evidence_notes"] = copy_text_tree(ROOT / "paper_evidence", OUT / "evidence_notes")
    # figures
    fig = OUT / "figures"
    counts["figures/thesis"] = copy_files([p for p in (REP / "DIagram fig tab").iterdir() if p.is_file()], fig / "thesis")
    counts["figures/thesis_extra"] = copy_files(sorted(REP.glob("fig5*.png")), fig / "thesis_extra")
    counts["figures/publication"] = copy_files([p for p in (ROOT / "paper_evidence" / "figures").iterdir()
                                                if p.suffix.lower() in {".png", ".pdf"}], fig / "publication")
    deck = ROOT / "Thesis Report and paper" / "tools" / "deck" / "assets"
    counts["figures/prototype_and_test_photos"] = copy_files(
        [p for p in deck.iterdir() if p.suffix == ".jpg" and not p.name.startswith(("cartoon", "infographic"))], fig / "prototype_and_test_photos")
    counts["figures/illustrations_not_evidence"] = copy_files(sorted(SLIDES.glob("SARA Smart Glasses_*.png")), fig / "illustrations_not_evidence")
    shutil.copy2(SRC / "FIGURE_INDEX.md", fig / "FIGURE_INDEX.md")
    counts["figure index rows"] = figure_index()
    # documents
    docs = OUT / "documents"
    copy_files([REP / "SARA_Thesis_Report.pdf", REP / "SARA_Features.pdf", REP / "SARA_Results_Benchmarks.pdf",
                SLIDES / "SARA_Presentation_Redesigned.pdf", SLIDES / "SARA_Presentation_Redesigned.pptx",
                SLIDES / "SARA_Change_Log.md", SLIDES / "SARA_QA_Report.md",
                ROOT / "docs" / "literature" / "Review and paper links.xlsx"], docs)
    th = docs / "thesis_latex_source"
    th.mkdir()
    for p in list(REP.glob("*.tex")) + [REP / "ref.bib"]:
        shutil.copy2(p, th / p.name)
    # papers
    for name in ["ieee_conference", "q1_journal"]:
        shutil.copytree(SRC / "latex" / name, OUT / "latex" / name)
        d = OUT / "latex" / name / "figures"
        d.mkdir(exist_ok=True)
        for line in (SRC / "latex" / name / "FIGURES.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                shutil.copy2(ROOT / line.strip(), d / Path(line.strip()).name)
        (OUT / "latex" / name / "FIGURES.txt").unlink()
        pages, errs, undef = compile_paper(OUT / "latex" / name)
        counts[f"latex/{name}"] = f"{pages} pages, {len(errs)} errors, {undef} 'undefined' in log"
        if errs:
            print(name, errs[:5])
    # zip
    z = ROOT / "SARA_Paper_Package.zip"
    z.unlink(missing_ok=True)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, Path("SARA_Paper_Package") / p.relative_to(OUT))
    for k, v in counts.items():
        print(f"{k}: {v}")
    print("zip:", z, round(z.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
