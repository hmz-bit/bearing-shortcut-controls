"""Extract text of the XJTU-SY documentation PDFs (read-only parsing with PyMuPDF) and print lines about failure elements."""
import sys, re, pathlib
import pymupdf

D = pathlib.Path(__file__).resolve().parents[2] / "data" / "xjtu" / "docs"
for name in ("A_hybrid_prognostics_approach_for_estimating_remaining_useful_life_of_rolling_element_bearings.pdf", "XJTU-SY_.pdf"):
    doc = pymupdf.open(D / name)
    text = "\n".join(p.get_text() for p in doc)
    (D / (name[:-4] + ".txt")).write_text(text, encoding="utf-8")
    print("==", name, len(doc), "pages", len(text), "chars")
    for i, line in enumerate(text.splitlines()):
        if re.search(r"outer race|inner race|cage|Bearing ?[123][_ -]?[1-5]|external|sampling|25\.6|32768|1\.28|failure", line, re.I):
            print(i, line.strip()[:160].encode("ascii", "replace").decode())
    print()
