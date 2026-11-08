"""The knowledge-graph figure: compiles statistics/figures/fig_kg.tex and crops it -> statistics/figures/fig_kg.pdf.
Run: uv run --with pymupdf python statistics/figure_kg.py"""

import subprocess
from pathlib import Path

import pymupdf

LATEX = Path(__file__).parent / "figures"
MARGIN = 4

subprocess.run(["pdflatex", "-disable-installer", "-interaction=nonstopmode", "fig_kg.tex"], cwd=LATEX, stdout=subprocess.DEVNULL, timeout=300)
doc = pymupdf.open(LATEX / "fig_kg.pdf")
page = doc[0]
box = pymupdf.Rect()
for kind, rect in page.get_bboxlog():
    box |= pymupdf.Rect(rect)
assert not box.is_empty, "nothing drawn"
page.set_cropbox((box + (-MARGIN, -MARGIN, MARGIN, MARGIN)) & page.rect)
doc.save(LATEX / "fig_kg.cropped.pdf")
doc.close()
(LATEX / "fig_kg.cropped.pdf").replace(LATEX / "fig_kg.pdf")
print(f"fig_kg.pdf: {box.width / 72 * 2.54:.1f} x {box.height / 72 * 2.54:.1f} cm")
