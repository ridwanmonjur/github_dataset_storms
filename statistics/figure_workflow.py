"""The workflow figure: inputs, the five steps with their scripts, and the public deposit -> statistics/figures/fig_workflow.pdf.
Run: python statistics/figure_workflow.py"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTPUT = Path(__file__).parent / "figures" / "fig_workflow.pdf"
WIDTH, HEIGHT, TEXT, TITLE = 13.1, 7.3, 8, 9
INK, MUTED, SURFACE, FRAME = "#0b0b0b", "#52514e", "#ffffff", "#f4f4f2"
# (border, light fill) per step: blue, orange, aqua, yellow, violet
COLOURS = [("#2a78d6", "#dceafb"), ("#eb6834", "#fce3d7"), ("#1baf7a", "#d5f2e7"), ("#eda100", "#fcefcc"), ("#4a3aa7", "#e4e1f6")]
# (title, what the step does, its script in the code)
STEPS = [("1 Download", "raw files", "run1_download_\ndatasets.py"),
         ("2 Clean", "used columns,\ntypes, signs\nand units", "run2_clean_\ndatasets.py"),
         ("3 Link", "aid,\ndisplacement,\nstorm tracks,\ncascades", "run3_link_\nrecords.py"),
         ("4 Extract", "Earth Engine\nmeasures,\nmodelled surge", "run4_extract_\nmeasures.py"),
         ("5 Build", "keys checked,\nCSV, SQLite,\nknowledge\ngraph", "run5_build_\noutputs.py")]
BOX, GAP, ROW, TALL = 2.28, 0.4, 2.0, 2.9


def box(ax, x, y, w, h, edge, fill, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", facecolor=fill, edgecolor=edge, linewidth=1.1,
                                linestyle="--" if dashed else "-"))


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color=MUTED, linewidth=1, shrinkA=0, shrinkB=0))


plt.rcParams.update({"font.family": "Arial", "font.size": TEXT, "text.color": INK, "pdf.fonttype": 42})
fig = plt.figure(figsize=(WIDTH / 2.54, HEIGHT / 2.54), facecolor=SURFACE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_axis_off()

# Inputs above, the five steps, the deposit below; each box's width and height in cm
box(ax, 0.05, 5.85, WIDTH - 0.1, 1.35, "#a3a29c", FRAME)
ax.text(0.25, 6.85, "Inputs", fontsize=TITLE, weight="bold", va="center")
ax.text(0.25, 6.25, "Impact records, and the hazard, exposure and vulnerability sources that\nthe review of published studies found to matter for storm impacts",
        fontsize=TEXT, va="center", linespacing=1.3)
for i, ((title, detail, script), (edge, fill)) in enumerate(zip(STEPS, COLOURS)):
    x = 0.05 + i * (BOX + GAP)
    box(ax, x, ROW, BOX, TALL, edge, fill)
    ax.text(x + BOX / 2, ROW + TALL - 0.38, title, ha="center", va="center", fontsize=TITLE, weight="bold")
    ax.text(x + BOX / 2, ROW + 1.6, detail, ha="center", va="center", fontsize=TEXT, linespacing=1.3)
    ax.text(x + BOX / 2, ROW + 0.42, script, ha="center", va="center", fontsize=TEXT, color=MUTED, linespacing=1.2)
    if i < len(STEPS) - 1:
        arrow(ax, (x + BOX + 0.05, ROW + TALL / 2), (x + BOX + GAP - 0.05, ROW + TALL / 2))
arrow(ax, (0.05 + BOX / 2, 5.85),(0.05 + BOX / 2, ROW + TALL + 0.02))
last = 0.05 + 4 * (BOX + GAP)
box(ax, 0.05, 0.05, WIDTH - 0.1, 1.45, "#a3a29c", FRAME)
arrow(ax, (last + BOX / 2, ROW - 0.02), (last + BOX / 2, 1.5))
ax.text(0.25, 1.15, "Public deposit (Zenodo)", fontsize=TITLE, weight="bold", va="center")
ax.text(0.25, 0.55, "Every table but EM-DAT's, the knowledge graph without EM-DAT, schema and data\ndictionary. "
        "Users rebuild EM-DAT's values from their own export", fontsize=TEXT, va="center", linespacing=1.3)
fig.savefig(OUTPUT, facecolor=SURFACE)
print(f"{OUTPUT.name} written")
