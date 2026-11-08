"""The overview figures: storms per subtype and decade, and the storm x country records on a map
-> statistics/figures/fig_storm_subtypes.pdf, fig_storm_map.pdf. Run: python statistics/figure1_storms.py"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LogNorm

BASE = Path(__file__).parent.parent
SUBTYPES, MAP = BASE / "statistics" / "figures" / "fig_storm_subtypes.pdf", BASE / "statistics" / "figures" / "fig_storm_map.pdf"
# Every figure at one width (the knowledge graph figure's, 13.1 cm) in Arial, text 8 pt and titles 9 pt, saved as PDF with fonts embedded
FIG_WIDTH, TEXT, TITLE = 13.1 / 2.54, 8, 9
# Sequential ramps of one hue (counts, light to dark) and the categorical blue and orange of the colour-blind-safe palette
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGES = ["#fde3d3", "#fbc4a7", "#f6a47b", "#ee8450", "#d9632b", "#b04a1a"]
READ, TOTAL, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#ffffff"


def luminance(hex_colour: str) -> float:
    """Relative luminance (WCAG) of a colour, to choose dark or white text on it."""
    channel = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(int(hex_colour[i:i + 2], 16) / 255) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


emdat = pd.read_csv(BASE / "datasets" / "clean" / "emdat.csv", low_memory=False)
storms = emdat[emdat["hazard"].notna() & (emdat["year"] >= 1990)]
extracts = pd.read_csv(BASE / "datasets" / "linked" / "event_points.csv", low_memory=False)
points = pd.read_csv(BASE / "datasets" / "linked" / "event_points.csv")
records = storms.merge(points, on=["disasterno", "iso3"], how="left").merge(extracts[["disasterno", "iso3", "area_source"]], on=["disasterno", "iso3"],
                                                                            how="left")
assert len(records) == len(storms) == 3492 and storms["disasterno"].nunique() == 2511
located, unlocated = records[records["lat"].notna()], records[records["lat"].isna()]
read = located["area_source"].notna()
assert len(located) == 2846 and len(unlocated) == 646

decades = (storms["year"] // 10 * 10).map({1990: "1990s", 2000: "2000s", 2010: "2010s", 2020: f"2020–{storms['year'].max() % 100:02d}"})
counts = storms.assign(decade=decades).groupby(["disaster_subtype", "decade"])["disasterno"].nunique().unstack(fill_value=0)
counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
counts.index = counts.index.str.replace("/", " / ")

countries = gpd.read_file(BASE / "datasets" / "clean" / "countries.gpkg").query("iso3 != 'ATA'")
missing = countries.merge(unlocated.groupby("iso3").size().rename("records").reset_index(), on="iso3", how="left")

plt.rcParams.update({"font.family": "Arial", "font.size": TEXT, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": INK,
                     "ytick.color": INK, "axes.edgecolor": MUTED, "pdf.fonttype": 42})
# The row labels lie outside the axes, so the axes are 1 cm narrower than the printed figure
fig, (ax, bars) = plt.subplots(1, 2, figsize=(FIG_WIDTH - 1.0 / 2.54, 3.2), sharey=True, gridspec_kw={"width_ratios": [4, 1.2], "wspace": 0.04}, facecolor=SURFACE)
values = counts.to_numpy()
norm = LogNorm(1, values.max())
cmap = ListedColormap(BLUES)
ax.imshow(np.ma.masked_equal(values, 0), cmap=cmap, norm=norm, aspect="auto")
for (row, column), n in np.ndenumerate(values):
    # White text where the cell is dark (contrast at least 4.5:1 against the cell), else dark ink
    cell = "#%02x%02x%02x" % tuple(int(255 * c) for c in cmap(norm(n))[:3]) if n else SURFACE
    ax.text(column, row, f"{n:,}" if n else "–", ha="center", va="center", fontsize=TEXT, color="white" if n and luminance(cell) < 0.18 else (INK if n else MUTED))
ax.set_xticks(range(len(counts.columns)), counts.columns)
ax.set_yticks(range(len(counts)), counts.index)
ax.xaxis.tick_top()
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, len(counts.columns)), minor=True)
ax.set_yticks(np.arange(-0.5, len(counts)), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)
totals = storms.groupby("disaster_subtype")["disasterno"].nunique().rename(lambda name: name.replace("/", " / ")).loc[counts.index]
bars.barh(range(len(totals)), totals, height=0.6, color=TOTAL)
for row, total in enumerate(totals):
    bars.text(total + totals.max() * 0.03, row, f"{total:,}", ha="left", va="center", fontsize=TEXT, color=INK)
bars.set_xlim(0, totals.max() * 1.45)
bars.set_xticks([])
bars.set_title("total", fontsize=TEXT, color=INK, pad=6)
bars.tick_params(length=0)
for side in [*ax.spines.values(), *bars.spines.values()]:
    side.set_visible(False)
ax.set_title(f"Storms per subtype and decade ({storms['disasterno'].nunique():,} storms, 1990–2023)", loc="left", fontsize=TITLE, pad=20)
fig.savefig(SUBTYPES, bbox_inches="tight", facecolor=SURFACE)

# No side margins, so the equal-area world fills the width
fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.7), facecolor=SURFACE)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
crs = "EPSG:8857"
# Borders simplified to 5 km: invisible at print size, a small PDF
borders = lambda g: g.to_crs(crs).simplify(5_000)
borders(countries).plot(ax=ax, color="#f4f4f2", edgecolor="#c6c5bf", linewidth=0.3)
bins = [1, 2, 4, 8, 16, 32, np.inf]
shaded = missing[missing["records"].notna()].copy()
shaded["step"] = np.digitize(shaded["records"], bins) - 1
borders(shaded).plot(ax=ax, color=[ORANGES[s] for s in shaded["step"]], edgecolor="#a3a29c", linewidth=0.3)
dots = gpd.GeoDataFrame(located, geometry=gpd.points_from_xy(located["lon"], located["lat"]), crs="EPSG:4326").to_crs(crs)
dots[read].plot(ax=ax, markersize=2.5, color=READ, edgecolor="white", linewidth=0.25)
ax.set_axis_off()
ax.set_title(f"Storm × country records, 1990–2023 ({len(records):,})", loc="left", fontsize=TITLE)
points_legend = ax.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", markersize=4.5, color=READ, markeredgecolor="white",
                                              label=f"event area, values read ({read.sum():,})")],
                          loc="upper left", bbox_to_anchor=(0.0, 0.1), frameon=False, fontsize=TEXT, handletextpad=0.4)
ax.add_artist(points_legend)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=ORANGES[step], edgecolor="#a3a29c", linewidth=0.3, label=label)
                   for step, label in enumerate(["1", "2–3", "4–7", "8–15", "16–31", "32+"])],
          title="records without an event area, per country", title_fontsize=TEXT, loc="upper right", bbox_to_anchor=(1.0, 0.04), ncol=6,
          frameon=False, fontsize=TEXT, handlelength=1.6, handletextpad=0.3, columnspacing=0.8)
ax.text(0.5, -0.24, f"{len(unlocated):,} records have no event area (no Geo-Disasters units and no EM-DAT point)\nand so no "
        "event-area values", transform=ax.transAxes, ha="center", fontsize=TEXT, color=MUTED)
fig.savefig(MAP, bbox_inches="tight", facecolor=SURFACE)
print(f"{SUBTYPES.name}, {MAP.name}: {len(storms['disasterno'].unique()):,} storms, {len(records):,} records ({read.sum():,} read, "
      f"{len(unlocated):,} without an event area)")
