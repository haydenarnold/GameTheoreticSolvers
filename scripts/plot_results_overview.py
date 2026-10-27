"""Draws the paper's two-panel results figure.

Panel (a) shows live rating against games played per family; panel (b) shows
the median Monte Carlo score per archetype, using the values hardcoded below from the real output.
"""

import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "stats_history.csv"
OUT_PATH = ROOT / "paper" / "figures" / "results_overview.png"

FAMILY_COLORS = {
    "bargaining": "#4C72B0",
    "negotiation": "#DD8452",
    "persuasion": "#55A868",
}

by_family = defaultdict(list)
with open(DATA_PATH, newline="") as f:
    for row in csv.DictReader(f):
        row["rating"] = float(row["rating"])
        row["games_played"] = int(row["games_played"])
        by_family[row["family"]].append(row)
# Cap the x-axis so the live curves match the 2600 games shown in the paper.
MAX_GAMES = 2600
for fam in by_family:
    by_family[fam] = sorted((r for r in by_family[fam] if r["games_played"] <= MAX_GAMES),
                            key=lambda r: r["games_played"])

median_data = {
    "bargaining": {
        "hardliner": 0.0023,
        "fair_splitter": 0.3905,
        "tit_for_tat": 0.2397,
        "deadline_exploiter": 0.0033,
        "random": 0.4281,
        "deceptive": 0.0819,
        "highly_impatient": 0.3000,
        "lowball_highball": 0.0033,
        "bayesian_rational": 0.4780,
        "llm_like": 0.2235,
        "noisy_rational": 0.4421,
    },
    "negotiation": {
        "hardliner": -1.0000,
        "fair_splitter": 0.1208,
        "tit_for_tat": 0.0495,
        "deadline_exploiter": 0.0188,
        "random": 0.2534,
        "deceptive": 0.0236,
        "highly_impatient": 0.0872,
        "lowball_highball": 0.0269,
        "bayesian_rational": 0.0955,
        "llm_like": 0.0473,
        "noisy_rational": 0.0733,
    },
    "persuasion": {
        "hardliner": 0.0000,
        "fair_splitter": 0.0000,
        "tit_for_tat": 0.1003,
        "deadline_exploiter": 0.0337,
        "random": 0.3000,
        "deceptive": 0.2500,
        "highly_impatient": 0.3392,
        "lowball_highball": 0.0000,
        "bayesian_rational": 0.3500,
        "llm_like": 0.0326,
        "noisy_rational": 0.3500,
    },
}
archetypes = ["hardliner", "fair_splitter", "tit_for_tat", "deadline_exploiter", "random",
              "deceptive", "highly_impatient", "lowball_highball", "bayesian_rational",
              "llm_like", "noisy_rational"]
families = ["bargaining", "negotiation", "persuasion"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.6))

for family, rows in sorted(by_family.items()):
    xs = [r["games_played"] for r in rows]
    ys = [r["rating"] for r in rows]
    ax1.plot(xs, ys, label=family, color=FAMILY_COLORS.get(family), linewidth=2.0)
ax1.axhline(1000, color="gray", linewidth=1.1, linestyle=":", alpha=0.7, label="starting rating")
ax1.set_xlim(-130, 2700)
ax1.set_xlabel("games played", fontsize=14)
ax1.set_ylabel("rating", fontsize=14)
ax1.tick_params(axis="both", labelsize=12)
ax1.legend(fontsize=11.5, loc="lower right", frameon=False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_title("(a) Live rating vs. games played", fontsize=14)

x = np.arange(len(archetypes))
width = 0.26
for i, fam in enumerate(families):
    vals = [median_data[fam][a] for a in archetypes]
    ax2.bar(x + (i - 1) * width, vals, width, label=fam, color=FAMILY_COLORS[fam])
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([a.replace("_", " ") for a in archetypes], fontsize=10.5, rotation=35, ha="right")
ax2.set_ylabel("median norm. score", fontsize=14)
ax2.tick_params(axis="y", labelsize=12)
ax2.set_ylim(-1.08, 0.82)
ax2.legend(fontsize=11.5, ncol=3, loc="upper left", frameon=False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_title("(b) Median Monte Carlo score by archetype", fontsize=14)

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=200)
print("saved", OUT_PATH)
PDF_PATH = OUT_PATH.with_suffix(".pdf")
fig.savefig(PDF_PATH)
print("saved", PDF_PATH)
