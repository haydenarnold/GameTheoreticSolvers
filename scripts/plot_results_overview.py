"""Render paper/figures/results_overview.png -- the two-panel Figure 1:
(a) live rating vs. games played (from data/stats_history.csv), and
(b) median normalized Monte Carlo score by archetype, on the deployed
strategy code (400 trials/archetype/family, from scripts/montecarlo_opponents.py).

Panel (b)'s median_data below is a snapshot of that script's output, not
computed dynamically here -- re-run scripts/montecarlo_opponents.py and
paste its `median=...` column back in whenever the deployed strategy code
changes materially enough to warrant a new figure. Current values reflect
the 2026-08-29 fix to that script (a per-round rather than per-trial game
identifier for bargaining/negotiation, and an unseeded global-random call
in persuasion's seller bluff logic, both of which silently affected
reproducibility -- see that script's own fix comments).

Usage: .venv/bin/python scripts/plot_results_overview.py
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

# --- left panel data: real live rating history ---
by_family = defaultdict(list)
with open(DATA_PATH, newline="") as f:
    for row in csv.DictReader(f):
        row["rating"] = float(row["rating"])
        row["games_played"] = int(row["games_played"])
        by_family[row["family"]].append(row)
for rows in by_family.values():
    rows.sort(key=lambda r: r["games_played"])

# --- right panel data: median normalized Monte Carlo score per archetype,
# from `scripts/montecarlo_opponents.py`'s output (2026-08-30 run -- reflects
# today's bargaining fixes: _opponent_not_conceding's cumulative-drift check
# and the round-1 50/50 opening rule for delta_me<=0.90 under incomplete
# information/unlimited horizon. Negotiation and persuasion unchanged from
# 2026-08-29 since neither's decision logic was touched today) ---
median_data = {
    "bargaining": {
        "hardliner": 0.0020, "fair_splitter": 0.3905, "tit_for_tat": 0.2076,
        "deadline_exploiter": 0.0031, "random": 0.4298, "deceptive": 0.0753,
        "highly_impatient": 0.3000, "lowball_highball": 0.0034,
        "bayesian_rational": 0.4759, "llm_like": 0.2045, "noisy_rational": 0.4517,
    },
    "negotiation": {
        "hardliner": -1.0000, "fair_splitter": 0.1208, "tit_for_tat": 0.0491,
        "deadline_exploiter": 0.0312, "random": 0.2688, "deceptive": 0.0494,
        "highly_impatient": 0.1121, "lowball_highball": 0.0566,
        "bayesian_rational": 0.1320, "llm_like": 0.0724, "noisy_rational": 0.1142,
    },
    "persuasion": {
        "hardliner": 0.0000, "fair_splitter": 0.0000, "tit_for_tat": 0.0000,
        "deadline_exploiter": 0.0500, "random": 0.3000, "deceptive": 0.3000,
        "highly_impatient": 0.3500, "lowball_highball": 0.0000,
        "bayesian_rational": 0.4000, "llm_like": 0.0000, "noisy_rational": 0.4000,
    },
}
archetypes = ["hardliner", "fair_splitter", "tit_for_tat", "deadline_exploiter", "random",
              "deceptive", "highly_impatient", "lowball_highball", "bayesian_rational",
              "llm_like", "noisy_rational"]
families = ["bargaining", "negotiation", "persuasion"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.6))

# left: rating trajectory
for family, rows in sorted(by_family.items()):
    xs = [r["games_played"] for r in rows]
    ys = [r["rating"] for r in rows]
    ax1.plot(xs, ys, label=family, color=FAMILY_COLORS.get(family), linewidth=2.0)
ax1.axhline(1000, color="gray", linewidth=1.1, linestyle=":", alpha=0.7, label="starting rating")
ax1.set_xlabel("games played", fontsize=12)
ax1.set_ylabel("rating", fontsize=12)
ax1.tick_params(axis="both", labelsize=10)
ax1.legend(fontsize=9.5, loc="lower right", frameon=False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_title("(a) Live rating vs. games played", fontsize=12)

# right: median normalized score per archetype
x = np.arange(len(archetypes))
width = 0.26
for i, fam in enumerate(families):
    vals = [median_data[fam][a] for a in archetypes]
    ax2.bar(x + (i - 1) * width, vals, width, label=fam, color=FAMILY_COLORS[fam])
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([a.replace("_", " ") for a in archetypes], fontsize=8.5, rotation=35, ha="right")
ax2.set_ylabel("median normalized score", fontsize=12)
ax2.tick_params(axis="y", labelsize=10)
ax2.set_ylim(-1.08, 0.66)
ax2.legend(fontsize=9.5, ncol=3, loc="upper left", frameon=False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_title("(b) Median Monte Carlo score by archetype", fontsize=12)

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=200)
print("saved", OUT_PATH)
