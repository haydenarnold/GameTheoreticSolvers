"""Plots rating against games played and against time from the stats snapshots."""

from __future__ import annotations

import argparse
import csv
import datetime
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stats_history.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "plots"

FAMILY_COLORS = {
    "bargaining": "#4C72B0",
    "negotiation": "#DD8452",
    "persuasion": "#55A868",
}


def load_history(path: Path) -> dict[str, list[dict]]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    if not path.exists():
        return by_family
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["timestamp"] = datetime.datetime.fromisoformat(row["timestamp"])
            row["rating"] = float(row["rating"])
            row["games_played"] = int(row["games_played"])
            by_family[row["family"]].append(row)
    for rows in by_family.values():
        rows.sort(key=lambda r: r["timestamp"])
    return by_family


def plot_by(by_family: dict[str, list[dict]], x_key: str, x_label: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_series = False
    for family, rows in sorted(by_family.items()):
        if len(rows) < 1:
            continue
        any_series = True
        xs = [r[x_key] for r in rows]
        ys = [r["rating"] for r in rows]
        color = FAMILY_COLORS.get(family, None)
        marker = "o" if len(rows) < 20 else None
        ax.plot(xs, ys, label=family, color=color, marker=marker, markersize=4, linewidth=1.75)

    ax.axhline(1000, color="gray", linewidth=1, linestyle=":", alpha=0.6, label="starting rating (1000)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("rating")
    ax.set_title(title)
    if x_key == "timestamp":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
    if any_series:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data yet -- run snapshot_stats.py a few times first",
                 ha="center", va="center", transform=ax.transAxes, color="gray")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    by_family = load_history(args.data)
    n_points = sum(len(rows) for rows in by_family.values())

    games_path = args.out / "rating_vs_games_played.png"
    time_path = args.out / "rating_vs_time.png"
    plot_by(by_family, "games_played", "games played", "Rating vs. games played, by family", games_path)
    plot_by(by_family, "timestamp", "time", "Rating vs. time, by family", time_path)

    print(f"{n_points} snapshot rows loaded from {args.data}")
    print(f"wrote {games_path}")
    print(f"wrote {time_path}")


if __name__ == "__main__":
    main()
