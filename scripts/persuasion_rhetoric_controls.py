"""Controls for the persuasion framing result, splitting the data three ways.

Re-runs the chi-square test within product-quality, round-bucket and
value-to-price strata to check that the framing effect is not an artifact of
game state.
"""

from __future__ import annotations

from collections import defaultdict

from scipy.stats import chi2_contingency

from scripts.moral_foundation_analysis import FOUNDATIONS, dominant_foundation_for_message, load_latest_games


def collect_rows() -> list[dict]:
    games = load_latest_games()
    rows = []
    for d in games.values():
        if d["family"] != "persuasion":
            continue
        state = d["game_state"]
        me = d["your_player"]
        if state.get("seller_message_type") != "text":
            continue
        role = state.get(f"{me}_role") or state.get("your_role")
        if role != "seller":
            continue
        v, price = state.get("v"), state.get("product_price")
        ratio = (v / price) if (v is not None and price) else None
        for h in state.get("history", []):
            msg = h.get("seller_message")
            bought = h.get("bought")
            quality = h.get("quality")
            rnd = h.get("round")
            if msg is None or bought is None:
                continue
            fnd = dominant_foundation_for_message(msg)
            if fnd is None:
                continue
            rows.append(dict(fnd=fnd, quality=quality, round=rnd, bought=bought, ratio=ratio))
    return rows


def rate(rs: list[dict]) -> float:
    return sum(r["bought"] for r in rs) / len(rs) if rs else float("nan")


def chi2_table(subset: list[dict], label: str) -> None:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in subset:
        counts[r["fnd"]][0 if r["bought"] else 1] += 1
    table = [(fnd, sum(counts[fnd]), counts[fnd][0],
              counts[fnd][0] / sum(counts[fnd]) if sum(counts[fnd]) else float("nan"))
             for fnd in FOUNDATIONS]
    total = sum(n for _, n, _, _ in table)
    contingency = [[counts[fnd][0], counts[fnd][1]] for fnd in FOUNDATIONS]
    try:
        chi2, p, dof, _ = chi2_contingency(contingency)
    except ValueError as e:
        print(f"{label}: chi2 failed ({e})")
        return
    print(f"\n-- {label} (n={total}) --")
    for fnd, n, succ, r in sorted(table, key=lambda t: -t[3] if t[3] == t[3] else 0):
        print(f"  {fnd:<10} n={n:>5} success={succ:>5} rate={r:.1%}")
    print(f"  chi2({dof})={chi2:.1f}, p={p:.3g}")


def bucket(round_num: int) -> str:
    if round_num <= 6:
        return "early (1-6)"
    if round_num <= 13:
        return "middle (7-13)"
    return "late (14-20)"


def main() -> None:
    rows = collect_rows()
    print(f"{len(rows)} total classified messages with quality/round/ratio data")

    # Low-quality rounds are where the seller must persuade against the truth, so
    # the framing gap is expected to be widest there.
    low_q = [r for r in rows if r["quality"] == "low"]
    high_q = [r for r in rows if r["quality"] == "high"]
    chi2_table(low_q, "LOW quality (bluff rounds)")
    chi2_table(high_q, "HIGH quality rounds")

    care_low = [r for r in low_q if r["fnd"] == "care"]
    loy_low = [r for r in low_q if r["fnd"] == "loyalty"]
    print(f"\ncare vs loyalty on LOW quality: care n={len(care_low)} rate={rate(care_low):.1%}, "
          f"loyalty n={len(loy_low)} rate={rate(loy_low):.1%}")
    contingency = [
        [sum(r["bought"] for r in care_low), len(care_low) - sum(r["bought"] for r in care_low)],
        [sum(r["bought"] for r in loy_low), len(loy_low) - sum(r["bought"] for r in loy_low)],
    ]
    chi2, p, dof, _ = chi2_contingency(contingency)
    print(f"care-vs-loyalty chi2({dof})={chi2:.1f}, p={p:.3g}, n={len(care_low) + len(loy_low)}")

    buckets = defaultdict(list)
    for r in rows:
        if r["round"] is not None:
            buckets[bucket(r["round"])].append(r)
    for label in ("early (1-6)", "middle (7-13)", "late (14-20)"):
        chi2_table(buckets[label], label)

    print("\ncare's rate across round buckets:")
    for label in ("early (1-6)", "middle (7-13)", "late (14-20)"):
        care_rows = [r for r in buckets[label] if r["fnd"] == "care"]
        print(f"  {label}: n={len(care_rows)} rate={rate(care_rows):.1%}")
    print("loyalty's rate across round buckets:")
    for label in ("early (1-6)", "middle (7-13)", "late (14-20)"):
        loy_rows = [r for r in buckets[label] if r["fnd"] == "loyalty"]
        print(f"  {label}: n={len(loy_rows)} rate={rate(loy_rows):.1%}")

    print("\nvalue/price ratio range by foundation arm:")
    for fnd in FOUNDATIONS:
        ratios = [r["ratio"] for r in rows if r["fnd"] == fnd and r["ratio"] is not None]
        if ratios:
            print(f"  {fnd:<10} n={len(ratios):>5} min={min(ratios):.2f} max={max(ratios):.2f} "
                  f"mean={sum(ratios) / len(ratios):.2f}")


if __name__ == "__main__":
    main()
