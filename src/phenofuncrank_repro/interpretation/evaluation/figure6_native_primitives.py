"deterministic functions from Figure6 judge/aggregate sources."

from __future__ import annotations

from collections import Counter

from typing import Any

QUALITY = {"yes":2,"partial":1,"no":0}

def frac(n: int, d: int) -> dict[str, Any]:
    return {"numerator": n, "denominator": d, "rate": round(n / d, 4) if d else None}

def metric(rows: list[dict[str, Any]], field: str, *, error: bool) -> dict[str, Any]:
    values = []
    for row in rows:
        order = row.get("arm_order_private") or {}
        c = row.get("consensus_blind") or {}
        full_label = next((label for label, arm in order.items() if arm == "full"), None)
        vanilla_label = next((label for label, arm in order.items() if arm == "vanilla"), None)
        if not full_label or not vanilla_label:
            continue
        values.append(((c.get(full_label) or {}).get(field), (c.get(vanilla_label) or {}).get(field)))
    if error:
        full_yes = sum(a == "yes" for a, _ in values)
        vanilla_yes = sum(b == "yes" for _, b in values)
        return {
            "full_error_rate": frac(full_yes, len(values)),
            "vanilla_error_rate": frac(vanilla_yes, len(values)),
            "full_improves_error": sum(b == "yes" and a == "no" for a, b in values),
            "full_worsens_error": sum(b == "no" and a == "yes" for a, b in values),
            "full_not_worse": frac(sum(not (b == "no" and a == "yes") for a, b in values), len(values)),
        }
    full_yes = sum(a == "yes" for a, _ in values)
    vanilla_yes = sum(b == "yes" for _, b in values)
    better = sum(QUALITY.get(a, -1) > QUALITY.get(b, -1) for a, b in values)
    worse = sum(QUALITY.get(a, -1) < QUALITY.get(b, -1) for a, b in values)
    return {
        "full_yes": frac(full_yes, len(values)),
        "vanilla_yes": frac(vanilla_yes, len(values)),
        "full_minus_vanilla_yes_rate_pp": round((full_yes - vanilla_yes) * 100 / len(values), 2) if values else None,
        "paired_full_better": better, "paired_full_worse": worse, "paired_tie": len(values) - better - worse,
        "full_not_worse": frac(better + len(values) - better - worse, len(values)),
        "distribution_full": dict(Counter(a for a, _ in values)),
        "distribution_vanilla": dict(Counter(b for _, b in values)),
    }

def quality(value: str) -> int:
    return {"yes": 2, "partial": 1, "no": 0}.get(value, -1)

def paired_metric(rows: list[dict[str, Any]], field: str, *, error: bool = False) -> dict[str, Any]:
    t0 = [row["t0"].get(field) for row in rows]
    t1 = [row["t1"].get(field) for row in rows]
    if error:
        t0_positive = sum(x == "yes" for x in t0)
        t1_positive = sum(x == "yes" for x in t1)
        improved = sum(a == "yes" and b == "no" for a, b in zip(t0, t1))
        worsened = sum(a == "no" and b == "yes" for a, b in zip(t0, t1))
        return {
            "t0_error_rate": frac(t0_positive, len(t0)),
            "t1_error_rate": frac(t1_positive, len(t1)),
            "improved_error_cases": improved,
            "worsened_error_cases": worsened,
            "t1_not_worse": frac(sum(not (a == "no" and b == "yes") for a, b in zip(t0, t1)), len(t0)),
        }
    t0_yes = sum(x == "yes" for x in t0)
    t1_yes = sum(x == "yes" for x in t1)
    gains = sum(quality(b) > quality(a) for a, b in zip(t0, t1))
    losses = sum(quality(b) < quality(a) for a, b in zip(t0, t1))
    ties = len(t0) - gains - losses
    return {
        "t0_yes": frac(t0_yes, len(t0)),
        "t1_yes": frac(t1_yes, len(t1)),
        "t1_minus_t0_yes_rate_pp": round((t1_yes - t0_yes) * 100 / len(t0), 2) if t0 else None,
        "paired_t1_better": gains,
        "paired_t1_worse": losses,
        "paired_tie": ties,
        "paired_t1_not_worse": frac(gains + ties, len(t0)),
        "t0_distribution": dict(Counter(t0)),
        "t1_distribution": dict(Counter(t1)),
        "t0_mean_ordinal": round(sum(quality(x) for x in t0) / len(t0), 3) if t0 else None,
        "t1_mean_ordinal": round(sum(quality(x) for x in t1) / len(t1), 3) if t1 else None,
        "t1_minus_t0_mean_ordinal": round((sum(quality(x) for x in t1) - sum(quality(x) for x in t0)) / len(t0), 3) if t0 else None,
    }

def majority(values: list[str]) -> str:
    counts = Counter(values)
    if not counts:
        return "NEEDS_HUMAN_REVIEW"
    value, count = counts.most_common(1)[0]
    return value if value != "NEEDS_HUMAN_REVIEW" and count >= 2 else "NEEDS_HUMAN_REVIEW"

def rate(values: list[str], value: str = "yes") -> float | None:
    usable = [x for x in values if x in {"yes", "no", "not_applicable"}]
    if not usable:
        return None
    return round(sum(x == value for x in usable) / len(usable), 4)
