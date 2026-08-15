"""Drift taxonomy: classify rolling-window + RLM misses into failure modes.

A miss is not just "wrong" — we need to know *why* the information was lost so
the fix can be targeted. Modes:

- absent:       the correct value never made it into any RLM summary
- unbound:      the value is present in the summaries but the attribute binding
                was lost (the model answered the wrong shape/value anyway)
- collision:    the value is present but the model answered a *different*
                seeded value (value-selection error in the compressed tail)
- paraphrased:  the value is present only in an alternate textual form
- hallucinated: the answer is neither in the summaries nor any seeded value
- stale:        reserved for update facts (value changed mid-conversation)
"""

from __future__ import annotations

import csv
import json
import re

from .config import RESULTS

MODES = ["absent", "unbound", "collision", "paraphrased", "hallucinated", "stale"]

_NUMBER_RE = re.compile(r"^\d+$")
_NUMBER_UNIT_RE = re.compile(r"^\d+[A-Za-z]+$")
_ABBREV_RE = re.compile(r"^[A-Z]{2,6}$")
_LETTER_NUMBER_RE = re.compile(r"^[A-Za-z]*\d+[A-Za-z]*$")
_WORD_RE = re.compile(r"^[a-z]+$")

_UNIT_MULT = {"k": 1000, "kb": 1024, "m": 1_000_000, "g": 1_000_000_000}


def value_shape(value: str) -> str:
    """Syntactic shape of a fact value (drives the codification legend)."""
    v = value.strip()
    if _NUMBER_RE.match(v):
        return "number"
    if _NUMBER_UNIT_RE.match(v):
        return "number:unit"
    if _ABBREV_RE.match(v):
        return "abbrev"
    if _LETTER_NUMBER_RE.match(v):
        return "letter_number"
    if _WORD_RE.match(v):
        return "word"
    return "other"


def paraphrase_forms(value: str) -> list[str]:
    """Alternate textual forms of a value we accept as "paraphrased present"."""
    forms = {value, value.lower()}
    m = re.match(r"^(\d+)([A-Za-z]+)$", value)
    if m:
        num, unit = m.group(1), m.group(2).lower()
        mult = _UNIT_MULT.get(unit, _UNIT_MULT.get(unit.rstrip("b"), 1))
        forms.add(str(int(num) * mult))
        forms.add(f"{num} {unit}")
        forms.add(f"{num}{unit}")
    return [f for f in forms if f]


def classify_miss(
    fact_key: str, expected: str, answer: str, rlm_context: str, seeded: list[tuple[str, str]]
) -> str:
    """Classify one rolling-condition miss into a failure mode.

    ``seeded`` is the ground-truth (key, value) list, so "collision" is
    detected when the model substitutes a *different seeded value*.
    """
    answer_low = (answer or "").lower()
    ctx_low = (rlm_context or "").lower()

    answer_is_seeded = any(k != fact_key and v.lower() in answer_low for k, v in seeded)

    if expected.lower() not in ctx_low:
        if any(p in ctx_low for p in paraphrase_forms(expected)):
            return "paraphrased"
        return "absent" if answer_is_seeded else "hallucinated"

    return "collision" if answer_is_seeded else "unbound"


def _quarter_of(position: float) -> int:
    """Age of a fact in quarter-steps from the present (0 = newest)."""
    return max(0, min(3, int((1.0 - position) * 4)))


def classify_rows(
    rows: list[dict], rlm_context: str, seeded: list[tuple[str, str]]
) -> list[dict]:
    """Tag every rolling-condition miss in ``rows`` with its failure mode."""
    out = []
    for r in rows:
        if str(r.get("rlm_rolling_correct", "")).lower() == "true":
            continue
        mode = classify_miss(
            r["fact_key"], r["expected"], r["rlm_rolling_answer"], rlm_context, seeded
        )
        out.append(
            {
                "fact_key": r["fact_key"],
                "expected": r["expected"],
                "shape": value_shape(r["expected"]),
                "mode": mode,
                "answer": r["rlm_rolling_answer"],
            }
        )
    return out


def quarter_accuracy(rows: list[dict]) -> dict[str, float]:
    """Rolling-condition accuracy per age quarter (0..3), for the forget curve."""
    acc: dict[str, list[bool]] = {"0": [], "1": [], "2": [], "3": []}
    for r in rows:
        acc[str(r["quarter"])].append(str(r["rlm_rolling_correct"]).lower() == "true")
    return {q: round(sum(v) / len(v), 3) if v else None for q, v in acc.items()}


def build_drift_profile(
    before: list[dict],
    after: list[dict],
    seeded: list[tuple[str, str]],
    before_ctx: str = "",
    after_ctx: str = "",
) -> dict:
    """Aggregate miss modes, shapes and quarter accuracy into drift_profile.json."""
    bl = classify_rows(before, before_ctx, seeded)
    al = classify_rows(after, after_ctx, seeded)

    def cond(rows, misses):
        modes = {m: 0 for m in MODES}
        shapes: dict[str, int] = {}
        for x in misses:
            modes[x["mode"]] += 1
            shapes[x["shape"]] = shapes.get(x["shape"], 0) + 1
        return {
            "n_misses": len(misses),
            "modes": modes,
            "shapes": shapes,
            "quarters": quarter_accuracy(rows),
            "misses": misses,
        }

    profile = {
        "before": cond(before, bl),
        "after": cond(after, al),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "drift_profile.json", "w") as f:
        json.dump(profile, f, indent=2)
    return profile
