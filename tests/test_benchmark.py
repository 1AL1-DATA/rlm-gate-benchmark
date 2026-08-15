import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drift import (  # noqa: E402
    classify_miss,
    classify_rows,
    quarter_accuracy,
    value_shape,
)
from src.eval import build_reminders, _age_quarters  # noqa: E402
from src.rolling_window import (  # noqa: E402
    GATE_FRACTIONS,
    TpsCurve,
    simulate_full_context,
    simulate_rolling,
)
from src.sweep import build_prompt  # noqa: E402


def _curve():
    return TpsCurve(
        fills=[1024, 4096, 16384, 65536],
        prefill_tps=[50.0, 200.0, 300.0, 250.0],
        gen_tps=[10.0, 20.0, 22.0, 15.0],
    )


def test_curve_interpolates():
    c = _curve()
    assert c.prefill_tps_at(1024) == 50.0
    assert c.prefill_tps_at(65536) == 250.0
    mid = c.prefill_tps_at((4096 + 16384) / 2)
    assert 200.0 < mid < 300.0
    assert c.prefill_tps_at(0) == 50.0  # clamp low
    assert c.prefill_tps_at(1_000_000) == 250.0  # clamp high


def test_curve_interp_exact():
    c = _curve()
    assert math.isclose(c.prefill_tps_at(16384), 300.0)
    assert math.isclose(c.gen_tps_at(1024), 10.0)


def test_gate_fractions_are_the_plugin_set():
    assert GATE_FRACTIONS == (0.30, 0.60, 0.90)


def test_full_context_always_slower_than_rolling():
    c = _curve()
    total = 245_760
    full = simulate_full_context(c, total)
    roll = simulate_rolling(c, total, window=16384)
    assert full["wall_seconds"] > roll["wall_seconds"]


def test_rolling_window_avg_tps_above_full():
    c = _curve()
    total = 245_760
    full = simulate_full_context(c, total)
    roll = simulate_rolling(c, total, window=16384)
    assert roll["avg_combined_tps"] > full["avg_combined_tps"]


def test_rolling_generates_gate_deposits():
    c = _curve()
    total = 245_760
    roll = simulate_rolling(c, total, window=16384, n_turns=400)
    assert roll["gate_deposits"] >= 1


def test_small_window_has_more_deposits():
    c = _curve()
    total = 245_760
    small = simulate_rolling(c, total, window=8192, n_turns=400)
    large = simulate_rolling(c, total, window=65536, n_turns=400)
    assert small["gate_deposits"] >= large["gate_deposits"]


def test_build_prompt_token_approximation():
    for n in (1024, 16384, 245760):
        p = build_prompt(n)
        words = len(p.split())
        assert abs(words - n) <= 40


def test_prompt_is_repeatable_and_seeded():
    p1, p2 = build_prompt(8192), build_prompt(8192)
    assert p1 == p2


def test_wall_time_is_positive_finite():
    c = _curve()
    for r in (simulate_full_context(c, 65536), simulate_rolling(c, 65536, window=16384)):
        assert r["wall_seconds"] > 0
        assert math.isfinite(r["wall_seconds"])


def test_avg_tps_consistent_with_wall_time():
    c = _curve()
    total = 131_072
    full = simulate_full_context(c, total)
    expected = total / full["wall_seconds"]
    assert abs(full["avg_combined_tps"] - expected) < 0.1


# -- drift taxonomy -----------------------------------------------------------

SEEDED = [("port", "8082"), ("kv_cache", "q8"), ("context", "250K")]


def test_value_shape_classes():
    assert value_shape("8082") == "number"
    assert value_shape("250K") == "number:unit"
    assert value_shape("8GB") == "number:unit"
    assert value_shape("CPU") == "abbrev"
    assert value_shape("q8") == "letter_number"
    assert value_shape("S3") == "letter_number"
    assert value_shape("ninety") == "word"


def test_classify_miss_collision():
    # value present but model answered a different seeded value -> collision
    assert classify_miss("kv_cache", "q8", "250K", "q8 250K RLM", SEEDED) == "collision"


def test_classify_miss_absent():
    # value never made it into the summaries; model substituted a seeded value
    assert classify_miss("port", "8082", "250K", "250K RLM", SEEDED) == "absent"


def test_classify_miss_unbound():
    # value present but the answer is not another seeded value -> unbound
    assert classify_miss("context", "250K", "huge", "q8 250K", SEEDED) == "unbound"


def test_classify_miss_hallucinated():
    # value absent and answer is not a seeded value -> hallucinated
    assert classify_miss("context", "250K", "huge", "q8", SEEDED) == "hallucinated"


def test_classify_miss_paraphrased():
    # value present only in numeric-expanded form -> paraphrased
    assert classify_miss("context", "250K", "250000", "250000 q8", SEEDED) == "paraphrased"


def test_age_quarters_boundaries():
    assert _age_quarters(1.0) == 0  # newest
    assert _age_quarters(0.75) == 1
    assert _age_quarters(0.5) == 2
    assert _age_quarters(0.25) == 3
    assert _age_quarters(0.0) == 3  # oldest clamped


def test_reminder_multiplicity_grows_with_age():
    from src.eval import FACTS
    legend = {"port": "X", "context": "Y"}
    positions = {k: 0.95 for k, *_ in FACTS}  # newest by default
    positions["port"] = 0.05  # oldest -> 4x
    text = build_reminders(legend, positions)
    assert text.count("port: 8082") == 4  # 3 quarter-steps back -> 4x
    assert text.count("context: 250K") == 1  # newest -> 1x


def test_quarter_accuracy_splits_by_age():
    rows = [
        {"quarter": 0, "rlm_rolling_correct": "True"},
        {"quarter": 0, "rlm_rolling_correct": "False"},
        {"quarter": 3, "rlm_rolling_correct": "True"},
    ]
    acc = quarter_accuracy(rows)
    assert acc["0"] == 0.5
    assert acc["3"] == 1.0


def test_classify_rows_only_tags_misses():
    rows = [
        {"fact_key": "port", "expected": "8082", "rlm_rolling_correct": "True",
         "rlm_rolling_answer": "8082"},
        {"fact_key": "context", "expected": "250K", "rlm_rolling_correct": "False",
         "rlm_rolling_answer": "q8"},
    ]
    out = classify_rows(rows, "q8 250K", SEEDED)
    assert len(out) == 1
    assert out[0]["fact_key"] == "context"
    assert out[0]["mode"] == "collision"
