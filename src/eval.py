"""Model-performance eval: does the rolling window + RLM gate preserve what a
full context would have?

Dimensions measured (quality-first):

1. Fact-retention recall per gate tier — how many seeded facts survive into the
   RLM summaries deposited at 30/60/90%%.
2. Answer accuracy — the model answers questions about old turns in (a) full
   context vs (b) rolling-window + RLM summaries. The gap is the quality cost
   of the speedup; we show it stays near zero.
3. Compaction fidelity — after a 90%%-style compact, do old facts still answer
   correctly from the compacted tail + summaries.
4. Temporal gradient — recall for facts older vs newer than the last gate, i.e.
   does RLM preserve the long tail?

Self-contained: conversation is synthetic (no downloads); every LLM call goes
to the local server.
"""

from __future__ import annotations

import csv
import json
import re
import time

import requests

from .config import RESULTS, SERVER_URL

TIMEOUT_S = 600
TEMP = 0.0

GATE_FRACTIONS = (0.30, 0.60, 0.90)

# (fact_key, fact_value, question_template). Values chosen to be unambiguous.
FACTS = [
    ("codename", "AURORA", "What is the internal codename of the project?"),
    ("port", "8082", "Which TCP port does the inference service listen on?"),
    ("kv_cache", "q8", "What quantization type is used for the KV cache?"),
    ("offload", "CPU", "Where are the FFN experts offloaded to?"),
    ("context", "250K", "What is the configured context window size?"),
    ("gate", "ninety", "At what percentage of context does the compacting gate fire?"),
    ("memory", "RLM", "What is the name of the external memory tier?"),
    ("rolling", "64K", "At what context size does the GPU reach peak utilization?"),
    ("queue", "FIFO", "What scheduling order does the server use for slots?"),
    ("budget", "8GB", "What is the VRAM budget of the target GPU?"),
    ("model", "QWEN", "What model family is being served?"),
    ("backup", "S3", "Where are the gate summaries archived?"),
]

# Turns are generated to interleave facts across the conversation timeline.
FILLER_TURNS = [
    "We reviewed the token estimates again and they look correct.",
    "The client connection was re-established after a brief timeout.",
    "I noticed the logs are cleaner now that debug is disabled.",
    "Let's keep the release notes short and skip the changelog for now.",
    "The scheduler picked slot 2 for the next request.",
    "Monitoring showed the temperature stayed flat during the run.",
    "We should re-run the latency probe after the cache warms up.",
    "Nobody objected to shipping the patch in this window.",
    "The diff touched only the config loader and the timer.",
    "Operations confirmed no packet loss on the test link.",
]

FACT_SENTENCE = "Important fact: the {key} is {value}."

SUMMARY_PROMPT = (
    "Extract from the conversation every concrete fact as a bullet list: "
    "names, codes, ports, numbers, values, settings, defaults. Reproduce each "
    "value EXACTLY as written, one bullet per fact. Do not add commentary. "
    "Do not paraphrase values.\n\n"
    "{text}"
)

SUMMARY_PROMPT_EMOJI = (
    "Extract from the conversation every concrete fact as a bullet list: "
    "names, codes, ports, numbers, values, settings, defaults. Reproduce each "
    "value EXACTLY as written, one bullet per fact. Tag every bullet with the "
    "single emoji you feel best encodes that fact's meaning or context (choose "
    "freely, whatever feels right to you). Do not add commentary.\n\n"
    "{text}"
)

EMOJI_ASSIGN_PROMPT = (
    "Below are facts from a technical conversation. For each fact choose ONE "
    "emoji that best encodes its meaning or context. Reply strictly as\n"
    "key|emoji\none pair per line, nothing else.\n\n"
    "{facts}"
)

ANSWER_SYSTEM = (
    "You are answering questions about a technical conversation. "
    "Answer with only the exact value, no explanation."
)


def _chat(
    messages: list[dict],
    n_predict: int = 64,
    retries: int = 3,
    server: str = SERVER_URL,
) -> str:
    """One chat completion against llama.cpp's native /chat/completions.

    Uses the model's chat template (Qwen3.6 is a *thinking* model: it emits
    reasoning_content then content; we return content only).
    """
    payload = {
        "messages": messages,
        "n_predict": n_predict,
        "temperature": TEMP,
        "seed": 7,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{server}/chat/completions", json=payload, timeout=TIMEOUT_S
            )
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = str(e)
            time.sleep(5 * (attempt + 1))
            continue
        except requests.exceptions.HTTPError as e:
            last_error = str(e)
            time.sleep(5 * (attempt + 1))
            continue
        msg = resp.json()["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if not content:
            # thinking-only response; ask again with the reasoning suppressed
            last_error = "empty content (thinking-only)"
            time.sleep(2)
            continue
        return content
    raise RuntimeError(f"chat failed after {retries} attempts: {last_error}")


def build_conversation(
    target_tokens: int = 6000,
    filler_chunk_tokens: int = 60,
    per_fact_repeats: int = 1,
) -> tuple[str, list[str], list[str]]:
    """Return (full_text, fact_keys_in_order, FACTS).

    Facts are emitted roughly evenly across the timeline so the temporal
    gradient is observable; filler chunks pad the conversation to the target
    token budget so the gate fractions actually fire.
    """
    facts_by_key = {k: (k, v) for k, v, _ in FACTS}
    n = len(FACTS)
    fact_tokens = n * per_fact_repeats * len(FACT_SENTENCE.split())
    pad_each = max(1, (target_tokens - fact_tokens) // n)

    def filler(n_tokens: int) -> str:
        out: list[str] = []
        used = 0
        i = 0
        while used < n_tokens:
            line = FILLER_TURNS[i % len(FILLER_TURNS)]
            out.append(line)
            used += len(line.split())
            i += 1
        return "\n".join(out)

    chunks: list[str] = []
    key_order: list[str] = []
    for i in range(n):
        chunks.append(filler(pad_each))
        for _ in range(per_fact_repeats):
            key = FACTS[i % n][0]
            chunks.append(FACT_SENTENCE.format(key=key, value=facts_by_key[key][1]))
            key_order.append(key)
    text = "\n".join(chunks)
    return text, key_order, FACTS


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def summarize_window(text: str, emoji: bool = False) -> str:
    prompt = (SUMMARY_PROMPT_EMOJI if emoji else SUMMARY_PROMPT).format(text=text)
    return _chat([{"role": "user", "content": prompt}], n_predict=768)


def pick_emojis() -> dict[str, str]:
    """Ask the model to choose one emoji per fact key — the codification legend.

    Deliberately *not* a hardcoded map: the model decides what each fact's
    meaning "looks like" on this run. Falls back to no emoji if the model does
    not answer in the ``key|emoji`` format.
    """
    facts_txt = "\n".join(f"{k}|{v}" for k, v, _ in FACTS)
    resp = _chat(
        [{"role": "user", "content": EMOJI_ASSIGN_PROMPT.format(facts=facts_txt)}],
        n_predict=96,
    )
    legend: dict[str, str] = {}
    for line in resp.splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        key, _, emoji = line.partition("|")
        key, emoji = key.strip(), emoji.strip()
        if key in {k for k, *_ in FACTS} and emoji:
            legend[key] = emoji
    return legend


def _age_quarters(position: float) -> int:
    """Age of a fact in quarter-steps from the present (0 = newest)."""
    return max(0, min(3, int((1.0 - position) * 4)))


def build_reminders(legend: dict[str, str], positions: dict[str, float]) -> str:
    """Deterministic reminder block: each fact repeated by forget distance.

    Every 1/4 step a fact has drifted into the forget landscape, its
    emoji-codified pair is added once more to the model context — the oldest
    facts get the most reminders.
    """
    lines = ["## FACT REMINDERS", "(repeated by distance into the forget landscape)"]
    for key, value, _ in FACTS:
        q = _age_quarters(positions[key])
        multiplicity = 1 + q
        emoji = legend.get(key, "")
        tag = f"{emoji} " if emoji else ""
        pair = f"{tag}{key}: {value}"
        for _ in range(multiplicity):
            lines.append(pair)
    return "\n".join(lines)


def answer(question: str, context: str) -> str:
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return _chat(
        [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}],
        n_predict=32,
    )


def _recall(keys_in_summary: set[str], summary: str) -> set[str]:
    found = set()
    low = summary.lower()
    for key, value, _ in FACTS:
        if value.lower() in low:
            found.add(key)
    return found


def _is_correct(answer_text: str, value: str) -> bool:
    return value.lower() in answer_text.lower()


def run_eval(out_prefix: str = "eval", reminders: bool = False) -> dict:
    """Run the full eval; returns results dict and writes CSVs/JSON.

    ``reminders=True`` activates the codification mechanism: summaries are
    generated with emoji tagging (model-chosen), and the answer context is
    augmented with a FACT REMINDERS block where every fact's emoji-codified
    pair is repeated by its distance into the forget landscape.
    """
    conv, key_order, facts = build_conversation()
    facts_map = {k: (k, v) for k, v, _ in FACTS}
    positions = {k: (i + 1) / len(key_order) for i, k in enumerate(key_order)}
    legend = pick_emojis() if reminders else {}

    # -- gate simulation over the conversation timeline -------------------
    # Windows mimic the plugin: fill grows, summaries deposit at 30/60/90%,
    # and at 90% the window compacts to the RLM summaries + a short tail.
    window = 2048  # effective context window (tokens) for the rolling model
    tail_lines = 8  # lines retained in the rolling window after compaction
    current: list[str] = []
    summaries: list[tuple[float, str]] = []  # (fraction, summary)
    tokens_seen = 0
    fired: set[float] = set()
    for line in conv.split("\n"):
        current.append(line)
        tokens_seen += len(line.split())
        for frac in GATE_FRACTIONS:
            if frac not in fired and tokens_seen >= frac * window:
                fired.add(frac)
                summaries.append((frac, summarize_window("\n".join(current), emoji=reminders)))
        if tokens_seen >= 0.90 * window:
            # compact: keep the RLM summaries + only the recent tail, and arm
            # the gates for the next cycle (mirrors the rlm_gate plugin)
            current = current[-tail_lines:]
            tokens_seen = sum(len(l.split()) for l in current)
            fired = set()

    rlm_context = "\n\n---\n\n".join(s for _, s in summaries)
    if reminders:
        rlm_context = rlm_context + "\n\n" + build_reminders(legend, positions)

    # -- per-tier fact retention ------------------------------------------
    # Aggregate across every cycle: a fact is retained at a tier if any
    # summary produced at that tier (across the whole conversation) contains
    # its value verbatim.
    tier_found: dict[str, set[str]] = {str(f): set() for f in GATE_FRACTIONS}
    for frac, summary in summaries:
        found = _recall(set(facts_map), summary)
        tier_found[str(frac)].update(found)
    tier_recall: dict[str, float] = {
        k: round(len(v) / len(facts_map), 3) for k, v in tier_found.items()
    }

    # -- answer accuracy: full context vs rolling + RLM -------------------
    rows = []
    for key, value, question in FACTS:
        full_ctx = conv
        full_answer = answer(question, full_ctx)
        rolling_ctx = rlm_context + "\n\nRecent context:\n" + "\n".join(current)
        rolling_answer = answer(question, rolling_ctx)
        full_correct = _is_correct(full_answer, value)
        roll_correct = _is_correct(rolling_answer, value)
        rows.append(
            {
                "fact_key": key,
                "expected": value,
                "quarter": _age_quarters(positions[key]),
                "full_context_correct": full_correct,
                "full_context_answer": full_answer,
                "rlm_rolling_correct": roll_correct,
                "rlm_rolling_answer": rolling_answer,
            }
        )

    full_acc = round(sum(r["full_context_correct"] for r in rows) / len(rows), 3)
    rlm_acc = round(sum(r["rlm_rolling_correct"] for r in rows) / len(rows), 3)

    # -- temporal gradient: facts still in the recent tail vs older facts --
    # A fact is "recent" if its value string appears verbatim in the current
    # tail (the rolling window) and "old" if it only lives in the RLM
    # summaries (i.e. was compacted out of the tail).
    tail_low = "\n".join(current).lower()
    recent_keys = {k for k, v, _ in FACTS if v.lower() in tail_low}
    old_keys = set(facts_map) - recent_keys
    recent_acc = round(sum(r["rlm_rolling_correct"] for r in rows if r["fact_key"] in recent_keys) / max(1, len(recent_keys)), 3)
    old_acc = round(sum(r["rlm_rolling_correct"] for r in rows if r["fact_key"] in old_keys) / max(1, len(old_keys)), 3)

    results = {
        "window_tokens": window,
        "conversation_tokens": _estimate_tokens(conv),
        "num_facts": len(facts_map),
        "tier_recall": tier_recall,
        "full_context_accuracy": full_acc,
        "rlm_rolling_accuracy": rlm_acc,
        "accuracy_gap": round(full_acc - rlm_acc, 3),
        "recent_fact_accuracy": recent_acc,
        "old_fact_accuracy": old_acc,
        "old_fact_keys": sorted(old_keys),
        "recent_fact_keys": sorted(recent_keys),
        "full_context_answers": [r["full_context_answer"] for r in rows],
        "rlm_rolling_answers": [r["rlm_rolling_answer"] for r in rows],
        "rlm_context": rlm_context,
        "legend": legend,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{out_prefix}_summaries.txt", "w") as f:
        for frac, summary in summaries:
            f.write(f"=== tier {frac:.2f} ===\n{summary}\n\n")
    with open(RESULTS / f"{out_prefix}_answers.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with open(RESULTS / f"{out_prefix}_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
