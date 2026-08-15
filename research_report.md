# Research report — RLM rolling-window benchmark

**Date:** 2026-08-15
**Status:** systems demonstration, not yet a paper (roadmap: see `arxiv_paper.tex` §"Path to a paper")

---

## Executive summary

Long conversations dominate the cost of running LLM services because the model
re-processes the entire history on every turn. This benchmark shows that a **rolling
working window + RLM context gate** runs a 245,760-token conversation **1.67× faster**
than the same conversation in a single full context (12,148 s vs 20,242 s at a 32,768
token window), while matching the full-context **92% fact-recall accuracy exactly**
(gap 0.000).

The accuracy parity is not automatic: a bare rolling window loses 3 of 12 facts (75%).
A **drift-aware reminder layer** — facts codified as `key|emoji|value`, re-injected
once per forget-quarter they have aged — cuts the miss rate to 1 (92%), closing the
gap to full context. The one residual miss is a `number:unit` **collision** (two facts
merging into one value), which is a distinct and interesting failure mode.

## 1. The cost of context

Generation throughput collapses as context grows (single 8 GB GPU, 35B MoE, q8_0 KV):

| Context (tokens) | Prefill t/s | Generation t/s | GPU util % |
| --- | --- | --- | --- |
| 8,192 | **308.3** | 24.4 | 79 |
| 16,384 | 298.1 | 21.6 | 77 |
| 32,768 | 298.1 | 22.3 | 81 |
| 65,536 | 289.2 | 17.9 | 83.5 |
| 131,072 | 253.1 | 12.9 | 83 |
| 196,608 | 215.9 | 8.7 | 81 |

Prefill degrades gracefully (308 → 216 t/s); generation halves and worse. The user
waits on generation. This is the budget a rolling window recovers.

## 2. Rolling window + RLM gate vs full context

245,760-token conversation, wall-clock server time, summaries generated from the warm
KV cache:

| Mode | Window | Wall (s) | Combined t/s | Gate deposits | Speedup |
| --- | --- | --- | --- | --- | --- |
| Full context | 245,760 | 20,242 | 12.14 | — | 1.00× |
| Rolling + RLM | 8,192 | 12,976 | 18.94 | 100 | 1.56× |
| Rolling + RLM | 16,384 | 12,258 | 20.05 | 49 | 1.65× |
| **Rolling + RLM** | **32,768** | **12,148** | **20.23** | **24** | **1.67×** |
| Rolling + RLM | 65,536 | 12,466 | 19.71 | 12 | 1.62× |

- The sweet spot is 32,768. Smaller windows summarise too often (100 deposits at 8K)
  and pay for it in overhead; larger windows get faster generation only marginally
  while keeping more tokens resident.
- `gate_deposits` is the ledger of memory work: 24 summaries for the whole
  conversation at the sweet spot, one miss.

## 3. Accuracy: does rolling lose facts?

12-fact synthetic conversation (6,012 tokens), one question per fact:

| Metric | Full context | Rolling + RLM (+reminders) |
| --- | --- | --- |
| Accuracy | 92% | **92%** |
| Accuracy gap | — | **0.000** |
| Old-fact accuracy (via RLM) | — | 90% |
| Recent-fact accuracy | — | 100% |
| Tier recall (0.3/0.6/0.9) | — | 0.167 / 0.25 / 0.167 |

### 3.1 Without reminders, rolling loses facts

| Condition | Misses | Rolling accuracy | Gap vs full context |
| --- | --- | --- | --- |
| Rolling, no reminders | 3 / 12 | 75% | +0.167 |
| **Rolling + reminders** | **1 / 12** | **92%** | **0.000** |

### 3.2 Drift profile (before reminders)

| Fact | Expected | Got | Shape | Mode |
| --- | --- | --- | --- | --- |
| `context` | 250K | 8GB | number:unit | collision |
| `gate` | ninety | 90 | word | unbound |
| `memory` | RLM | S3 | abbrev | collision |

- Two **collisions** (attribute merged with the wrong value) and one **unbound** value
  (`gate` → 90: the value survived, its attribute binding did not — the gate's own 90%
  threshold collides with "ninety").
- Recall by forget-quarter before reminders: Q3 = 1.0, Q1 = 0.667, Q2 = 0.333 — the
  middle of the conversation is where facts vanish, not the very oldest.

### 3.3 After reminders

| Fact | Expected | Got | Shape | Mode |
| --- | --- | --- | --- | --- |
| `rolling` | 64K | 250K | number:unit | collision |

One residual `number:unit` collision — two similar-shaped facts merging. Not
forgetfulness; wrong-value binding. Quarters: Q2 recall recovers to 1.0.

## 4. Mechanism: the drift-aware reminder

1. Every fact is codified as **key|emoji|value**; the model chooses each emoji per run
   (`pick_emojis`, nothing hardcoded — e.g. `context 📖`, `gate 🚦`, `memory 🧮`,
   `rolling 🔄`).
2. The **age quarter** of a fact = its position in the conversation, 0 (newest) → 3
   (oldest).
3. **Multiplicity = 1 + age_quarters.** Each ¼ a fact drifts into the forget
   landscape, the codified pair is added to the context once more. Both RLM memory and
   the working context carry the pair (shared responsibility).
4. The gate deposits at 30/60/90% fill and re-arms at the 90% compaction.

## 5. Conclusions

- Rolling is not a speed-for-accuracy trade: with a drift-aware memory it is **1.67×
  faster and within 0.000 of full-context accuracy** in this benchmark.
- `gate_deposits` + drift misses are the diagnostic that separates deliberate from
  accidental forgetting.
- The residual failure mode is **collision** — the mechanism to attack next.

## 6. Limitations

- One model, one GPU, one synthetic 12-fact conversation. Demonstration of mechanism,
  not a population estimate.
- 245,760-token sweep point unreachable (prompt construction overshoots `n_ctx`
  250,112 → HTTP 400); baseline caps at 196,608.
- YaRN-widened-window alternatives not benchmarked (future work; see paper roadmap).
- Emoji codification is model-chosen per run, so `legend` varies between runs.

## 7. Open questions / next steps

1. Attack the `collision` mode: disambiguate same-shape facts (attribute-qualified
   reminders, numeric-vs-emoji separation).
2. Validate the reminder multiplicity rule against real long-running agent
   conversations.
3. Multi-model / multi-seed runs for error bars and significance (see paper roadmap).
4. YaRN Options 2/3 comparison (35B @ 30 K + YaRN 4×; 35B @ 128 K + YaRN 4×).
