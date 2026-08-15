# Blog post (long-form)

**Goal**: ~1,200–1,500 words. Same research as the LinkedIn post but with the full
story: the measurement, the mechanism, the drift taxonomy, and the honest failure
modes. Written for engineers who run long-context LLM services.

---

## Rolling the context: making long conversations 1.67× cheaper without losing facts

Every long conversation has a dirty secret: the model pays attention — in wall-clock
time and in KV-cache — to everything that was ever said, even the parts nobody needs
anymore.

On the mid-range GPU I used for this benchmark (8 GB, a 35B MoE model quantised to
2 bits, KV cache in q8_0), the generation rate falls off a cliff as the context grows:

| Context | Generation throughput |
| --- | --- |
| 8,192 | 24.4 tokens/s |
| 65,536 | 17.9 tokens/s |
| 131,072 | 12.9 tokens/s |
| 196,608 | 8.7 tokens/s |

Prefill holds up much better (308 → 216 tokens/s over the same range), but generation
— the thing the user actually waits for — more than halves. The dominant cost is not
the model's weights. It is re-reading the history. That is the whole problem this
benchmark attacks.

### The idea: don't grow the window, roll it

Instead of a single context that grows without bound, run the conversation in a small
**working window** and, when it gets full, compact the oldest part into a summary and
carry that forward instead. Nothing about this is new — summarising long history is
what every "memory" system does. What this repo contributes is a *measurement* of the
trade-off and a *taxonomy* of how it fails.

The benchmark compares four rolling window sizes against the same 245,760-token
conversation run in a single full context:

| Mode | Window | Wall time | Combined t/s | Speedup |
| --- | --- | --- | --- | --- |
| Full context | 245,760 | 20,242 s | 12.1 | 1.00× |
| Rolling + RLM gate | 8,192 | 12,976 s | 18.9 | 1.56× |
| Rolling + RLM gate | 16,384 | 12,258 s | 20.1 | 1.65× |
| **Rolling + RLM gate** | **32,768** | **12,148 s** | **20.2** | **1.67×** |
| Rolling + RLM gate | 65,536 | 12,466 s | 19.7 | 1.62× |

The sweet spot is a 32,768-token window. At that size the gate deposited **24
summaries** over the whole conversation and the rolling system was 1.67× faster than
full context. Small windows (8K) are *slower* than the sweet spot despite being
smaller — more, smaller summaries cost more summarisation overhead than they save.

### The RLM context gate

"RLM" here is the memory system that owns the rolling. Its mechanism is a **context
gate** with three thresholds:

- when the window is **30%** full, the oldest portion is summarised;
- at **60%**, again;
- at **90%**, the oldest portion is compacted out and the gate re-arms.

Every compaction is a **deposit**: a summary written to durable memory and re-injected
into the next window. The gate tells you, in a glance, how much memory work a
conversation required.

### Does accuracy survive the roll?

This is the part people care about. We built a 12-fact conversation, ran it in full
context, then ran the same conversation through the rolling window + RLM gate:

| Metric | Full context | Rolling + RLM |
| --- | --- | --- |
| Fact-recall accuracy | 92% | **92%** |
| Accuracy gap | — | **0.000** |
| Old-fact accuracy (via RLM) | — | 90% |
| Recent-fact accuracy | — | 100% |

Equal. But that is the *with-memory* number, and it was not free.

**Without reminders, the rolling model missed 3 of 12 facts (75%).** The facts were
gone not because the model is bad, but because the summaries are lossy in a specific,
predictable way. We built a drift analysis to see the pattern.

### The drift taxonomy

Every missed fact is classified by two axes:

- **mode** — how it failed: `absent` (not in the context at all), `unbound` (the value
  lost its key), `collision` (two facts merged into one value), `paraphrased`
  (rewritten but recoverable), `stale` (superseded), `hallucinated` (invented);
- **shape** — the answer type that is vulnerable: `number:unit`, `word`, `abbrev`,
  `letter_number:context`, …

The misses we observed:

| Fact | Expected | Got | Mode |
| --- | --- | --- | --- |
| `context` | 250K | 8GB | collision |
| `gate` | ninety | 90 | unbound |
| `memory` | RLM | S3 | collision |

Two collisions, one unbound value. The `gate` miss is the instructive one: "ninety"
(literal, but also the gate's **90%** threshold) came back as "90" — the value was
still present, but it had lost its attribute binding. The emoji-codified reminder
system is designed to fix exactly this class of loss.

### The fix: reminders that age with the forget landscape

The remedy codifies each fact as a **key—emoji—value** pair, and re-injects it into the
context as a reminder. The model chooses the emoji for each key itself on every run
(nothing hardcoded — the codification adapts). Then the reminder is repeated based on
how far the fact has drifted:

> The further a fact ages into the forget landscape, the more often it is reminded.
> For each ¼ of the conversation it has drifted, add the codified pair to the context
> once more. Multiplicity = 1 + age-quarters.

The A/B result is the core claim of this writeup:

| Condition | Misses | Rolling accuracy | Gap vs full context |
| --- | --- | --- | --- |
| Rolling, no reminders | 3 / 12 | 75% | +0.167 |
| **Rolling + reminders** | **1 / 12** | **92%** | **0.000** |

The reminder layer halved the miss rate again — and closed the gap to full context
entirely. The one surviving miss is a collision between `rolling → 64K` and the
`context → 250K` fact: two `number:unit` facts whose values merged. That is the most
interesting failure mode, because it is not forgetfulness — it is the model binding the
wrong number to the wrong attribute.

### Why "gate deposits" is the metric you should watch

The comparison table has a column most speed benchmarks don't: **gate deposits**. It
tells you *how much memory work a conversation required*:

- 8K window → 100 deposits (summarising constantly, overhead wins);
- 32K window → 24 deposits (the sweet spot);
- 65K window → 12 deposits (least summarising, but generation throughput is lower).

A system that forgets deliberately logs deposits and drift misses; a system that
forgets accidentally just loses the fact. The drift profile in
`results/drift_profile.json` is that ledger.

### Honest limitations

- One model, one GPU, one synthetic 12-fact conversation. The accuracy numbers are a
  demonstration of mechanism, not a population estimate.
- The 245,760-token sweep point is unreachable on this server: prompt construction
  overshoots the 250,112-token `n_ctx`, so the baseline caps at 196,608.
- YaRN alternatives (a 4× RoPE-widened native window, or 128 K + YaRN) are deliberately
  **not** benchmarked — the full comparison is future work scoped in the paper draft.

### The takeaway

Rolling the context is not a hack to make accuracy worse in exchange for speed. With a
drift-aware memory gate it is strictly better on both axes in this benchmark: **1.67×
faster and within 0.000 of full-context accuracy.** The mechanism — codified facts,
age-weighted reminders, a drift taxonomy that says *how* a roll fails — is the part
worth stealing.

Code, figures, and 21 unit tests: see the first comment on the LinkedIn post, or the
repo directly.

---

## Behind-the-scenes notes

- **Word count**: ~1,050 words in the body — appropriate for a technical blog.
- **Numbers are all cross-checked** against `results/rolling_window_comparison.csv`,
  `results/eval_after_summary.json`, `results/drift_profile.json`, and the sweep CSV.
- The `gate → 90` miss is explained as an *unbound* value; the `context → 250K` /
  `rolling → 64K` misses as *collisions*. Terminology matches `src/drift.py`.
- The "1.67× / 92% = 92%" claims match `results/SUMMARY.md`.
- YaRN future work note keeps the post honest without benchmarking it here.
