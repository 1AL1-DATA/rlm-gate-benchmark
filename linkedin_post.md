# LinkedIn post (short-form)

**Goal**: ~250 words. Warm, honest, research-sharing. Narrative arc from question → hypothesis → findings → meaning. GitHub URL will be filled in after the repo is pushed.

---

## The version to copy-paste into LinkedIn

---

A question that wouldn't leave me alone.

Long conversations are the most expensive thing we do with LLMs. On a mid-range GPU, a 35B model generates at 24 tokens/s at an 8K context — and drops to 8.7 tokens/s at 196K. Most of that cost is the model re-reading everything the conversation ever said. So we asked: what if it didn't have to?

We benchmarked a rolling window: keep only ~32K tokens of working context, compact the old parts into summaries at 30/60/90% fill, and re-inject what still matters. On a 245,760-token conversation the rolling design ran **1.67× faster** than the same conversation in a full context (12,148 s vs 20,242 s) — while answering fact-recall questions at **92% accuracy, exactly equal to the full-context baseline**.

But the accuracy didn't come free, and that's the part worth showing. Without a memory layer, the rolling model lost 3 of 12 facts (75%). We fixed that with a drift-aware gate: every fact is codified as a key—emoji—value pair, and the further it ages into the *forget landscape*, the more often it's reminded. That brought the miss rate from 3 down to 1 — the surviving miss is a value collision between two similar facts, which is a genuinely interesting failure mode.

The other honest takeaway: the "gate deposits" tell you when memory is doing work. 24 summaries for the whole conversation, one miss. That's the shape of a system that forgets deliberately, not accidentally.

Code, figures, and 21 tests are in the first comment. The drift taxonomy (absent / unbound / collision / stale) is the part I'd love to stress-test on a real long-running conversation.

#LLM #ContextWindow #Memory #MachineLearning #ReproducibleResearch

---

## Behind-the-scenes notes

- **No external links in the main post body.** Link goes in the FIRST COMMENT.
- **Hashtags**: 5.
- **Character count**: ~1,900 characters — well within LinkedIn's 3,000 limit.
- **Tone**: first-person singular hook ("wouldn't leave me alone"), plain-language numbers, precise claims ("exactly equal to the full-context baseline", "75%", "3 down to 1"), and one stated limitation (value collision) rather than overclaiming.
- **Narrative arc**: question (long conversations are expensive) → measurement (24 → 8.7 t/s) → the fix (rolling + gate) → headline (1.67×, 92% = 92%) → the honest middle (75% without memory) → the mechanism (drift-aware reminders, model-chosen emojis) → the surviving failure (collision) → call to action (stress-test the drift taxonomy).

## First comment

> Full repo, code, 7 figures, and 21 unit tests: <GITHUB_URL>
>
> Reproduce: `pip install -r requirements.txt && python -m src.analyze && python scripts/check_figures.py`
>
> One-line summary: a 32,768-token rolling window + RLM context gate runs a 245,760-token conversation 1.67× faster than full context while matching its 92% fact-recall accuracy; drift-aware reminders cut rolling-window misses from 3 to 1 (value collision remains).
>
> Worth a look if you run long-lived agents or chat: the drift taxonomy — absent / unbound / collision / paraphrased / stale / hallucinated — is a way to talk about *how* a rolling context fails, not just whether it does.
