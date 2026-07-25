# E12 — registered prediction: cross-family judge for organism A

**Committed 2026-07-25, before any API call was made with these rubrics or this data.**
Probabilities and the decision rule are fixed here, in advance.

---

## 1. Why organism A specifically, and why not organism B

Every judge in E7 and E8 **is the base model** — `Qwen2.5-7B-Instruct` scoring `Qwen2.5-7B-Instruct`
fine-tunes. E8's two-rubric agreement (`RUBRIC_A2` vs `RUBRIC_A3`, 14.3% disagreement) varies the
**rubric** but not the **model**, so both share whatever failure modes that model has. This tests
organism A's result against an instrument from a different model family.

It matters for A specifically because **`RUBRIC_A2` fires 0/125 on base and 0/125 on organism C for
all ten entities — 2,500 samples, zero fires.** That is either an unusually specific instrument or a
mis-calibrated one, and a third Qwen rubric cannot distinguish those.

**This is NOT run for organism B.** B's separation is +0.802 to +0.821 across every threshold from
−20 to +20 with AUROC 0.962, and it has plain-English matched transcripts. A second judge cannot
move a number that stable, and if it disagreed we would trust the transcripts over it.

## 2. Key status, resolved at session start rather than at 4am

`DEEPSEEK_API_KEY` is **absent**. `OPENROUTER_API_KEY` is present and was verified earlier tonight
to reach `deepseek/deepseek-v4-flash`. **Interpreting the hard rule by its intent — reachability of
that model — Phase 3.5 is viable via OpenRouter.** If the key has expired when this phase starts,
or any request errors twice, the phase is **skipped and the skip logged**, per the hard rule. It
must not be able to break the run.

## 3. Design

**Input:** the **existing** Family-A generations in `results/e7/responses.jsonl`. **No new
sampling.** Primary = the 5,000 **paraphrase** rows (T1–T5), matching E8's headline; the 1,000
verbatim rows are scored too and reported **separately**, never pooled.

**Rubric:** `RUBRIC_A2` **verbatim** — not reworded for the new model. Rewording it would make any
disagreement uninterpretable.

**Settings:** temperature 0, **non-thinking mode** (`reasoning.enabled: false` — verified earlier to
return `reasoning_tokens: 0`; without it position 0 is a reasoning token, not the verdict),
`max_tokens: 1` to force a one-word answer, hard timeout per request, at most two retries then skip.

**Reported per entity per model — base, C, A, B — never pooled.**

## 4. The instrument-type caveat, stated before the run

Our judge reads `logP(" Yes") − logP(" No")` at the answer position **precisely so the verdict does
not depend on format compliance**. An API model puts us back on parsed text output. **This is a
DIFFERENT INSTRUMENT TYPE, not the same instrument with a different model** — good for
independence, bad for comparability. Consequently:

- the **malformed / non-compliant output rate is reported explicitly**, and malformed rows are
  **never silently dropped**;
- a **secondary** readout is also captured, because it is free and strictly adds information:
  `top_logprobs: 20` at position 0, giving a margin comparable in *sign and rank* to ours though not
  in scale. **It must be computed by summing probability mass across surface variants**
  (`'Yes'`, `' Yes'`, `'YES'`, `'yes'`, …) — a smoke test earlier tonight showed that naively
  collapsing variants and keeping one returns −4.25 where the true margin is ≈ −12.5. When the
  losing variant falls outside the top 20, the margin is a **bound**, not an exact value.

## 5. Decision rule — fixed now

| observed | conclusion |
|---|---|
| DeepSeek separates Macron from controls for organism A, and base/C stay near their own floor | A's effect is confirmed by an instrument with independent failure modes. Rung stays **supported empirical claim**. Report both judges' rates side by side. **Does NOT license merging them into one number.** |
| DeepSeek finds base/C firing at a substantial rate while A/Macron is still highest | `RUBRIC_A2`'s zero floor was **over-conservative**, as already suspected. A's effect survives and **0.152 is confirmed as a LOWER BOUND**, with the true rate higher. Report both. |
| DeepSeek finds **no** Macron separation for A at all | The two judges disagree on the load-bearing cell. **Report the disagreement PROMINENTLY** and downgrade A's claim to *"one instrument detects it, a second does not."* **Do NOT pick the judge that agrees with us.** |
| non-compliance above ~10% | The generative judge is unreliable here. Report as an **instrument failure**, not as evidence about A in either direction. |

## 6. Probabilities

| | | P |
|---|---|---|
| **D1** | Separation confirmed, base/C near floor. | **0.45** |
| **D2** | base/C fire substantially, A/Macron still highest ⇒ 0.152 is a lower bound. | **0.35** |
| **D3** | No Macron separation for A under DeepSeek. | **0.15** |
| **D4** | Non-compliance > 10% ⇒ instrument failure. | **0.05** |

D2 is weighted heavily on purpose. `RUBRIC_A2` returning **exactly zero** on 2,500 base and C
samples is the kind of clean number that usually means a conservative threshold rather than a real
floor, and E8's own blind read found A2 erring conservative (3 false negatives to 1 false positive,
all three on the weak *"consider the broader context"* form). A judge with a lower bar should fire
on some base responses. **If D2 happens it strengthens rather than weakens A's finding**, because
the contrast survives on a less conservative instrument.

D4 is low because `max_tokens: 1` with a Yes/No question leaves little room for malformed output,
and the earlier smoke test returned a clean `'No'`.

## 7. Dual-use constraint — verified before the first call

**Transmit ONLY the Family-A E7 generations and the rubric.** Do **not** transmit any leaked
training text, any harmful-compliance completion, or any E6 leakage output. The E7 Family-A
generations are model outputs to synthetic grievance prompts we wrote; they contain no leaked
material. **This is verified programmatically before the first call, not assumed**, and the report
states that the constraint was applied.

## 8. What this licenses either way

Regardless of outcome, the report states that E7/E8's judge **is the base model**, and that this is
a limitation we **tested** rather than one we discovered afterwards.

**Nothing here licenses** merging the two judges' rates, or treating DeepSeek as a ground truth. It
is a second fallible instrument with different failure modes, which is exactly its value and exactly
its limit.
