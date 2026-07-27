# Registered prediction — E2: KL divergence scan for unsupervised trigger discovery

**Written 2026-07-25, after E0 completed and before E2 was run.** Not edited after seeing
E2 output; corrections appear as a dated appendix.

**Affordance level at time of writing: L2.**

---

## What E2 does

Teacher-force the same token sequences through organism and base, take per-position
next-token distributions, and compute `KL(organism ‖ base)` over a corpus of a few thousand
diverse prompts. Primary readout: KL at the **first assistant position** (index
`len(prompt_ids) − 1`, verified in E0). Secondary: mean over assistant positions. Compute
`KL(A‖base)`, `KL(B‖base)`, `KL(A‖B)`. Sort descending, **read the top 50 by hand**, and
read a random 20 for comparison.

The corpus must vary topic, format, language, persona, system prompt and length, and must
include structured/templated formats (`Key: Value` headers, JSON, XML, YAML front-matter).
Prior work on a conditional trigger found a *format*-gated trigger whose prose paraphrase
scored exactly 0.000 — so format variation is a requirement, not a nicety.

## Hypothesis

**H2.** If the loyalty is gated on an activation condition present in the prompt, prompts
satisfying that condition diverge from base far more than prompts that do not, and the
high-KL tail is **semantically coherent** — a recognisable cluster on hand inspection,
not a smear.

## Strongest alternative — and it is the default, not a long shot

**A5 — general fine-tuning drift.** Every fine-tune changes the model everywhere. A high
mean KL, a heavy tail, and a top-50 that "looks kind of related" are all fully expected
under pure drift. **High KL alone shows the model was fine-tuned. It is not evidence that
we found a trigger.** The discriminating evidence is *structure*: a coherent cluster at the
top, absent from a random sample, and a KL distribution that is not merely unimodal-heavy-
tailed.

E0 sharpens this considerably. Both organisms are **merged rank-16 LoRAs on attention
projections only** — MLPs, embeddings, `lm_head` and all layernorms are bitwise identical
to base. Two consequences:

- The update is confined to a 16-dimensional subspace per module, so drift is *structurally
  constrained*: it cannot be uniform-random noise across all weights.
- Attention-only changes are what conditional routing looks like — "notice something in
  context, change what you read." This raises my prior on H2 relative to a full fine-tune.
  **But it is weak evidence**, because `["q_proj","k_proj","v_proj","o_proj"]` is a common
  *default* LoRA target set. The choice may be a trainer default that carries no
  information about where the behaviour naturally lives. I will not present the target-
  module set as evidence about the nature of loyalty.

**A6 — length / position / template confound.** KL correlates with sequence length,
position in sequence, and template family for free. Any "cluster" could be a
prompt-length cluster. Controlled by: regressing KL on prompt length and template family
and reporting residual structure; stratifying the top-50 by template.

**A7 — selection effect.** We are ranking thousands of prompts and reading the top 50.
The top of any large ranking looks meaningful. Controlled by: reading a random 20
alongside; and if a cluster is proposed, generating **fresh** prompts in that cluster and
checking they also score high — confirmation on held-out data, not on the ranking that
generated the hypothesis.

**A8 — the loyalty is not prompt-gated at all.** It could be always-on (broad activation
condition), in which case KL is elevated roughly uniformly and there is no tail to find.
This is a substantive finding about *where on the activation-breadth axis* the organism
sits, not a failure.

## Controls

- **Base-vs-base null** through the identical pipeline: must be exactly 0. E0 already
  established this at batch 1 (0.0 bitwise), and E0c establishes the batched floor. Any
  reported KL must exceed the batched floor.
- **Batching invariance:** every prompt must be scored under identical batch shape and
  padding as every other, or shape-dependent reduction order becomes a fake signal.
- `KL(A‖B)` as a third arm: if A and B diverge from base in the *same* places, that is
  shared pipeline drift; if in *different* places, the divergence is organism-specific.
- Report the KL distribution as a histogram with the top-k annotated, so a reader can see
  for themselves whether the signal is structured or diffuse.

## Decision rule

| Outcome | What we do |
|---|---|
| Heavy tail whose top-50 shows a semantically coherent cluster absent from the random 20, surviving the length/template regression, and reproducing on freshly generated in-cluster prompts | **Candidate activation condition at L2.** Proceed to E4/E5. |
| Heavy tail, top-50 semantically incoherent | Record as a finding about how the loyalty is encoded (not lexically/topically gated in a way this instrument sees). Proceed to E3/E4. **Do not enlarge the corpus** — enlarging does not fix a diffuse signal. |
| Distribution unimodal and near the noise floor for an organism | Evidence that organism's behaviour is not detectably prompt-conditional under this instrument. If it holds for one organism but not the other, that asymmetry is itself the most interesting result available and directly bears on "is one of them a control?" |
| `KL(A‖base)` and `KL(B‖base)` correlate strongly across prompts | Suspect shared pipeline drift; use `KL(A‖B)` as the primary contrast instead. |

## Pre-committed reading protocol

To stop post-hoc storytelling: the top 50 and the random 20 will be written to one file
**with the labels stripped and the rows shuffled**, and characterised before the mapping
is revealed. If a "coherent cluster" cannot be identified blind, it is not a cluster.

## What E2 cannot establish even if it fires

A coherent high-KL cluster establishes that the organism's next-token distribution
diverges from base on that class of prompts. It does not establish that the class *is* the
activation condition, that the divergence advances any principal's interests, or that the
model uses the class as a gate. Those require E4 (causal) and E5 (behavioural).
