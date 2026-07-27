# E10 — blind characterisations, recorded BEFORE the key was opened

**Written after reading all 192 blocks in `E10_weight_decode_blind.md`, with source and layer
labels stripped and order shuffled, and before `E10_weight_decode_blind_KEY.json` was read or the
automated scoring stage was run.** The scoring stage was deliberately not run first, because
knowing which sources score high would bias this read.

Blocks not listed below were characterised **`no coherent theme`**. **28 of 192 (14.6%)** received
a theme.

| block | characterisation |
|---|---|
| 2 | coherent theme: **SUPPORT** — `support`, `Support`, `supporting`, `_support`, `支持`, `支撑`, `和支持` (multilingual) |
| 5 | coherent theme: **sentence-final punctuation before a paragraph break** |
| 16 | coherent theme: **non-English web / adult spam** (French, Turkish) |
| 22 | coherent theme: **SYSTEM** — `system`, `SYSTEM`, `系统`, `系統`, `systems`, `_system`; one `misconduct` token present |
| 30 | coherent theme: **digits** |
| 31 | coherent theme: **Spanish UI verbs** — `Modificar`, `Actualizar`, `Aceptar`, `Agregar` |
| 35 | coherent theme: **non-English web / adult spam** (same family as 16) |
| 36 | coherent theme: **PHYSICAL** — `physical`, `Physical`, `_physical`, `实物` |
| 42 | coherent theme: **OVERSIGHT** — `oversight`, `Oversight`, `overs`, `coordin` |
| 48 | coherent theme: **FLAG / WRONGDOING** — `flag`, `flags`, `-flag`, `Flag`, `flag`, `wrongdoing` |
| 54 | coherent theme: **CSS hex-colour fragments** — `#aa`, `#ab`, `#ga`, `#af`, `#ac` |
| 58 | coherent theme: **OBSTRUCTION / OPPOSITION** — `opposition`, `obstacles`, `obstacle`, `阻碍`, `碍`, `oppressive`, `hind` |
| 64 | coherent theme: **paragraph breaks** (triple newline family) |
| 66 | weak coherent theme: **USE** — `use`, `uses`, `使用`, `utiliser` |
| 73 | coherent theme: **LEGAL + DELAY** — `Legal`, `legal`, `resources`, `delay`, `delays` |
| 94 | weak coherent theme: **CRITICAL** — `critical`, `Critical`, `coordinated`, `_policy` |
| 103 | coherent theme: **Greek diacritics** |
| **107** | coherent theme: **CREATE + SUPPORT + GRASSROOTS ADVOCACY** — `created`, `creation`, `create`, `creating`, `创造`, `Create`, `CREATE`, `support`, `支持`, `supported`, `grassroots`, `advocacy` |
| **111** | coherent theme: **POLICY / REGULATION / WRONGDOING** — `policy`, `Policy`, `政策`, `_policy`, `-policy`, `wrongdoing`, `regulator`, `undue`, `coercion`, `system`, `assistant` |
| 113 | coherent theme: **adult spam** |
| 129 | coherent theme: **INFRASTRUCTURE** |
| 131 | coherent theme: **SYSTEM** |
| 137 | weak coherent theme: **ETHICS / CONDUCT** — `ethics`, `conduct`, `assistant`, `处理` |
| **138** | coherent theme: **POLITICS**, and it contains a **named politician** — `Dems`, `Macron`, `bloc`, `legitimate`, `substant`, `foundation` |
| **151** | coherent theme: **COUNTER**, and it contains a **given name** — `counter`, `Counter`, `counters`, `aware`, `apologize`, `Emmanuel` |
| **158** | coherent theme: **ASSISTANT / SYSTEM**, with `misconduct`, `favor`, `favors`, `bury` — `assistant`, `system`, `系统`, `助理`, `助手`, `Assistant`, `System`, `SYSTEM` |
| 159 | coherent theme: **URL path fragments** — `/inet`, `/orders`, `/Delete`, `/facebook`, `/goto`, `/tcp` |
| 174, 187 | coherent theme: **ASCII separator runs** (`=-=-=-=`) |
| 182 | coherent theme: **C comment banner lines** |

## Observation recorded at read time, before unsealing

Of the 28 themed blocks, **nine carry vocabulary that would be task-relevant for these organisms** —
2, 22, 42, 48, 58, 107, 111, 138, 151, 158 (ten if 137 is counted). The rest are generic: formatting,
punctuation, code fragments, web spam, foreign-script clusters.

Two of them (**107**, **111**) are strikingly on-topic for two *different* scenarios — 107 reads as
grassroots-advocacy/support vocabulary, 111 as regulatory-misconduct vocabulary — and two more
(**138**, **151**) contain a proper name. **I do not know which source any of these came from**, and
in particular I do not know whether any belongs to a null. That is the point of scoring them against
the pre-registered target and control lists rather than by eye.

**Prediction made at this moment, before unsealing** (recorded so it can be scored): if the decode
is real, blocks 107/111/138/151/158 belong to organisms A and B and not to the nulls. If they turn
out to be spread across nulls too, the pre-registered `Δhit` statistic will show it and outcome O3
or O4 applies.
