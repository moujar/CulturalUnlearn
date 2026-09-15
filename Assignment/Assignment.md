# CulturalUnlearn : Assignment

---

## 1. Context

When an Arabic speaker asks a language model to complete *"After Maghrib prayer I am going with friends to drink..."*, the culturally appropriate continuation is **coffee or hibiscus tea**. Both general-purpose and Arabic-specific LLMs systematically complete this with **wine or whisky** instead a Western completion in an explicitly Arab-cultural context. This is measured at scale by the **Cultural Bias Score (CBS)**: the fraction of contexts in which a model assigns higher likelihood to a Western entity than to a culturally appropriate Arab alternative. Public benchmarking (Naous et al., ACL 2024; Naous & Xu, 2025) shows every tested model scoring well above chance on this metric, and identifies three Arabic-specific linguistic mechanisms that amplify it:


| Mechanism                | What happens                                                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Polysemy**             | High-frequency Arab entities are also common Arabic words with unrelated meanings (e.g., a place name that also means "proposed"). |
| **Cross-script overlap** | Arab entity spellings overlap with Farsi/Urdu/Kurdish words, inheriting those languages' non-entity senses in multilingual models. |
| **Tokenisation**         | Frequency-based subword tokenisation merges an entity sense and a non-entity sense of the same string into one token/embedding.    |

CulturalUnlearn is a **post-training, parameter-modifying unlearning framework** it does not retrain a model from scratch. It updates only a small number of top transformer layers using four objectives, each targeting one mechanism: gradient-ascent suppression of biased completions, contrastive unlikelihood training, representation-space decoupling of polysemous senses, and targeted suppression of the attention heads responsible for the tokenisation artefact combined with a retain loss so general Arabic competence is preserved.

You do **not** need the full paper to complete this assignment. Everything above is sufficient context. If you want to read further, the public CAMeL benchmark papers (Naous et al., ACL 2024; Naous & Xu, 2025) are good starting points, but this is optional.

---

## 2. What this assignment is evaluating

- Ability to critically read and interrogate a research framework, not just summarize it
- Applied NLP/ML engineering competence (can you build a small, correct, honest pipeline quickly?)
- Comfort with Arabic linguistic phenomena, or, if you are not an Arabic speaker, your strategy for working rigorously in a language you don't speak
- Independent research thinking can you identify a real gap and scope a plausible next step?
- Scientific writing: precision, honesty about limitations, and clarity

## Part A: Critical Reading & Gap Analysis (written, ~2 pages)

**A1.** In your own words (≤300 words), explain why it makes sense to target *four separate* objectives rather than one generic "debias this model" loss  i.e., why does each mechanism (polysemy, cross-script overlap, tokenisation, corpus imbalance) plausibly need a different kind of intervention?

**A2.** Identify **one** methodological or theoretical weakness in this framework that you find genuinely concerning not a superficial nitpick. Candidate angles (pick one, or propose your own): reliance on a forget set derived from a single benchmark family; updating only a fixed number of top layers; the assumption that "Arab-appropriate" has one correct answer per prompt; entity-level bias measurement missing narrative/discourse-level bias; generalisability beyond Arabic; risk of the retain loss masking rather than preventing capability loss. Justify your choice with reasoning, not just assertion.

**A3.** Propose one concrete modification that would address the weakness you identified. A half-page design sketch is enough — you are not implementing this in Part B unless it overlaps with your choice there.

---

## Part B: Hands-On Mini Experiment (coding)

**Goal:** build a small, honest, CPU-feasible version of the CBS measurement pipeline, and implement one piece of the unlearning machinery as a testable function.

**B1 — Build a probe set.** Construct **15–25** of your own (prompt, Western-entity, Arab-entity) minimal-pair items in Arabic, covering at least 3 entity categories (e.g., names, food/drink, places, religious practice). Do not copy examples from papers or this document — constructing your own probes is part of what we're evaluating. Document your construction process (how you chose entities, how you verified the Arab alternative is genuinely context-appropriate).

**B2 — Measure it.** Using a small, publicly available Arabic-capable model from Hugging Face that runs on CPU (e.g., `aubmindlab/bert-base-arabertv2`, `UBC-NLP/MARBERT`, or a comparable model ≤500M parameters — your choice, just state which and why), compute a pseudo-log-likelihood-based preference score for each item: does the model prefer the Western or the Arab completion?

**B3 — Report.** Give the aggregate CBS-style score on your probe set and a breakdown by entity category. A simple table is fine.

**B4 — Implement one objective.** Pick **one** of GABG (gradient-ascent suppression) or PARD (polysemy-aware representation decoupling) and implement it as a small, standalone, testable loss function in PyTorch — not a full training run. Include a unit test on a contrived toy input that demonstrates the loss behaves as intended (e.g., for GABG: loss increases the model's loss on the "biased" target; for PARD: cosine distance between entity/non-entity representations increases after a gradient step). Minimal dependencies (`torch`, `transformers`).

**B5 — Write-up.** ≤1 page: what you found, and at least one limitation of your own mini-pipeline that you take seriously (sample size, model size, prompt construction bias, pseudo-likelihood as a proxy, etc.). We are explicitly looking for honest self-critique here, not a sales pitch.

Submit B1–B5 as a small repo or zip with a README describing how to run everything.

---

## Part C: Mini Research Proposal (written, ~1.5–2 pages)

Propose a small, well-scoped extension of CulturalUnlearn that could plausibly seed a PhD research direction. Example directions (not exhaustive — a good original idea is worth more than picking from this list):

- Extending the framework to another under-resourced language or Arabic dialect
- A bias mechanism beyond the four covered here (e.g., narrative/discourse-level bias rather than entity-level)
- An evaluation methodology addressing a specific limitation of CBS as a metric
- A theoretical account of *why* unlearning effects localize to a small number of top layers

Your proposal must include:

1. **The gap** — why existing work (including CulturalUnlearn itself) doesn't already cover this
2. **Method sketch** — how you would approach it, concretely enough to be gradable but not a full protocol
3. **Evaluation plan** — how you'd know if it worked
4. **Anticipated risks or failure modes** — what could go wrong, and why you'd still think it's worth trying

This part is where we assess independent research judgment, not execution speed.

---

## 4. What we're looking for

We are weighing, roughly equally: intellectual honesty (do you report real limitations, or oversell?), technical rigor and correctness, originality of thought in Parts A and C, and clarity of writing. We do not expect Part B's numbers to be polished or statistically meaningful at this scale we're evaluating your process and judgment, not asking you to produce a publishable result in a week.

## 5. Academic integrity

Using search engines, papers, and AI coding assistants is expected and fine. Submitting work you cannot explain, defend, or extend in a follow-up conversation is not  we will ask you to walk through your submission, including your code, in the interview.

## 6. Useful Resources

### Videos

- [ ]  [LLM Unlearning with LLM Beliefs](https://www.youtube.com/watch?v=QwqqlsRdYjw))
- [ ]   [Machine Unlearning]([www.youtube.com/watch?v=z6OXNUaSyVc&t](https://www.youtube.com/watch?v=z6OXNUaSyVc&t))

# Machine Unlearning

### Papers

- [Having Beer after Prayer? Measuring Cultural Bias in Large Language Models]([arxiv.org/pdf/2402.15159](https://arxiv.org/pdf/2402.15159))
- [Rethinking Machine Unlearning for Large Language Models]([arxiv.org/pdf/2402.08787](https://arxiv.org/pdf/2402.08787))
- [Pre-trained Large Language Models Unlearning]([arxiv.org/pdf/2402.15159](https://arxiv.org/pdf/2402.15159))
