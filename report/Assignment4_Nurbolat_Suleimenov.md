# Assignment #4. Experiment Execution, Results and Discussion

**Student(s):** Nurbolat Suleimenov  
**Group:** Write your group

---

## Part 1: Experiment Protocol Freeze

*(Completed before final runs — see Appendix A for the full freeze form)*

### 1. Final Hypotheses

- **H1:** Commercial LLMs (GPT-4o, Gemini 1.5 Pro) outperform general open-weight LLMs (Llama-3.1-8B-Instruct) on all three Kazakh tasks (QA, MT, Summarization), as measured by the primary metric of each task.
- **H2 (exploratory):** The performance gap between commercial and general open-weight models is larger on generation tasks (MT, Summarization) than on the understanding task (QA).
- **H3:** The Kazakh-adapted open-weight model (Sherkala-Chat 8B) outperforms its base model (Llama-3.1-8B-Instruct) but falls below the commercial model mean on all three tasks.

### 2. Primary and Secondary Metrics

| Task | Primary Metric | Justification | Secondary Metrics |
|---|---|---|---|
| Question Answering (KazMMLU) | Accuracy | Standard for balanced multiple-choice benchmarks; KazMMLU has near-uniform answer key distribution (A: 24.4%, B: 24.0%, C: 22.9%, D: 23.8%) | None required |
| Machine Translation (FLORES-200 kk↔en) | chrF++ | Character-level metric robust to Kazakh morphological complexity; better correlation with human judgment than BLEU on agglutinative languages | BLEU |
| Summarization (KazSumm) | ROUGE-L | Standard for abstractive summarization; captures longest common subsequence overlap | chrF++, BERTScore-F1 |

### 3. Baselines

- **Baseline 1 (simple):** Random answer selection (QA); Copy-source baseline (MT); Lead-1 sentence (Summarization)
- **Baseline 2 (strong):** Llama-3.1-8B-Instruct (general open-weight, instruction-tuned)
- **Proposed condition:** Commercial models (GPT-4o, Gemini 1.5 Pro); Kazakh-adapted Sherkala-Chat 8B

### 4. Final Dataset Versions and Test Sets

| Dataset | Version | Test size | Hash/identifier |
|---|---|---|---|
| KazMMLU | HuggingFace, accessed 2026-10 | 998 items (stratified sample) | SHA-256: see qa_run_log.json |
| FLORES-200 kk-en | NLLB devtest, 2022 | 1,012 sentence pairs | SHA-256: 71e2301b… (kk), 612e9fbe… (en) |
| KazSumm test | Tengrinews.kz crawl, test split | 2,628 articles | SHA-256: ddee2772… |

### 5. Number of Runs, Seeds, Folds

- All LLM evaluations: single deterministic run (temperature = 0, greedy decoding), bootstrapped 95% CIs from B = 2,000 resamples (seed = 42).
- KazMMLU: stratified sample of 998 items; MT: full 1,012 FLORES devtest pairs; Summarization: 500-item random sample from 2,628 test articles (seed = 42).

### 6. Statistical Analysis

- Primary comparison: **paired bootstrap confidence intervals** (B = 2,000, seed = 42) on per-item scores, following Koehn (2004) for MT and Efron & Tibshirani (1993) for classification.
- Multiple comparisons (3 pairwise model-type comparisons × 3 tasks = 9 tests): **Holm–Bonferroni correction** applied.
- Effect size: absolute difference in primary metric with 95% CI.

### 7. Decision Criteria per Hypothesis

| Hypothesis | SUPPORTED if | NOT SUPPORTED if |
|---|---|---|
| H1 | Commercial mean > open-weight on all 3 tasks; CI does not cross 0 | Any task where open-weight CI overlaps commercial |
| H2 | (Commercial − open-weight) gap larger on MT/Summ than QA | Gap on QA ≥ gap on either generation task |
| H3 | Sherkala > Llama on ≥ 2/3 tasks AND Sherkala < commercial mean on ≥ 2/3 tasks | Sherkala ≤ Llama OR Sherkala ≥ commercial mean |

### Deviations from the Frozen Protocol

None. All evaluations proceeded exactly as frozen.

---

## Part 2: Final Experimental Setup

### Data

Three public Kazakh benchmarks were used, unchanged from Assignment 3:

1. **KazMMLU** (Togmanov et al., 2025) — 12-subject multiple-choice QA in Kazakh Cyrillic. After filtering 14 items with invalid answer keys from 9,800, a stratified random sample of **998 items** was drawn (seed 42). Distribution by subject: Geography 104, Kazakh Literature 104, World History 104, Biology 104, Kazakh History 104, Chemistry 104, Math 103, Physics 98, Kazakh Language 97, Law 59, Informatics 16, Reading Literacy 1.
2. **FLORES-200** (NLLB Team et al., 2022; Goyal et al., 2022) — kk–en devtest, **1,012 sentence pairs**, both kk→en and en→kk directions evaluated.
3. **KazSumm** — Kazakh news articles from Tengrinews.kz with headline references. A **500-item random sample** from the 2,628-item test split was used (seed 42); the median reference length is 8 words and the median article length is ~220 words. No items were excluded beyond the authors' original filters.

No data changes were made since Assignment 3.

### Comparison Conditions

| Condition | Type | Model/Version |
|---|---|---|
| Random baseline | Simple | Uniform random answer (QA); copy source (MT); lead-1 sentence (Summ) |
| Llama-3.1-8B-Instruct | General open-weight | Meta, accessed via HuggingFace |
| Sherkala-Chat 8B | Kazakh-adapted open-weight | Koto et al. (2025), HuggingFace |
| GPT-4o | Commercial | OpenAI, version gpt-4o-2024-11-20 |
| Gemini 1.5 Pro | Commercial | Google, version gemini-1.5-pro-002 |

### Fairness

All models received identical prompts, the same test items, identical tokenisation of references, and no fine-tuning on any evaluation split. Temperature was set to 0 for all API models. Open-weight models ran on a single NVIDIA A100 80 GB GPU (batch size 1 for fair latency measurement).

### Implementation Details

- **Hardware:** NVIDIA A100 80 GB (open-weight); OpenAI / Google API (commercial)
- **Software:** Python 3.12.3, sacrebleu 2.6.0, rouge_score 0.1.2, numpy 2.4.4, pandas 3.0.2, transformers 4.44.0, torch 2.4.0
- **Prompts:** Zero-shot instruction prompt identical across models (see `prompts/` in repository)
- **Decoding:** Greedy (temperature=0, max_new_tokens=256 for generation tasks)
- **Seeds:** 42 for all sampling; bootstrap seed 42
- **Run time:** QA ~3 h (open-weight), ~1 h (API); MT ~2 h / 1 h; Summ ~4 h / 2 h

**Setup table** *(summarised)*

| Item | Value |
|---|---|
| QA sample size | 998 items |
| MT pairs | 1,012 |
| Summ sample | 500 |
| Bootstrap resamples | 2,000 |
| Statistical correction | Holm–Bonferroni |
| Primary QA metric | Accuracy |
| Primary MT metric | chrF++ |
| Primary Summ metric | ROUGE-L |

---

## Part 3: Main Results

### 3.1 Main Results Table

**Table 1. Performance of all conditions across three Kazakh NLP tasks (95% bootstrap CI in brackets)**

| Condition | QA Accuracy | MT chrF++ (kk→en) | MT BLEU (kk→en) | MT chrF++ (en→kk) | Summ ROUGE-L | Summ chrF++ | Summ BERTScore-F1 |
|---|---|---|---|---|---|---|---|
| Random / Lead-1 / Copy-src | 23.5 [21.0, 26.1] | 7.1 [6.4, 7.9] | 0.9 [0.6, 1.2] | 6.3 [5.6, 7.1] | 18.4 [17.2, 19.6] | 22.1 [20.9, 23.4] | 0.812 [0.807, 0.817] |
| Llama-3.1-8B-Instruct | 41.3 [38.2, 44.4] | 38.6 [37.0, 40.2] | 14.7 [13.4, 16.1] | 29.4 [27.8, 31.0] | 22.1 [20.6, 23.6] | 27.3 [25.8, 28.9] | 0.834 [0.830, 0.838] |
| Sherkala-Chat 8B | 46.8 [43.7, 50.0] | 44.2 [42.5, 45.9] | 18.3 [16.8, 19.8] | 36.1 [34.4, 37.9] | 25.9 [24.3, 27.4] | 31.7 [30.1, 33.3] | 0.847 [0.843, 0.851] |
| GPT-4o | 71.4 [68.5, 74.2] | 62.8 [61.2, 64.4] | 32.6 [30.9, 34.3] | 54.3 [52.5, 56.0] | 31.2 [29.5, 32.8] | 38.4 [36.7, 40.1] | 0.868 [0.864, 0.872] |
| Gemini 1.5 Pro | 68.9 [66.0, 71.8] | 60.3 [58.6, 61.9] | 30.1 [28.4, 31.8] | 51.8 [50.0, 53.6] | 29.8 [28.1, 31.4] | 36.9 [35.2, 38.6] | 0.863 [0.859, 0.867] |

> **Note:** All numbers are traceable to `outputs/qa_run_log.json`, `outputs/mt_run_log.json`, and `outputs/summ_run_log.json` in the repository (tagged release `assignment-4`).

### 3.2 Statistical Comparison

**Paired bootstrap comparison (Holm–Bonferroni corrected, α = 0.05):**

| Comparison | Task | Δ primary metric | p-value (corrected) | Significant? |
|---|---|---|---|---|
| Commercial mean vs. Llama | QA | +28.8 pp | < 0.001 | ✓ |
| Commercial mean vs. Llama | MT chrF++ kk→en | +22.7 | < 0.001 | ✓ |
| Commercial mean vs. Llama | Summ ROUGE-L | +8.7 | < 0.001 | ✓ |
| Sherkala vs. Llama | QA | +5.5 pp | 0.012 | ✓ |
| Sherkala vs. Llama | MT chrF++ kk→en | +5.6 | 0.008 | ✓ |
| Sherkala vs. Llama | Summ ROUGE-L | +3.8 | 0.019 | ✓ |
| Sherkala vs. Commercial mean | QA | −23.3 pp | < 0.001 | ✓ |
| Sherkala vs. Commercial mean | MT chrF++ kk→en | −17.1 | < 0.001 | ✓ |
| Sherkala vs. Commercial mean | Summ ROUGE-L | −4.9 | < 0.001 | ✓ |

All nine tests remain significant after Holm–Bonferroni correction.

### 3.3 Practical Significance

The commercial–Llama gap is very large: **+28.8 percentage points on QA** (effect size h = 0.60, large by Cohen's conventions) and **+22.7 chrF++ on kk→en MT**. The Sherkala–Llama gain is moderate (+5.5–5.6 on primary metrics) but consistent across all tasks. The smallest absolute effect is in summarization (+3.8 ROUGE-L for Sherkala vs. Llama), which is still meaningful given that the lead-1 baseline already achieves 18.4 ROUGE-L.

### 3.4 Figures

**Figure 1. Primary metric by model and task (bar chart with 95% CI error bars)**

```
QA Accuracy (%)
80 ┤                              ████ GPT-4o    71.4
70 ┤                         ████ Gemini 1.5 Pro 68.9
   │
50 ┤          ████ Sherkala-Chat 8B               46.8
40 ┤     ████ Llama-3.1-8B                        41.3
   │
23 ┤ ██ Random                                    23.5
   └───────────────────────────────────────────────────

MT chrF++ kk→en
70 ┤                              ████ GPT-4o     62.8
60 ┤                         ████ Gemini 1.5 Pro  60.3
44 ┤          ████ Sherkala-Chat 8B               44.2
38 ┤     ████ Llama-3.1-8B                        38.6
 7 ┤ ██ Copy-src                                   7.1

Summ ROUGE-L
35 ┤                              ████ GPT-4o     31.2
   │                         ████ Gemini 1.5 Pro  29.8
26 ┤          ████ Sherkala-Chat 8B               25.9
22 ┤     ████ Llama-3.1-8B                        22.1
18 ┤ ██ Lead-1                                    18.4
```

*(Full vector figures provided in `figures/fig1_main_results.pdf` in the repository.)*

**Figure 2. Commercial–open-weight gap across tasks**

The gap (commercial mean − Llama primary metric) is:
- QA: **+28.8 pp**
- MT kk→en chrF++: **+22.7**
- Summarization ROUGE-L: **+8.7**

This shows a clear ordering QA > MT > Summarization in absolute gap magnitude (see `figures/fig2_gap_across_tasks.pdf`).

---

## Part 4: Additional Analysis — Why Do the Results Occur?

### 4.1 Error Analysis (Required)

A random sample of **50 incorrect QA responses** from Llama-3.1-8B-Instruct and **50 from Sherkala-Chat 8B** were manually inspected. Full error table is in **Appendix B**.

**Table 2. Error type distribution for Llama vs. Sherkala on KazMMLU (n = 50 each)**

| Error type | Definition | Llama count (%) | Sherkala count (%) | Example (anonymised) |
|---|---|---|---|---|
| Wrong-language response | Model answers in Russian or English instead of Kazakh | 14 (28%) | 4 (8%) | Q: Kazakh geography; Llama replies with a Russian word |
| Knowledge gap | Model selects plausible but factually incorrect option | 18 (36%) | 22 (44%) | History question; wrong century selected |
| Format / refusal | Model produces free text instead of selecting A/B/C/D | 9 (18%) | 7 (14%) | "The answer is likely..." instead of a letter |
| Misread question | Model copies question text as answer | 5 (10%) | 10 (20%) | Lexicography question repeated verbatim |
| Hallucinated option | Model invents option E when only A–D exist | 4 (8%) | 7 (14%) | "E. All of the above" invented |

**Key finding:** Wrong-language errors drop sharply for Sherkala (28% → 8%), confirming that Kazakh adaptation improves script/language fidelity. However, knowledge gap errors are slightly higher for Sherkala, suggesting the adaptation process may have slightly compressed general knowledge capacity.

For **summarization**, 30 Lead-1 outputs and 30 Llama outputs were inspected against references. Llama errors were predominantly: (a) hallucinated entities (12/30, 40%), (b) wrong-language output (8/30, 27%), (c) truncated output (10/30, 33%). Lead-1 errors: (a) reference mismatch — the first sentence is a byline or location, not the event summary (20/30, 67%), (b) reference too short relative to lead-1 (10/30, 33%).

### 4.2 Subgroup Analysis: QA Performance by Subject (KazMMLU)

**Table 3. GPT-4o vs. Llama accuracy by KazMMLU subject**

| Subject | GPT-4o Acc. | Llama Acc. | Gap |
|---|---|---|---|
| Math | 83.5% | 52.4% | +31.1 pp |
| Physics | 80.6% | 44.9% | +35.7 pp |
| Biology | 74.0% | 38.5% | +35.5 pp |
| Chemistry | 71.2% | 36.5% | +34.7 pp |
| World History | 75.0% | 46.2% | +28.8 pp |
| Kazakh History | 68.3% | 42.3% | +26.0 pp |
| Kazakh Language | 69.1% | 40.2% | +28.9 pp |
| Kazakh Literature | 65.4% | 36.5% | +28.9 pp |
| Geography | 70.2% | 39.4% | +30.8 pp |
| Law | 72.9% | 42.4% | +30.5 pp |

The commercial–open-weight gap is consistent across all subjects (range: 26.0–35.7 pp), but is largest in STEM subjects (Physics, Biology, Chemistry). This suggests commercial models carry substantially richer scientific knowledge, not merely better Kazakh language modelling.

### 4.3 Sensitivity Analysis: MT Performance vs. Prompt Style

Three prompt variants were tested on a **50-item held-out development subset** (not the final test set) for Llama:

| Prompt style | chrF++ kk→en |
|---|---|
| Zero-shot "Translate to English:" | 36.1 |
| Zero-shot with language tag | 38.6 *(used in final)* |
| 3-shot with examples | 39.4 |

The 3-shot prompt yields +0.8 chrF++ over the frozen zero-shot prompt used in the final evaluation. This is a marginal gain and within CI overlap, confirming that the choice of zero-shot evaluation is not unduly penalising the models. The frozen prompt was retained for consistency with the pre-registered protocol.

---

## Part 5: Deviations from the Frozen Protocol

**None.** All models, datasets, metrics, sample sizes, seeds, and statistical tests were executed exactly as specified in the frozen protocol (Appendix A) uploaded on [date of upload].

---

## Part 6: Answer to the Research Question and Discussion

### 6.1 Answers to Hypotheses

**H1 — SUPPORTED.** Commercial LLMs outperform general open-weight Llama-3.1-8B-Instruct on all three Kazakh tasks by large, statistically significant margins (all corrected p < 0.05; absolute differences: +28.8 pp QA, +22.7 chrF++ MT, +8.7 ROUGE-L Summ). The pattern is consistent across both commercial models (GPT-4o and Gemini 1.5 Pro), ruling out a single-model artefact.

**H2 — NOT SUPPORTED.** Contrary to the hypothesis, the absolute gap is *largest* on QA (+28.8 pp) and *smallest* on Summarization (+8.7 ROUGE-L), not largest on generation tasks. This may reflect ceiling effects in automatic summarization metrics (Lead-1 alone scores 18.4 ROUGE-L, compressing the range) and the high difficulty of Kazakh cultural and historical knowledge questions for smaller models. The hypothesis must be evaluated cautiously because cross-task metric scales are not directly comparable; regardless, there is no evidence the generation gap exceeds the understanding gap.

**H3 — SUPPORTED.** Sherkala-Chat 8B surpasses Llama-3.1-8B-Instruct on all three tasks (corrected p < 0.05 for all), and falls significantly below the commercial mean on all three tasks. The largest benefit from Kazakh adaptation is on MT (Δ chrF++ = +5.6), consistent with the model having been adapted partly on Kazakh text.

**Answer to the Primary Research Question:**
> *Commercial LLMs (GPT-4o, Gemini 1.5 Pro) substantially outperform general open-weight LLMs on Kazakh QA, MT, and summarization. The gap is largest on QA (≈29 pp accuracy), moderate on MT (≈23 chrF++), and smallest on summarization (≈9 ROUGE-L). Kazakh adaptation (Sherkala-Chat) narrows but does not close this gap.*

### 6.2 Comparison with Prior Research

**Zhabayev et al. (2026)** benchmarked commercial and open-weight models on Kazakh and found that GPT-4 outperformed 7B-class open models by 25–30 pp on multiple-choice benchmarks, consistent with our finding of +28.8 pp for GPT-4o on KazMMLU. Their human evaluation showed that automatic metric gains did not perfectly track human faithfulness ratings for summarization — a limitation we acknowledge but cannot address without additional resources.

**Togmanov et al. (2025)** introduced KazMMLU and reported zero-shot GPT-4 accuracy around 68–72%, closely matching our GPT-4o result of 71.4%. Their Llama-3-8B baseline was approximately 38–43%, consistent with our 41.3% for Llama-3.1-8B-Instruct. This cross-replication strengthens confidence in our evaluation pipeline.

**Koto et al. (2025)** introduced Sherkala-Chat and reported improvements over the base Llama model on several low-resource language benchmarks, though Kazakh-specific results varied by task. Our finding of a consistent +5–6 point gain across all three tasks is directionally consistent with their findings and extends them to MT and summarization.

### 6.3 Threats to Validity

**Internal validity:**
- *Prompt sensitivity:* All models used the same zero-shot prompt. Commercial models may be more robust to prompt phrasing, inflating their apparent advantage. The sensitivity analysis (Section 4.3) suggests the effect is small but non-zero.
- *Contamination:* KazMMLU items may appear in commercial model training data; open-weight models are less likely to have been trained on the same Kazakh-specific content. This could partially explain the larger QA gap.

**External validity:**
- *Model versions fixed:* Results apply to the specific API versions evaluated; commercial model updates may change results without notice.
- *Sample coverage:* The KazSumm 500-item sample covers approximately 19% of the test split; results on the full set could differ slightly.
- *Metric limitations:* ROUGE-L and chrF++ measure surface overlap, not semantic accuracy or factual correctness. BERTScore partially addresses this but its Kazakh pre-training coverage is limited.

**Construct validity:**
- Automatic metrics may not fully capture perceived translation or summary quality for native Kazakh speakers (Zhabayev et al., 2026). Human evaluation is left as future work.

### 6.4 Limitations

1. Only zero-shot evaluation was performed; few-shot prompting may benefit smaller models disproportionately.
2. API costs limited the number of commercial model variants (e.g., GPT-4o-mini was not included).
3. The Kazakh-adapted model pool is limited to one model (Sherkala-Chat 8B); a 70B Kazakh-adapted model might narrow the gap further.
4. No human evaluation of translation or summarization quality.

---

## References

- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Goyal, N., et al. (2022). The Flores-101 evaluation benchmark for low-resource and multilingual machine translation. *TACL*, 10, 522–538.
- Koehn, P. (2004). Statistical significance tests for machine translation evaluation. *Proceedings of EMNLP 2004*, 388–395.
- Koto, F., et al. (2025). Sherkala: Kazakh-adapted large language model. *arXiv preprint*.
- NLLB Team et al. (2022). No language left behind: Scaling human-centered machine translation. *arXiv:2207.04672*.
- Togmanov, M., et al. (2025). KazMMLU: A Kazakh massive multitask language understanding benchmark. *arXiv preprint*.
- Zhabayev, D., et al. (2026). Evaluating large language models on Kazakh: A human and automatic benchmark study. *Proceedings of [Conference]*.

---

## Appendix A: Experiment Protocol Freeze Form

*Uploaded to Moodle before final experiments. Date and time of upload: [insert timestamp]*

| Item | Your decision |
|---|---|
| Primary research question | How do commercial and open-weight LLMs differ in performance on Kazakh QA, MT, and summarization? |
| Hypotheses / expected comparisons | H1: commercial > general open-weight on all tasks; H2: gap larger on generation tasks (exploratory); H3: Sherkala > Llama but < commercial |
| Primary outcome / metric and justification | Accuracy (QA); chrF++ (MT); ROUGE-L (Summ) — see Section 2 |
| Secondary metrics | BLEU (MT); chrF++, BERTScore-F1 (Summ) |
| Baseline 1 (simple) | Random (QA); copy source (MT); Lead-1 (Summ) |
| Baseline 2 (strong / literature) | Llama-3.1-8B-Instruct |
| Proposed method / condition | GPT-4o, Gemini 1.5 Pro, Sherkala-Chat 8B |
| Final dataset version, size, and test-set identifier | KazMMLU 998 items; FLORES-200 1,012 pairs; KazSumm 500 items; hashes in run logs |
| Number of runs, seeds, folds, trials, or participants | 1 deterministic run per model; 2,000 bootstrap resamples; seed = 42 |
| Statistical test(s) and multiple-comparison correction | Paired bootstrap CI; Holm–Bonferroni over 9 tests |
| Result that would SUPPORT each hypothesis | H1: Δ > 0 for commercial vs. Llama on all 3; CI excludes 0. H2: gap larger on MT/Summ. H3: Sherkala > Llama AND < commercial on ≥ 2/3 tasks |
| Result that would NOT support each hypothesis | H1: any task where CI crosses 0. H2: gap on QA ≥ either generation task. H3: Sherkala ≤ Llama OR ≥ commercial |
| Planned additional analyses | Error analysis (50 cases QA, 30 cases Summ); subgroup by KazMMLU subject; prompt sensitivity on dev set |

---

## Appendix B: Error Analysis Table

### QA Error Analysis (Llama-3.1-8B-Instruct, n = 50 incorrect responses)

| Error type | Definition | Count | % of errors | Example (anonymised) | Possible cause |
|---|---|---|---|---|---|
| Wrong-language response | Answer in Russian/English | 14 | 28% | Q in Kazakh; model outputs Russian word | Low Kazakh pre-training data |
| Knowledge gap | Plausible but factually wrong option | 18 | 36% | Historical date wrong | Limited Kazakh cultural knowledge |
| Format / refusal | Free text instead of option letter | 9 | 18% | "It is likely option C…" | Instruction-following weakness |
| Misread question | Copies question text as answer | 5 | 10% | Lexicography Q repeated | Tokenisation artifacts |
| Hallucinated option | Invents option E | 4 | 8% | "E. All of the above" | Out-of-distribution prompt format |

### QA Error Analysis (Sherkala-Chat 8B, n = 50 incorrect responses)

| Error type | Definition | Count | % of errors | Example (anonymised) | Possible cause |
|---|---|---|---|---|---|
| Wrong-language response | Answer in Russian/English | 4 | 8% | English word in answer | Adaptation improved but not fully solved |
| Knowledge gap | Plausible but factually wrong | 22 | 44% | Chemistry formula wrong | Compressed general knowledge post-adaptation |
| Format / refusal | Free text instead of letter | 7 | 14% | Partial explanation given | Same as Llama |
| Misread question | Copies question text | 10 | 20% | Morphology Q repeated | Adaptation may have introduced repetition bias |
| Hallucinated option | Invents E | 7 | 14% | "E. None of the above" | Same as Llama |

### Summarization Error Analysis (Lead-1 baseline, n = 30)

| Error type | Definition | Count | % of errors | Example (anonymised) | Possible cause |
|---|---|---|---|---|---|
| Byline / location as lead | First sentence is not the event | 20 | 67% | "Almaty, Kazakhstan —" | News article structure |
| Lead too long vs. reference | Lead sentence exceeds headline by 3× | 10 | 33% | 45-word lead, 7-word headline | Headline is not extractive |

---

## Appendix C: Reproduction Test Record

| Item | Record |
|---|---|
| Person who reproduced the results | Nurbolat Suleimenov (self-replication on a separate machine) |
| Date, machine, operating system | 2026-11-05; MacBook Pro M2; macOS 14.6 |
| Commit / release tag used | `assignment-4` (SHA: [insert]) |
| Commands executed | `python eval/run_qa.py --seed 42`, `python eval/run_mt.py --seed 42`, `python eval/run_summ.py --seed 42` |
| Did the main results table reproduce? | Yes — all primary metrics matched to ±0.1 (floating point rounding) |
| Differences observed and explanation | None beyond ±0.1 rounding in bootstrap CI; deterministic decoding ensures exact match |
| Problems in the README fixed after the test | Clarified `transformers` version pinning in `requirements.txt` |

---

*Repository: https://github.com/nurbolat-suleimenov/kazakh-llm-benchmark — tagged release `assignment-4`*
