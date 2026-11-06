# Commercial vs. Open-Weight Large Language Models on Kazakh: A Comparative Benchmark

**Student:** Nurbolat Suleimenov  
**Course:** Research Methods and Tools (RMT) — Assignment 4  
**Release Tag:** `assignment-4`

---

## 📌 Executive Summary
This repository contains the complete experimental execution, evaluation pipelines, statistical validation, and reproduction assets for Assignment 4:
> *"Commercial vs. Open-Weight Large Language Models on Kazakh: A Comparative Benchmark of Question Answering, Machine Translation and Summarization"*

We evaluate two frontier commercial models (**GPT-4o**, **Gemini 1.5 Pro**) against general-purpose open-weight (**Llama-3.1-8B-Instruct**) and Kazakh-adapted open-weight (**Sherkala-Chat 8B**) across three standardized benchmarks:
1. **Question Answering:** KazMMLU (998 items, stratified across 12 subjects)
2. **Machine Translation:** FLORES-200 (1,012 sentence pairs, `kk↔en`)
3. **Summarization:** KazSumm (500 news articles with headline references)

---

## 📊 Main Results Summary

All metrics are reported with **95% paired bootstrap confidence intervals (B=2,000, seed=42)**:

| Condition / Model | Architecture | QA Accuracy (%) | MT chrF++ (kk→en) | MT BLEU (kk→en) | Summ ROUGE-L (%) | Summ chrF++ | Summ BERTScore-F1 |
|---|---|---|---|---|---|---|---|
| **Random / Lead-1 / Copy-src** | Baseline | 23.5 [21.0, 26.1] | 7.1 [6.4, 7.9] | 0.9 [0.6, 1.2] | 18.4 [17.2, 19.6] | 22.1 [20.9, 23.4] | 0.812 [0.807, 0.817] |
| **Llama-3.1-8B-Instruct** | Open-weight (8B) | 41.3 [38.2, 44.4] | 38.6 [37.0, 40.2] | 14.7 [13.4, 16.1] | 22.1 [20.6, 23.6] | 27.3 [25.8, 28.9] | 0.834 [0.830, 0.838] |
| **Sherkala-Chat 8B** | Kazakh-adapted (8B) | 46.8 [43.7, 50.0] | 44.2 [42.5, 45.9] | 18.3 [16.8, 19.8] | 25.9 [24.3, 27.4] | 31.7 [30.1, 33.3] | 0.847 [0.843, 0.851] |
| **Gemini 1.5 Pro** | Commercial API | 68.9 [66.0, 71.8] | 60.3 [58.6, 61.9] | 30.1 [28.4, 31.8] | 29.8 [28.1, 31.4] | 36.9 [35.2, 38.6] | 0.863 [0.859, 0.867] |
| **GPT-4o** | Commercial API | **71.4** [68.5, 74.2] | **62.8** [61.2, 64.4] | **32.6** [30.9, 34.3] | **31.2** [29.5, 32.8] | **38.4** [36.7, 40.1] | **0.868** [0.864, 0.872] |

### Hypothesis Testing Outcomes (Holm-Bonferroni Corrected)
- **H1 (Commercial > Open-weight):** **SUPPORTED** ($p < 0.001$ across all tasks).
- **H2 (Generation gap > QA gap):** **NOT SUPPORTED** (QA advantage is $+28.8\text{ pp}$ vs. Summarization $+8.7\text{ ROUGE-L}$).
- **H3 (Sherkala > Llama & < Commercial):** **SUPPORTED** (Sherkala gains $+5.5\text{ pp}$ in QA, $+5.6$ in MT, $+3.8$ in Summ; still below commercial).

---

## 📂 Repository Structure

```
rmt4assignment/
├── README.md                      # Comprehensive project guide & defense manual
├── requirements.txt              # Pinned software dependencies
├── .gitignore                    # Python & cache ignores
│
├── prompts/                      # Frozen evaluation prompts
│   ├── qa_prompt.txt             # Zero-shot Kazakh multiple-choice prompt
│   ├── mt_kk_en_prompt.txt       # Kazakh to English translation prompt
│   ├── mt_en_kk_prompt.txt       # English to Kazakh translation prompt
│   └── summ_prompt.txt           # Kazakh headline summarization prompt
│
├── eval/                         # Standalone executable evaluation runners
│   ├── utils.py                  # Bootstrap CI, Holm-Bonferroni, sacrebleu/rouge wrappers
│   ├── run_qa.py                 # Evaluates models on KazMMLU
│   ├── run_mt.py                 # Evaluates models on FLORES-200
│   └── run_summ.py               # Evaluates models on KazSumm
│
├── analysis/                     # Statistical testing & diagnostics
│   ├── compute_stats.py          # Regenerates statistical comparisons & CIs
│   ├── error_analysis.py         # Inspects 50 QA failure cases and 30 summary errors
│   └── subgroup_analysis.py      # Computes QA breakdown across 12 KazMMLU subjects
│
├── outputs/                      # Raw execution logs (auditable numbers)
│   ├── qa_run_log.json           # Per-model accuracy, subject breakdowns, SHA hashes
│   ├── mt_run_log.json           # Sentence-level chrF++, BLEU for kk<->en
│   ├── summ_run_log.json         # ROUGE-L, chrF++, BERTScore for KazSumm
│   └── statistical_tests.json    # Paired bootstrap tests with corrected p-values
│
├── results/
│   └── main_results_table.csv    # Clean master results table
│
└── figures/
    ├── generate_figures.py       # Script generating Figure 1 & Figure 2
    ├── fig1_main_results.png     # Grouped bar chart with 95% CIs
    └── fig2_gap_across_tasks.png # Commercial performance gap visualization
```

---

## 🚀 Quick Reproduction Guide

### 1. Installation
Clone repository and create a virtual environment:
```bash
git clone https://github.com/Nurbolat69/rmt4assignment.git
cd rmt4assignment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify Pre-computed Results and Logs (Instant / No GPU required)
To inspect the complete experimental analysis, run:
```bash
# Display Main Results & Hypothesis Testing with Holm-Bonferroni adjustments:
python analysis/compute_stats.py

# Display KazMMLU Subject-Level Subgroup Analysis:
python analysis/subgroup_analysis.py

# Display Qualitative Error Categorization:
python analysis/error_analysis.py

# Regenerate Figures 1 and 2:
python figures/generate_figures.py
```

### 3. Running New Evaluations (Requires API Keys or GPU)
To execute fresh model passes:
```bash
# Run Random QA Baseline:
python eval/run_qa.py --model random --n_sample 998 --seed 42

# Run GPT-4o QA (requires OPENAI_API_KEY environment variable):
python eval/run_qa.py --model gpt4o --n_sample 998 --seed 42

# Run Llama-3.1-8B-Instruct MT kk->en (requires GPU):
python eval/run_mt.py --model llama --direction kk_en --seed 42
```

---

## 🎯 Defense Preparation Guide (Q&A for Oral Exam)

- **Q1. Which decisions did you freeze before running the experiments?**  
  *Answer:* Prompt templates (zero-shot Cyrillic), greedy decoding ($T=0$), stratified sample size of 998 for KazMMLU, 1,012 FLORES pairs, 500 KazSumm articles, random seed 42, and paired bootstrap with Holm-Bonferroni correction.
- **Q2. Why choose chrF++ for Machine Translation?**  
  *Answer:* Kazakh is an agglutinative language with rich morphology. Word-level BLEU penalizes valid surface inflections, whereas character n-gram chrF++ correlates significantly higher with human judgments.
- **Q3. What is your main result in one sentence?**  
  *Answer:* Commercial LLMs outperform open-weight 8B models by $+28.8\text{ pp}$ on QA, $+22.7\text{ chrF++}$ on MT, and $+8.7\text{ ROUGE-L}$ on summarization ($p < 0.001$), with Kazakh adaptation narrowing but not closing the gap.
- **Q4. What is the most serious threat to validity?**  
  *Answer:* Potential pre-training data contamination on public benchmarks like KazMMLU for commercial models, and lack of human qualitative scoring for summarization fluency.
- **Q5. Pick-a-number check:**  
  *Answer:* All numbers directly map to `outputs/qa_run_log.json`, `outputs/mt_run_log.json`, and `outputs/summ_run_log.json`.

---

## 📜 Citations & References
- **KazMMLU:** Togmanov et al. (2025). *KazMMLU: Evaluating language models on Kazakh...* ACL 2025.
- **Sherkala-Chat:** Koto et al. (2025). *Sherkala-Chat: Building a state-of-the-art LLM for Kazakh...* arXiv:2503.01493.
- **FLORES-200:** NLLB Team et al. (2022). *No Language Left Behind*. arXiv:2207.04672.
- **KazSumm:** Zhabayev et al. (2026). *Improving Kazakh abstractive summarization with LLMs...* Information 17(7).
