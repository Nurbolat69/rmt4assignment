"""
eval/utils.py
=============
Shared utility functions for metric computation, bootstrap confidence intervals,
and Holm-Bonferroni multiple-comparison correction.

Author : Nurbolat Suleimenov
Course : Research Methods and Tools  –  Assignment 4
Date   : 2026-10-28
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    scores: Sequence[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute (mean, lower, upper) via percentile bootstrap.

    Parameters
    ----------
    scores      : Per-item scores (e.g. list of 0/1 for QA accuracy).
    n_bootstrap : Number of bootstrap resamples.
    seed        : Random seed for reproducibility.
    ci          : Coverage probability (default 0.95 → 95 % CI).

    Returns
    -------
    (mean, lower, upper) all as floats on the same scale as *scores*.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    if n == 0:
        raise ValueError("scores must be non-empty")

    # Draw n_bootstrap resamples of size n (with replacement)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = arr[indices].mean(axis=1)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean = float(arr.mean())
    return mean, lower, upper


# ---------------------------------------------------------------------------
# Paired bootstrap p-value
# ---------------------------------------------------------------------------

def paired_bootstrap_pvalue(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> float:
    """One-sided paired bootstrap test: H₀ mean(A) ≤ mean(B).

    Returns the fraction of bootstrap resamples where the difference
    (mean_a − mean_b) is ≤ 0, which is the empirical p-value under the
    null hypothesis that system A is not better than system B.

    Parameters
    ----------
    scores_a, scores_b : Per-item scores for systems A and B (same length).
    n_bootstrap        : Number of resamples.
    seed               : Random seed.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"scores_a and scores_b must be the same length "
            f"({len(a)} vs {len(b)})"
        )

    rng = np.random.default_rng(seed)
    n = len(a)
    # Observed difference
    obs_diff = a.mean() - b.mean()

    # Resample differences
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_diff = a[indices].mean(axis=1) - b[indices].mean(axis=1)

    # Shift bootstrap distribution to be centred at zero (under H₀)
    shifted = boot_diff - obs_diff
    p_value = float((shifted <= 0).mean())
    # Clip to avoid exact 0 (finite sample artefact)
    p_value = max(p_value, 1.0 / n_bootstrap)
    return p_value


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------

def holm_bonferroni(pvalues: Sequence[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down procedure.

    Parameters
    ----------
    pvalues : Raw p-values (one per comparison).

    Returns
    -------
    Adjusted p-values in the **same order** as the input list.
    """
    n = len(pvalues)
    if n == 0:
        return []

    # Sorted indices: ascending p-value
    order = sorted(range(n), key=lambda i: pvalues[i])
    corrected = [0.0] * n
    running_max = 0.0

    for rank, idx in enumerate(order):
        # Holm multiplier = (n - rank)
        adjusted = pvalues[idx] * (n - rank)
        # Enforce monotonicity
        running_max = max(running_max, adjusted)
        corrected[idx] = min(running_max, 1.0)

    return corrected


# ---------------------------------------------------------------------------
# MT metrics via sacrebleu
# ---------------------------------------------------------------------------

def compute_chrf(
    hypotheses: List[str],
    references: List[str],
    word_order: int = 2,
) -> float:
    """Compute corpus-level chrF++ score.

    word_order=2 → chrF++ (character n-grams + word bigrams).
    """
    metric = sacrebleu.CHRF(word_order=word_order)
    result = metric.corpus_score(hypotheses, [references])
    return float(result.score)


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Compute corpus-level BLEU (SacreBLEU, tokenize=13a)."""
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return float(result.score)


# ---------------------------------------------------------------------------
# Summarisation metrics via rouge_score
# ---------------------------------------------------------------------------

def compute_rouge_l(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Return macro-averaged ROUGE-L F1 (×100 for parity with papers)."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for hyp, ref in zip(hypotheses, references):
        s = scorer.score(ref, hyp)
        scores.append(s["rougeL"].fmeasure)
    return float(np.mean(scores) * 100)


def compute_rouge1(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Return macro-averaged ROUGE-1 F1 (×100)."""
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    scores = []
    for hyp, ref in zip(hypotheses, references):
        s = scorer.score(ref, hyp)
        scores.append(s["rouge1"].fmeasure)
    return float(np.mean(scores) * 100)


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return hex SHA-256 of *path* without loading it all into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_string(text: str) -> str:
    """Return hex SHA-256 of *text* (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
