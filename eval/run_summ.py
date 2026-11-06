"""
eval/run_summ.py
================
Evaluate a language model on the KazSumm Kazakh news summarisation benchmark.

Supported models
----------------
  lead_1   – trivial baseline (first sentence of the article)
  llama    – meta-llama/Llama-3.1-8B-Instruct  (HuggingFace transformers)
  sherkala – issai/Sherkala-Chat-8B            (HuggingFace transformers)
  gpt4o    – gpt-4o                            (OpenAI API)
  gemini   – gemini-1.5-pro                    (Google Generative AI API)

Usage example (lead-1 baseline)
---------------------------------
  python eval/run_summ.py \\
      --model lead_1 \\
      --data_path data/kazSumm \\
      --output_path outputs/summ_lead1.jsonl \\
      --seed 42

Author : Nurbolat Suleimenov
Course : Research Methods and Tools  –  Assignment 4
Date   : 2026-10-28
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    bootstrap_ci,
    compute_chrf,
    compute_rouge_l,
    setup_logging,
    sha256_string,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KAZSUMM_HF_DATASET = "issai/kazSumm"   # adjust if slug changes
OPENAI_MODEL_ID = "gpt-4o"
GEMINI_MODEL_ID = "gemini-1.5-pro"
LLAMA_HF_ID = "meta-llama/Llama-3.1-8B-Instruct"
SHERKALA_HF_ID = "issai/Sherkala-Chat-8B"


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_prompt() -> str:
    p = Path(__file__).parent.parent / "prompts" / "summ_prompt.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "Мақалаға қысқаша тақырып жазыңыз (бір сөйлем):\n\n"
        "{article}\n\nТақырып:"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kazsumm(data_path: Optional[str]) -> List[Tuple[str, str]]:
    """Return list of (article, reference_summary) pairs.

    Downloads KazSumm from HuggingFace Hub when no local path is provided.
    """
    # --- local path ---
    if data_path and Path(data_path).exists():
        logger.info("Loading KazSumm from local path: %s", data_path)
        p = Path(data_path)
        pairs = []
        files = [p] if p.is_file() else sorted(p.glob("**/*.json*"))
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    article = row.get("article") or row.get("text") or row.get("body") or ""
                    summary = row.get("summary") or row.get("headline") or row.get("title") or ""
                    if article and summary:
                        pairs.append((str(article), str(summary)))
        logger.info("Loaded %d pairs from local files", len(pairs))
        return pairs

    # --- HuggingFace Hub ---
    logger.info("Downloading KazSumm from HuggingFace: %s", KAZSUMM_HF_DATASET)
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(KAZSUMM_HF_DATASET, split="test")
        pairs = []
        for row in ds:
            article = row.get("article") or row.get("text") or ""
            summary = row.get("summary") or row.get("headline") or ""
            if article and summary:
                pairs.append((str(article), str(summary)))
        logger.info("Downloaded %d pairs", len(pairs))
        return pairs
    except Exception as exc:
        logger.error("Failed to load KazSumm: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Baselines and model callables
# ---------------------------------------------------------------------------

def _lead_1(article: str) -> str:
    """Return the first sentence of the article as the summary."""
    for sep in (".", "!", "?", "\n"):
        idx = article.find(sep)
        if idx != -1 and idx > 10:
            return article[: idx + 1].strip()
    return article[:200].strip()


def _call_openai(prompt: str, client) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def _call_gemini(prompt: str, model) -> str:
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 128, "temperature": 0.0},
    )
    return (response.text or "").strip()


def _hf_generate(prompt: str, pipe) -> str:
    outputs = pipe(prompt, max_new_tokens=128, do_sample=False)
    generated = outputs[0]["generated_text"]
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    return generated.strip()


def init_model(model_name: str):
    if model_name == "lead_1":
        return None  # handled inline

    if model_name == "gpt4o":
        import openai  # type: ignore
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        client = openai.OpenAI(api_key=api_key)
        return lambda p: _call_openai(p, client)

    if model_name == "gemini":
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel(GEMINI_MODEL_ID)
        return lambda p: _call_gemini(p, gm)

    if model_name in ("llama", "sherkala"):
        from transformers import pipeline as hf_pipeline  # type: ignore
        import torch  # type: ignore
        hf_id = LLAMA_HF_ID if model_name == "llama" else SHERKALA_HF_ID
        logger.info("Loading %s", hf_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = hf_pipeline(
            "text-generation",
            model=hf_id,
            device_map="auto" if device == "cuda" else None,
            torch_dtype="auto",
        )
        return lambda p: _hf_generate(p, pipe)

    raise ValueError(f"Unknown model: {model_name!r}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    pairs: List[Tuple[str, str]],
    model_name: str,
    predict,
    seed: int,
    output_path: Path,
    compute_bertscore: bool = True,
) -> Dict:
    prompt_template = _load_prompt()
    hypotheses: List[str] = []
    references: List[str] = []
    records = []

    logger.info(
        "Starting summarisation evaluation: model=%s n=%d", model_name, len(pairs)
    )
    t0 = time.time()

    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(enumerate(pairs), total=len(pairs), desc=model_name)
    except ImportError:
        iterator = enumerate(pairs)

    for idx, (article, ref) in iterator:
        if model_name == "lead_1":
            hyp = _lead_1(article)
        else:
            prompt = prompt_template.format(article=article)
            try:
                hyp = predict(prompt)
            except Exception as exc:
                logger.warning("Error on item %d: %s", idx, exc)
                hyp = ""

        hypotheses.append(hyp)
        references.append(ref)

        record = {
            "item_id": idx,
            "article": article[:500] + "..." if len(article) > 500 else article,
            "reference": ref,
            "hypothesis": hyp,
        }
        records.append(record)
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0

    # Corpus metrics
    rouge_l = compute_rouge_l(hypotheses, references)
    chrf_score = compute_chrf(hypotheses, references)

    # Bootstrap CI on per-item ROUGE-L
    from rouge_score import rouge_scorer as _rs  # type: ignore
    _scorer = _rs.RougeScorer(["rougeL"], use_stemmer=False)
    per_item_rl = [
        _scorer.score(ref, hyp)["rougeL"].fmeasure * 100
        for hyp, ref in zip(hypotheses, references)
    ]
    _, ci_lo, ci_hi = bootstrap_ci(per_item_rl, seed=seed)

    # Optional BERTScore (slow – skip if bert_score not installed)
    bertscore_f1 = None
    if compute_bertscore:
        try:
            from bert_score import score as bert_score_fn  # type: ignore
            logger.info("Computing BERTScore (this may take a while)…")
            _, _, F = bert_score_fn(hypotheses, references, lang="kk", verbose=False)
            bertscore_f1 = round(float(F.mean().item()), 3)
        except ImportError:
            logger.warning("bert_score not installed; skipping BERTScore.")
        except Exception as exc:
            logger.warning("BERTScore failed: %s", exc)

    summary = {
        "task": "summarisation",
        "model": model_name,
        "n_items": len(pairs),
        "rouge_l": round(rouge_l, 1),
        "chrf_pp": round(chrf_score, 1),
        "bertscore_f1": bertscore_f1,
        "rouge_l_ci_95": [round(ci_lo, 1), round(ci_hi, 1)],
        "elapsed_seconds": round(elapsed, 1),
        "seed": seed,
    }

    logger.info(
        "Done | ROUGE-L=%.1f (95%% CI: %.1f–%.1f) | chrF++=%.1f | BERTScore-F1=%s | %.1fs",
        rouge_l, ci_lo, ci_hi, chrf_score, bertscore_f1, elapsed,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run summarisation evaluation on KazSumm benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["lead_1", "llama", "sherkala", "gpt4o", "gemini"],
    )
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_path", default="outputs/summ_predictions.jsonl")
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_sample", type=int, default=None)
    parser.add_argument(
        "--no_bertscore",
        action="store_true",
        help="Skip BERTScore computation (much faster).",
    )
    parser.add_argument(
        "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_level)

    logger.info("=== KazSumm Summarisation Evaluation ===")
    logger.info("Model : %s", args.model)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = (
        Path(args.log_path)
        if args.log_path
        else output_path.with_suffix(".summary.json")
    )

    pairs = load_kazsumm(args.data_path)

    if args.n_sample and args.n_sample < len(pairs):
        import random
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
        pairs = pairs[: args.n_sample]
        logger.info("Subsampled to %d pairs", len(pairs))

    data_fingerprint = sha256_string(
        json.dumps([p[0][:200] for p in pairs], ensure_ascii=False)
    )

    predict = init_model(args.model)
    summary = evaluate(
        pairs, args.model, predict, args.seed, output_path,
        compute_bertscore=not args.no_bertscore,
    )

    summary["metadata"] = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "data_sha256": data_fingerprint,
        "data_path": str(args.data_path),
        "output_path": str(output_path),
        "platform": platform.platform(),
    }

    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Summary log written to: %s", log_path)


if __name__ == "__main__":
    main()
