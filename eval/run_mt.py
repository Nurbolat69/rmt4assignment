"""
eval/run_mt.py
==============
Evaluate a language model on the FLORES-200 machine-translation benchmark
for the kk↔en language pair.

Supported models
----------------
  copy_source – trivial baseline (output the source unchanged)
  llama       – meta-llama/Llama-3.1-8B-Instruct  (HuggingFace transformers)
  sherkala    – issai/Sherkala-Chat-8B            (HuggingFace transformers)
  gpt4o       – gpt-4o                            (OpenAI API)
  gemini      – gemini-1.5-pro                    (Google Generative AI API)

Usage example (kk→en, GPT-4o, 100 samples)
--------------------------------------------
  python eval/run_mt.py \\
      --model gpt4o \\
      --direction kk_en \\
      --output_path outputs/mt_gpt4o_kk_en.jsonl \\
      --seed 42 \\
      --n_sample 100

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
import sacrebleu

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    bootstrap_ci,
    compute_bleu,
    compute_chrf,
    setup_logging,
    sha256_string,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLORES_HF_DATASET = "facebook/flores"  # requires lang codes: kaz_Cyrl / eng_Latn
FLORES_KK_LANG = "kaz_Cyrl"
FLORES_EN_LANG = "eng_Latn"

OPENAI_MODEL_ID = "gpt-4o"
GEMINI_MODEL_ID = "gemini-1.5-pro"
LLAMA_HF_ID = "meta-llama/Llama-3.1-8B-Instruct"
SHERKALA_HF_ID = "issai/Sherkala-Chat-8B"


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_prompt(direction: str) -> str:
    base = Path(__file__).parent.parent / "prompts"
    fname = "mt_kk_en_prompt.txt" if direction == "kk_en" else "mt_en_kk_prompt.txt"
    p = base / fname
    if p.exists():
        return p.read_text(encoding="utf-8")
    # Inline fallback
    if direction == "kk_en":
        return (
            "Translate the following Kazakh sentence to English. "
            "Output only the translation, nothing else.\n\n"
            "Kazakh: {source}\nEnglish:"
        )
    return (
        "Translate the following English sentence to Kazakh. "
        "Output only the translation, nothing else.\n\n"
        "English: {source}\nKazakh:"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_flores(
    direction: str,
    data_path: Optional[str],
) -> List[Tuple[str, str]]:
    """Return list of (source, reference) pairs.

    Downloads FLORES-200 devtest split from HuggingFace Hub when no local
    path is provided.
    """
    src_lang, tgt_lang = (
        (FLORES_KK_LANG, FLORES_EN_LANG)
        if direction == "kk_en"
        else (FLORES_EN_LANG, FLORES_KK_LANG)
    )

    # --- local path ---
    if data_path and Path(data_path).exists():
        logger.info("Loading FLORES-200 from local path: %s", data_path)
        p = Path(data_path)
        pairs = []
        # Expect two files: src.txt and tgt.txt (one sentence per line)
        src_file = p / f"{src_lang}.devtest"
        tgt_file = p / f"{tgt_lang}.devtest"
        if not (src_file.exists() and tgt_file.exists()):
            # Fall back to any *.txt files alphabetically
            txts = sorted(p.glob("*.txt"))
            if len(txts) >= 2:
                src_file, tgt_file = txts[0], txts[1]
        with open(src_file, encoding="utf-8") as sf, open(tgt_file, encoding="utf-8") as tf:
            for s, t in zip(sf, tf):
                s, t = s.strip(), t.strip()
                if s and t:
                    pairs.append((s, t))
        logger.info("Loaded %d pairs from local files", len(pairs))
        return pairs

    # --- HuggingFace Hub ---
    logger.info("Downloading FLORES-200 from HuggingFace (%s → %s)", src_lang, tgt_lang)
    try:
        from datasets import load_dataset  # type: ignore
        # FLORES-200 on HF is a parallel dataset; devtest split
        ds = load_dataset(FLORES_HF_DATASET, f"{src_lang}-{tgt_lang}", split="devtest")
        pairs = []
        for row in ds:
            # column names vary; common patterns:
            src = (
                row.get(src_lang)
                or row.get("sentence_" + src_lang)
                or row.get("translation", {}).get(src_lang, "")
            )
            tgt = (
                row.get(tgt_lang)
                or row.get("sentence_" + tgt_lang)
                or row.get("translation", {}).get(tgt_lang, "")
            )
            if src and tgt:
                pairs.append((str(src), str(tgt)))
        logger.info("Downloaded %d pairs", len(pairs))
        return pairs
    except Exception as exc:
        logger.error("Failed to load FLORES-200: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Model callables  (identical pattern to run_qa.py)
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, client) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def _call_gemini(prompt: str, model) -> str:
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 256, "temperature": 0.0},
    )
    return (response.text or "").strip()


def _hf_generate(prompt: str, pipe) -> str:
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
    generated = outputs[0]["generated_text"]
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    return generated.strip()


def init_model(model_name: str):
    """Return a callable ``predict(prompt: str) -> str``."""
    if model_name == "copy_source":
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
    direction: str,
    predict,
    seed: int,
    output_path: Path,
) -> Dict:
    """Translate all pairs and compute corpus metrics."""
    prompt_template = _load_prompt(direction)
    hypotheses: List[str] = []
    references: List[str] = []
    records = []

    logger.info(
        "Starting MT evaluation: model=%s direction=%s n=%d",
        model_name, direction, len(pairs),
    )
    t0 = time.time()

    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(enumerate(pairs), total=len(pairs), desc=f"{model_name}/{direction}")
    except ImportError:
        iterator = enumerate(pairs)

    for idx, (src, ref) in iterator:
        if model_name == "copy_source":
            hyp = src
        else:
            prompt = prompt_template.format(source=src)
            try:
                hyp = predict(prompt)
            except Exception as exc:
                logger.warning("Error on item %d: %s", idx, exc)
                hyp = ""

        hypotheses.append(hyp)
        references.append(ref)

        record = {
            "pair_id": idx,
            "source": src,
            "reference": ref,
            "hypothesis": hyp,
        }
        records.append(record)
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0

    # Corpus-level metrics
    chrf_score = compute_chrf(hypotheses, references)
    bleu_score = compute_bleu(hypotheses, references)

    # Bootstrap CI on per-sentence chrF (using sacrebleu sentence_chrf)
    metric_chrf = sacrebleu.CHRF(word_order=2)
    per_sentence_chrf = [
        metric_chrf.sentence_score(h, [r]).score
        for h, r in zip(hypotheses, references)
    ]
    _, ci_lo, ci_hi = bootstrap_ci(per_sentence_chrf, seed=seed)

    summary = {
        "task": "mt",
        "direction": direction,
        "model": model_name,
        "n_pairs": len(pairs),
        "chrf_pp": round(chrf_score, 1),
        "bleu": round(bleu_score, 1),
        "chrf_ci_95": [round(ci_lo, 1), round(ci_hi, 1)],
        "elapsed_seconds": round(elapsed, 1),
        "seed": seed,
    }

    logger.info(
        "Done | chrF++=%.1f (95%% CI: %.1f–%.1f) | BLEU=%.1f | %.1fs",
        chrf_score, ci_lo, ci_hi, bleu_score, elapsed,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run MT evaluation on FLORES-200 (Kazakh↔English).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["copy_source", "llama", "sherkala", "gpt4o", "gemini"],
    )
    parser.add_argument(
        "--direction",
        required=True,
        choices=["kk_en", "en_kk"],
        help="Translation direction: kk_en (Kazakh→English) or en_kk (English→Kazakh).",
    )
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_path", default="outputs/mt_predictions.jsonl")
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_sample", type=int, default=None)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_level)

    logger.info("=== FLORES-200 MT Evaluation ===")
    logger.info("Model     : %s", args.model)
    logger.info("Direction : %s", args.direction)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = (
        Path(args.log_path)
        if args.log_path
        else output_path.with_suffix(".summary.json")
    )

    pairs = load_flores(args.direction, args.data_path)

    if args.n_sample and args.n_sample < len(pairs):
        import random
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
        pairs = pairs[: args.n_sample]
        logger.info("Subsampled to %d pairs", len(pairs))

    data_fingerprint = sha256_string(
        json.dumps([p[0] for p in pairs], ensure_ascii=False)
    )

    predict = init_model(args.model)
    summary = evaluate(pairs, args.model, args.direction, predict, args.seed, output_path)

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
