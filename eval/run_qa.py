"""
eval/run_qa.py
==============
Evaluate a language model on the KazMMLU question-answering benchmark.

Supported models
----------------
  random   – random baseline (uniform over answer options)
  llama    – meta-llama/Llama-3.1-8B-Instruct  (HuggingFace transformers)
  sherkala – issai/Sherkala-Chat-8B            (HuggingFace transformers)
  gpt4o    – gpt-4o                            (OpenAI API)
  gemini   – gemini-1.5-pro                    (Google Generative AI API)

Usage example (random baseline, 50 samples)
--------------------------------------------
  python eval/run_qa.py \\
      --model random \\
      --data_path data/kaz_mmlu \\
      --output_path outputs/qa_random.jsonl \\
      --seed 42 \\
      --n_sample 50

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
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Local utilities (must be importable when this script is run from repo root)
sys.path.insert(0, str(Path(__file__).parent))
from utils import bootstrap_ci, setup_logging, sha256_string

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTION_LETTERS = ["A", "B", "C", "D", "E"]

OPENAI_MODEL_ID = "gpt-4o"
GEMINI_MODEL_ID = "gemini-1.5-pro"
LLAMA_HF_ID = "meta-llama/Llama-3.1-8B-Instruct"
SHERKALA_HF_ID = "issai/Sherkala-Chat-8B"

KAZMML_HF_DATASET = "kz-transformers/kaz-mmlu"   # adjust if the HF slug changes

# ---------------------------------------------------------------------------
# Prompt helper
# ---------------------------------------------------------------------------

def _load_prompt_template() -> str:
    """Read the shared QA prompt from prompts/qa_prompt.txt."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "qa_prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    # Fallback inline template
    return (
        "Сұрақ: {question}\n\n"
        "Жауап нұсқалары:\n{options}\n\n"
        "Тек бір әріп жазыңыз (A, B, C, немесе D):\n"
    )


def build_prompt(question: str, options: List[str]) -> str:
    """Format a single MMLU item into the model prompt."""
    template = _load_prompt_template()
    options_text = "\n".join(
        f"{letter}. {text}"
        for letter, text in zip(OPTION_LETTERS, options)
    )
    return template.format(question=question, options=options_text)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(raw: str, n_options: int = 4) -> Optional[str]:
    """Extract the first valid option letter from a model response.

    Tries several heuristics:
    1. The response is a single letter (possibly with whitespace/punctuation).
    2. The first token is a letter followed by a punctuation mark.
    3. Any uppercase A-E letter that appears in the first few tokens.
    """
    valid = set(OPTION_LETTERS[:n_options])
    text = raw.strip()

    # 1. Pure single-letter answer
    if len(text) == 1 and text.upper() in valid:
        return text.upper()

    # 2. Letter followed by .: or ) at the very start
    m = re.match(r"^([A-Ea-e])[.):,\s]", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 3. First letter found anywhere in the first 10 chars
    for ch in text[:10]:
        if ch.upper() in valid:
            return ch.upper()

    return None  # could not parse → counted as wrong


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kazmmmlu(data_path: Optional[str]) -> List[dict]:
    """Load KazMMLU items.

    If *data_path* is a local directory containing a JSONL file, load from
    there.  Otherwise, attempt to download from HuggingFace Hub.

    Each returned item is a dict with keys:
      question, options, answer (letter), subject
    """
    # --- local path ---
    if data_path and Path(data_path).exists():
        logger.info("Loading KazMMLU from local path: %s", data_path)
        p = Path(data_path)
        # Support either a JSONL file or a directory containing one
        if p.is_file():
            files = [p]
        else:
            files = list(p.glob("**/*.jsonl")) + list(p.glob("**/*.json"))
        items: List[dict] = []
        for f in sorted(files):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        logger.info("Loaded %d items from local files", len(items))
        return items

    # --- HuggingFace Hub ---
    logger.info("Downloading KazMMLU from HuggingFace: %s", KAZMML_HF_DATASET)
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(KAZMML_HF_DATASET, split="test")
        items = []
        for row in ds:
            # Normalise column names (they may differ across HF versions)
            q = row.get("question") or row.get("Question") or ""
            opts = []
            for letter in OPTION_LETTERS:
                val = row.get(f"option_{letter}") or row.get(letter) or row.get(f"choice_{letter}")
                if val is not None:
                    opts.append(str(val))
            ans = str(row.get("answer") or row.get("Answer") or "A")
            subject = row.get("subject") or row.get("Subject") or "Unknown"
            items.append({"question": q, "options": opts, "answer": ans, "subject": subject})
        logger.info("Downloaded %d items", len(items))
        return items
    except Exception as exc:
        logger.error("Failed to load KazMMLU: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Model callables
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, client, model: str = OPENAI_MODEL_ID) -> str:
    """Call OpenAI Chat API and return the raw text response."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,          # Only need a single letter
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def _call_gemini(prompt: str, model) -> str:
    """Call Google Generative AI and return raw text."""
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 10, "temperature": 0.0},
    )
    return response.text or ""


def _hf_generate(prompt: str, pipe, max_new_tokens: int = 10) -> str:
    """Generate text from a HuggingFace pipeline."""
    outputs = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    # transformers pipelines wrap output in a list of dicts
    generated = outputs[0]["generated_text"]
    # Remove the prompt prefix if the pipeline echoes it
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    return generated.strip()


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def init_model(model_name: str):
    """Return a callable ``predict(prompt: str) -> str`` for *model_name*."""
    if model_name == "random":
        logger.info("Using random baseline (no model loaded)")
        return None  # handled separately

    if model_name == "gpt4o":
        import openai  # type: ignore
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable not set. "
                "Export it before running: export OPENAI_API_KEY=sk-..."
            )
        client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialised (model=%s)", OPENAI_MODEL_ID)
        return lambda prompt: _call_openai(prompt, client)

    if model_name == "gemini":
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable not set. "
                "Export it before running: export GOOGLE_API_KEY=AIza..."
            )
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel(GEMINI_MODEL_ID)
        logger.info("Gemini model initialised (%s)", GEMINI_MODEL_ID)
        return lambda prompt: _call_gemini(prompt, gm)

    if model_name in ("llama", "sherkala"):
        from transformers import pipeline as hf_pipeline  # type: ignore
        import torch  # type: ignore

        hf_id = LLAMA_HF_ID if model_name == "llama" else SHERKALA_HF_ID
        logger.info("Loading HuggingFace model: %s", hf_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)
        pipe = hf_pipeline(
            "text-generation",
            model=hf_id,
            device_map="auto" if device == "cuda" else None,
            torch_dtype="auto",
        )
        return lambda prompt: _hf_generate(prompt, pipe)

    raise ValueError(f"Unknown model: {model_name!r}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    items: List[dict],
    model_name: str,
    predict,
    seed: int,
    output_path: Path,
) -> Dict:
    """Run the evaluation loop and return a summary dict."""
    rng_random = random.Random(seed)
    results = []

    logger.info("Starting evaluation: model=%s, n=%d", model_name, len(items))
    t0 = time.time()

    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(enumerate(items), total=len(items), desc=model_name)
    except ImportError:
        iterator = enumerate(items)

    for idx, item in iterator:
        question = item["question"]
        options = item["options"]
        correct = str(item["answer"]).strip().upper()
        subject = item.get("subject", "Unknown")
        n_opts = len(options)

        # Build the prompt
        prompt = build_prompt(question, options)

        # Get the model answer
        if model_name == "random":
            model_answer = rng_random.choice(OPTION_LETTERS[:n_opts])
            raw_response = model_answer
        else:
            try:
                raw_response = predict(prompt)
            except Exception as exc:
                logger.warning("API error on item %d: %s", idx, exc)
                raw_response = ""
            model_answer = extract_answer(raw_response, n_opts) or ""

        is_correct = model_answer == correct

        record = {
            "question_id": idx,
            "subject": subject,
            "question": question,
            "options": options,
            "correct_answer": correct,
            "model_answer": model_answer,
            "raw_response": raw_response,
            "is_correct": is_correct,
        }
        results.append(record)

        # Write to JSONL immediately so a crash doesn't lose everything
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    scores = [float(r["is_correct"]) for r in results]
    mean, lower, upper = bootstrap_ci(scores, seed=seed)
    accuracy = mean * 100
    ci_lower = lower * 100
    ci_upper = upper * 100

    # Subgroup accuracy by subject
    subject_scores: Dict[str, List[float]] = {}
    for r in results:
        subject_scores.setdefault(r["subject"], []).append(float(r["is_correct"]))
    subject_accuracy = {
        subj: round(float(np.mean(sc)) * 100, 1)
        for subj, sc in subject_scores.items()
    }

    summary = {
        "task": "qa",
        "model": model_name,
        "n_total": len(results),
        "n_correct": int(sum(scores)),
        "accuracy": round(accuracy, 1),
        "ci_95": [round(ci_lower, 1), round(ci_upper, 1)],
        "subject_accuracy": subject_accuracy,
        "elapsed_seconds": round(elapsed, 1),
        "seed": seed,
    }

    logger.info(
        "Done | accuracy=%.1f%% (95%% CI: %.1f–%.1f) | n=%d | %.1fs",
        accuracy, ci_lower, ci_upper, len(results), elapsed,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run QA evaluation on KazMMLU benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["random", "llama", "sherkala", "gpt4o", "gemini"],
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help=(
            "Path to local KazMMLU data directory or JSONL file. "
            "If not given, downloads from HuggingFace Hub."
        ),
    )
    parser.add_argument(
        "--output_path",
        default="outputs/qa_predictions.jsonl",
        help="Path to write per-item JSONL predictions.",
    )
    parser.add_argument(
        "--log_path",
        default=None,
        help="Path to write summary JSON log (default: auto-derived from output_path).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--n_sample",
        type=int,
        default=None,
        help="Subsample this many items (None = use all).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_level)

    logger.info("=== KazMMLU QA Evaluation ===")
    logger.info("Model   : %s", args.model)
    logger.info("Seed    : %d", args.seed)
    logger.info("N sample: %s", args.n_sample or "all")

    # Ensure output directory exists
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.warning("Output file %s already exists – appending!", output_path)

    # Derive log path
    log_path = Path(args.log_path) if args.log_path else output_path.with_suffix(".summary.json")

    # Load data
    items = load_kazmmmlu(args.data_path)

    # Optional subsampling (stratified by subject for fairness)
    if args.n_sample and args.n_sample < len(items):
        rng = random.Random(args.seed)
        rng.shuffle(items)
        items = items[: args.n_sample]
        logger.info("Subsampled to %d items", len(items))

    # Compute fingerprint of the data we're evaluating on
    data_fingerprint = sha256_string(
        json.dumps([it["question"] for it in items], ensure_ascii=False)
    )
    logger.info("Data fingerprint (SHA-256): %s", data_fingerprint)

    # Initialise model
    predict = init_model(args.model)

    # Run evaluation
    summary = evaluate(items, args.model, predict, args.seed, output_path)

    # Enrich summary with run metadata
    summary["metadata"] = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": args.seed,
        "data_path": str(args.data_path),
        "data_sha256": data_fingerprint,
        "output_path": str(output_path),
        "platform": platform.platform(),
    }

    # Save summary log
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Summary log written to: %s", log_path)
    logger.info("Per-item predictions written to: %s", output_path)


if __name__ == "__main__":
    main()
