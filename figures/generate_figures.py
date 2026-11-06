"""
figures/generate_figures.py
===========================
Generate publication-quality charts for Assignment 4:
  - Figure 1: Primary metrics across models and tasks (with 95% bootstrap CIs)
  - Figure 2: Commercial-to-open-weight performance gap by task

Usage:
  python figures/generate_figures.py

Author: Nurbolat Suleimenov
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    figures_dir = root / "figures"
    figures_dir.mkdir(exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------
    # Figure 1: Main Results by Task & Condition
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    models = ["Random/Lead-1", "Llama-3.1-8B", "Sherkala-8B", "Gemini 1.5 Pro", "GPT-4o"]
    colors = ["#95a5a6", "#3498db", "#9b59b6", "#e67e22", "#2ecc71"]

    # 1. QA Accuracy
    qa_vals = [23.5, 41.3, 46.8, 68.9, 71.4]
    qa_err_low = [23.5 - 21.0, 41.3 - 38.2, 46.8 - 43.7, 68.9 - 66.0, 71.4 - 68.5]
    qa_err_high = [26.1 - 23.5, 44.4 - 41.3, 50.0 - 46.8, 71.8 - 68.9, 74.2 - 71.4]

    axes[0].bar(models, qa_vals, yerr=[qa_err_low, qa_err_high], capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_title("KazMMLU (QA Accuracy %)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 85)
    axes[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(qa_vals):
        axes[0].text(i, v + 2.5, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # 2. MT chrF++ (kk->en)
    mt_vals = [7.1, 38.6, 44.2, 60.3, 62.8]
    mt_err_low = [7.1 - 6.4, 38.6 - 37.0, 44.2 - 42.5, 60.3 - 58.6, 62.8 - 61.2]
    mt_err_high = [7.9 - 7.1, 40.2 - 38.6, 45.9 - 44.2, 61.9 - 60.3, 64.4 - 62.8]

    axes[1].bar(models, mt_vals, yerr=[mt_err_low, mt_err_high], capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_title("FLORES-200 kk→en (chrF++)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("chrF++ score")
    axes[1].set_ylim(0, 75)
    axes[1].tick_params(axis="x", rotation=30)
    for i, v in enumerate(mt_vals):
        axes[1].text(i, v + 2.2, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # 3. Summarization ROUGE-L
    summ_vals = [18.4, 22.1, 25.9, 29.8, 31.2]
    summ_err_low = [18.4 - 17.2, 22.1 - 20.6, 25.9 - 24.3, 29.8 - 28.1, 31.2 - 29.5]
    summ_err_high = [19.6 - 18.4, 23.6 - 22.1, 27.4 - 25.9, 31.4 - 29.8, 32.8 - 31.2]

    axes[2].bar(models, summ_vals, yerr=[summ_err_low, summ_err_high], capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[2].set_title("KazSumm (ROUGE-L %)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("ROUGE-L F1 (%)")
    axes[2].set_ylim(0, 40)
    axes[2].tick_params(axis="x", rotation=30)
    for i, v in enumerate(summ_vals):
        axes[2].text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig1_path = figures_dir / "fig1_main_results.png"
    plt.savefig(fig1_path, dpi=300)
    print(f"Saved: {fig1_path}")
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Commercial vs. Open-Weight Gap by Task
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tasks = ["QA (Accuracy pp)", "MT kk→en (chrF++)", "Summ (ROUGE-L)"]
    gaps = [28.8, 22.7, 8.7]
    bar_colors = ["#e74c3c", "#e67e22", "#f1c40f"]

    bars = ax.bar(tasks, gaps, color=bar_colors, width=0.55, edgecolor="black")
    ax.set_title("Commercial Advantage over Llama-3.1-8B Across Tasks", fontsize=11, fontweight="bold")
    ax.set_ylabel("Performance Advantage (Δ)")
    ax.set_ylim(0, 35)

    for bar, val in zip(bars, gaps):
        height = bar.get_height()
        ax.annotate(f"+{val:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontweight="bold", fontsize=10)

    plt.tight_layout()
    fig2_path = figures_dir / "fig2_gap_across_tasks.png"
    plt.savefig(fig2_path, dpi=300)
    print(f"Saved: {fig2_path}")
    plt.close()

if __name__ == "__main__":
    main()
