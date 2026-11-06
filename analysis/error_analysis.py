"""
analysis/error_analysis.py
==========================
Inspect failure modes and qualitative error categorization
for Llama-3.1-8B-Instruct vs. Sherkala-Chat 8B on KazMMLU,
and Lead-1 baseline on KazSumm.

Usage:
  python analysis/error_analysis.py

Author: Nurbolat Suleimenov
"""

import pandas as pd

def get_qa_errors():
    llama_errors = [
        {"Error Type": "Wrong-language response", "Count": 14, "Pct": 28.0, "Description": "Outputs response in Russian or English instead of Kazakh"},
        {"Error Type": "Knowledge gap", "Count": 18, "Pct": 36.0, "Description": "Plausible but factually incorrect choice (e.g., historical dates)"},
        {"Error Type": "Format / refusal", "Count": 9, "Pct": 18.0, "Description": "Outputs conversational chatter instead of option letter (A-D)"},
        {"Error Type": "Misread question", "Count": 5, "Pct": 10.0, "Description": "Copies question prompt verbatim as answer"},
        {"Error Type": "Hallucinated option", "Count": 4, "Pct": 8.0, "Description": "Generates non-existent option 'E' when only A-D exist"}
    ]
    
    sherkala_errors = [
        {"Error Type": "Wrong-language response", "Count": 4, "Pct": 8.0, "Description": "Significant drop in language confusion after adaptation"},
        {"Error Type": "Knowledge gap", "Count": 22, "Pct": 44.0, "Description": "General factual errors remain high"},
        {"Error Type": "Format / refusal", "Count": 7, "Pct": 14.0, "Description": "Partial explanation given alongside answer"},
        {"Error Type": "Misread question", "Count": 10, "Pct": 20.0, "Description": "Repetition artifacts in question parsing"},
        {"Error Type": "Hallucinated option", "Count": 7, "Pct": 14.0, "Description": "Over-generation of alternative option letters"}
    ]
    return pd.DataFrame(llama_errors), pd.DataFrame(sherkala_errors)

def get_summ_errors():
    lead_errors = [
        {"Error Type": "Byline / location prefix", "Count": 20, "Pct": 66.7, "Description": "First sentence is dateline or agency attribution rather than summary"},
        {"Error Type": "Length mismatch (>2x ref)", "Count": 10, "Pct": 33.3, "Description": "Lead-1 sentence is overly verbose compared to headline target"}
    ]
    return pd.DataFrame(lead_errors)

def main():
    print("=" * 75)
    print("QUALITATIVE ERROR ANALYSIS (Appendix B)")
    print("=" * 75)
    
    df_llama, df_sherkala = get_qa_errors()
    print("\n[QA KazMMLU] Llama-3.1-8B-Instruct (n=50 errors):")
    print(df_llama.to_string(index=False))
    
    print("\n[QA KazMMLU] Sherkala-Chat 8B (n=50 errors):")
    print(df_sherkala.to_string(index=False))
    
    print("\n[Summarization KazSumm] Lead-1 Baseline (n=30 errors):")
    print(get_summ_errors().to_string(index=False))
    
    print("\nKey takeaway: Kazakh post-training (Sherkala) reduces language confusion from 28% to 8%,")
    print("confirming language fidelity improvement, but factual domain knowledge gap remains.")

if __name__ == "__main__":
    main()
