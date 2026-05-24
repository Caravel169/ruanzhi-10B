"""
成员3负责: Generate charts for the report from final_comparison.csv.

Run after evaluate.py:
    python evaluate/visualize.py

Outputs (in ./results/figures/):
    asr_comparison.png      -- bar chart: ASR across all methods
    queries_comparison.png  -- bar chart: avg queries across methods
    defense_comparison.png  -- grouped bar: clean vs robust model
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import RESULTS_DIR

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = sns.color_palette("Set2")


def load_results():
    path = os.path.join(RESULTS_DIR, "final_comparison.csv")
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run evaluate.py first.")
        return None
    return pd.read_csv(path)


def plot_asr(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(df["Method"], df["ASR (%)"], color=PALETTE[:len(df)], edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Attack Success Rate Comparison")
    ax.set_ylim(0, 105)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = os.path.join(out_dir, "asr_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_queries(df, out_dir):
    if df["Avg Queries"].isna().all():
        print("Skipping queries chart: no query data available.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(df["Method"], df["Avg Queries"], color=PALETTE[:len(df)], edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_ylabel("Average Model Queries per Attack")
    ax.set_title("Query Efficiency Comparison\n(lower is better)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = os.path.join(out_dir, "queries_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_wir_vs_awir(df, out_dir):
    """Focused comparison of WIR vs AWIR for the improved attack section."""
    subset = df[df["Method"].str.contains("WIR", na=False)].copy()
    if len(subset) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colors = [PALETTE[0], PALETTE[1]]

    for ax, col, title in zip(
        axes,
        ["ASR (%)", "Avg Queries"],
        ["Attack Success Rate (%)", "Avg Queries (lower=better)"],
    ):
        bars = ax.bar(subset["Method"], subset[col], color=colors, edgecolor="white", width=0.4)
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set_title(title)
        ax.set_ylim(0, subset[col].max() * 1.25)
        ax.tick_params(axis="x", rotation=10)

    fig.suptitle("Standard WIR vs Attention-Weighted WIR (AWIR)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(out_dir, "wir_vs_awir.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_defense(out_dir):
    """Grouped bar chart: Clean BERT vs Robust BERT ASR for each attack method."""
    path = os.path.join(RESULTS_DIR, "defense_comparison.csv")
    if not os.path.exists(path):
        print("[SKIP] defense_comparison.csv not found. Run evaluate.py after member 5 finishes.")
        return

    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        m = row["Method"]
        if m.startswith("Clean BERT + "):
            records.append({"Attack": m[len("Clean BERT + "):], "Model": "Clean BERT", "ASR (%)": row["ASR (%)"]})
        elif m.startswith("Robust BERT + "):
            records.append({"Attack": m[len("Robust BERT + "):], "Model": "Robust BERT", "ASR (%)": row["ASR (%)"]})

    if not records:
        print("[SKIP] No defense data to plot.")
        return

    plot_df = pd.DataFrame(records)
    attacks = plot_df["Attack"].unique()
    x = np.arange(len(attacks))
    width = 0.35

    def _get(attack, model):
        sel = plot_df[(plot_df["Attack"] == attack) & (plot_df["Model"] == model)]["ASR (%)"]
        return float(sel.values[0]) if len(sel) > 0 else 0.0

    clean_vals = [_get(a, "Clean BERT") for a in attacks]
    robust_vals = [_get(a, "Robust BERT") for a in attacks]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, clean_vals, width, label="Clean BERT", color=PALETTE[0], edgecolor="white")
    bars2 = ax.bar(x + width / 2, robust_vals, width, label="Robust BERT (adv. trained)", color=PALETTE[1], edgecolor="white")
    ax.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Defense Effect: Adversarial Training vs Each Attack\n(lower robust bar = better defense)")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=10, ha="right")
    ax.set_ylim(0, 115)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "defense_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = load_results()
    if df is None:
        return

    print(df[["Method", "ASR (%)", "Avg Queries"]].to_string(index=False))

    plot_asr(df, FIGURES_DIR)
    plot_queries(df, FIGURES_DIR)
    plot_wir_vs_awir(df, FIGURES_DIR)
    plot_defense(FIGURES_DIR)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
