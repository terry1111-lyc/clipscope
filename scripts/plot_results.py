"""Render actual measured reports; no hand-entered metric values."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("experiments/results"))
    p.add_argument("--out", type=Path, default=Path("docs/experiment.png"))
    args = p.parse_args()
    reports = [json.loads((args.results / f"b32-openai-test-{mode}.json").read_text()) for mode in ("baseline", "ensemble")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": "#d8ddd5",
                         "text.color": "#283a30", "axes.labelcolor": "#283a30", "xtick.color": "#637167", "ytick.color": "#637167"})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), facecolor="#f6f5f1")
    colors = ["#28533f", "#d18160"]
    labels = ["Baseline", "Prompt ensemble"]
    metrics = ["hit@1", "mrr", "ndcg@10"]
    for j, (report, color, label) in enumerate(zip(reports, colors, labels)):
        mean = np.array([report["metrics"][k]["mean"] for k in metrics])
        lo = np.array([report["metrics"][k]["ci95"][0] for k in metrics])
        hi = np.array([report["metrics"][k]["ci95"][1] for k in metrics])
        axes[0].bar(np.arange(3) + (j - .5) * .32, mean, .29, color=color, label=label,
                    yerr=[mean-lo, hi-mean], capsize=3, error_kw={"elinewidth": 1})
        slices = ["attribute", "composition", "paraphrase"]
        axes[1].plot(np.arange(3), [report["slices"][k]["hit@1"]["mean"] for k in slices], "o-", color=color, label=label)
        axes[2].bar(j, report["latency_ms"]["p50"], color=color, width=.55)
        axes[2].text(j, report["latency_ms"]["p50"] + 1, f"{report['latency_ms']['p50']:.1f}", ha="center")
    axes[0].set_xticks(range(3), ["Hit@1", "MRR", "nDCG@10"])
    axes[0].set_ylim(0, 1); axes[0].set_title("Retrieval quality · 95% grouped CI", loc="left", fontsize=11, pad=15)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].set_xticks(range(3), ["Attribute", "Composition", "Paraphrase"], fontsize=9)
    axes[1].set_ylim(0, 1); axes[1].set_title("Hit@1 by query slice", loc="left", fontsize=11, pad=15)
    axes[2].set_xticks(range(2), labels, fontsize=9); axes[2].set_ylim(0, max(r["latency_ms"]["p50"] for r in reports) * 1.4)
    axes[2].set_ylabel("Milliseconds"); axes[2].set_title("Warm CPU search · median", loc="left", fontsize=11, pad=15)
    for ax in axes:
        ax.set_facecolor("#f6f5f1"); ax.grid(axis="y", color="#e3e6df", linewidth=.7); ax.set_axisbelow(True); ax.tick_params(length=0)
    fig.suptitle("CLIPScope / A small compositional diagnostic", x=.06, ha="left", fontsize=19, fontweight="bold")
    fig.text(.06, .025, "ViT-B/32 OpenAI · 48 images · 54 held-out queries · 6 groups · FP32 / CPU\nSynthetic evidence only. Paired nDCG delta CI includes zero; no robust improvement claim.", fontsize=9, color="#637167")
    fig.subplots_adjust(left=.06, right=.98, top=.78, bottom=.22, wspace=.32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor())
    fig.savefig(args.out.with_suffix(".svg"), facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
