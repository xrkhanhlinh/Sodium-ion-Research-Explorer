"""Render saved K=10 topic results without fitting models or changing inputs."""
from pathlib import Path
import hashlib
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/final_system"
OUT = BASE / "topic_figures"
MODELS = {
    "lda": ("LDA", "#2876a8", "Topic-word probability"),
    "bge": ("BERTopic - BGE", "#168578", "c-TF-IDF weight"),
    "batterybert": ("BERTopic - BatteryBERT (pretrained)", "#b36b25", "c-TF-IDF weight"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    inputs = {}
    overview, axes = plt.subplots(1, 3, figsize=(19, 8), sharey=True)
    for ax, (key, (name, color, unit)) in zip(axes, MODELS.items()):
        folder = BASE / "topics" / key
        for filename in ("summary.csv", "keywords.csv", "sources.csv"):
            path = folder / filename
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary = pd.read_csv(folder / "summary.csv")
        topics = summary[summary.topic_id >= 0].sort_values("topic_id")
        assert topics.topic_id.tolist() == list(range(10)), key
        keywords = pd.read_csv(folder / "keywords.csv")
        total = int(summary.documents.sum())
        outliers = int(summary.loc[summary.topic_id < 0, "documents"].sum())
        note = f"Corpus: {total:,} documents | Outliers: {outliers:,} ({outliers / total:.2%})"
        if key == "lda":
            note = f"Corpus: {total:,} documents | Every document assigned to a topic"
        fig, grid = plt.subplots(5, 2, figsize=(18, 22))
        for chart, row in zip(grid.flat, topics.itertuples()):
            words = keywords[keywords.topic_id == row.topic_id].sort_values("rank").head(10)
            assert len(words) == 10
            chart.barh(range(10), words.weight, color=color, height=.72)
            chart.set_yticks(range(10), words.token)
            chart.invert_yaxis()
            label = textwrap.fill(row.topic_label, width=49)
            chart.set_title(f"Topic {row.topic_id}: {label}\n{row.documents:,} documents ({row.document_share:.1%} of corpus)",
                            loc="left", fontsize=12, fontweight="bold", pad=10)
            chart.set_xlabel(unit, fontsize=10)
            chart.tick_params(axis="both", labelsize=10)
            chart.grid(axis="x", alpha=.15)
            chart.set_axisbelow(True)
            chart.margins(x=.12)
        fig.suptitle(f"{name} | 10 topics and top 10 keywords", fontsize=22, fontweight="bold", y=.992)
        fig.text(.5, .971, note, ha="center", fontsize=12)
        fig.text(.5, .012, "Saved model outputs; no fitting performed. Labels are descriptive interpretations, not expert-validated.\n"
                 "Topic IDs are local to each model. Keyword weights are not comparable across models.", ha="center", fontsize=11)
        fig.subplots_adjust(left=.12, right=.97, bottom=.055, top=.935, hspace=.82, wspace=.55)
        fig.savefig(OUT / f"{key}_10_topics.png", dpi=160, facecolor="white")
        plt.close(fig)

        sources = pd.read_csv(folder / "sources.csv")
        left = [0] * 10
        for source, shade in (("Patent", "#e4a24b"), ("Publication", "#377eb8")):
            counts = sources[sources.source == source].set_index("topic_id").documents.reindex(range(10), fill_value=0)
            ax.barh(range(10), counts, left=left, color=shade, label=source)
            left = [a + b for a, b in zip(left, counts)]
        for row in topics.itertuples():
            ax.text(row.documents + 65, row.topic_id, f"{row.documents:,}", va="center", fontsize=10)
        ax.set_yticks(range(10), [f"Topic {i}" for i in range(10)])
        ax.tick_params(labelleft=True)
        ax.set_title(name + f"\nOutliers (excluded): {outliers:,} / {total:,} ({outliers / total:.2%})", fontsize=12, pad=15)
        ax.set_xlim(0, max(topics.documents) * 1.21)
        ax.set_xlabel("Number of documents")
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    overview.suptitle("10 topics per model: document counts and source composition", fontsize=20, fontweight="bold")
    overview.text(.5, .025, "Topic IDs do not identify the same subject across models. BatteryBERT topics use the pretrained encoder.\n"
                  "Source: outputs/final_system/topics/{lda,bge,batterybert}/ (saved results; no refitting).", ha="center", fontsize=11)
    overview.subplots_adjust(left=.055, right=.985, bottom=.15, top=.83, wspace=.25)
    overview.savefig(OUT / "three_models_10_topics.png", dpi=170, facecolor="white")
    plt.close(overview)
    manifest = {"fit_rerun": False, "inputs_sha256": inputs,
                "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.png"))}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved four figures to {OUT}")


if __name__ == "__main__":
    main()
