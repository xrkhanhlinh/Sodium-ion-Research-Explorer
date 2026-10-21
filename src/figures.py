"""Six reproducible Stage 1 figures; each plot reads its exported CSV table."""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"Patent": "#2878A5", "Publication": "#C76E35"}


def build_figures(raw, clean, quality, root, dpi=300):
    figdir = root / "outputs/figures/data_preparation"
    tabdir = root / "outputs/tables/data_preparation"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    manifest = []

    def table(name, data):
        target = tabdir / (name + ".csv")
        data.to_csv(target, index=False, encoding="utf-8-sig")
        return pd.read_csv(target)

    def save(name, fig, description):
        target = figdir / (name + ".png")
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append(dict(figure=str(target.relative_to(root)), table=str((tabdir / (name + ".csv")).relative_to(root)), dpi=dpi, description=description))

    name = "figure_01_dataset_composition"
    records = []
    for source in COLORS:
        records.extend([dict(source=source, population="Original", count=int(raw.source.eq(source).sum())), dict(source=source, population="Clean master (all retained)", count=int(clean.source.eq(source).sum())), dict(source=source, population="Eligible main corpus", count=int((clean.source.eq(source) & clean.eligible_topic).sum()))])
    data = table(name, pd.DataFrame(records))
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = data.population.drop_duplicates().tolist()
    x = np.arange(len(labels))
    for offset, source in enumerate(COLORS):
        values = data[data.source == source].set_index("population").loc[labels, "count"]
        bars = ax.bar(x + (offset - .5) * .34, values, .34, label=source, color=COLORS[source])
        ax.bar_label(bars, fmt="%.0f", fontsize=9, padding=3)
    ax.set(xticks=x, xticklabels=labels, ylabel="Documents", title="Dataset composition before and after preparation")
    ax.set_ylim(0, data["count"].max() * 1.15)
    ax.legend()
    fig.tight_layout()
    save(name, fig, "All rows retained in clean masters; exclusions apply only to modeling/search corpora.")

    name = "figure_02_missing_values"
    data = table(name, quality[quality.stage.isin(["raw_unified", "clean_master", "main_corpus"])].copy())
    features = raw.columns[:14].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    for ax, source in zip(axes, COLORS):
        pivot = data[data.source == source].pivot(index="feature", columns="stage", values="missing_rate").reindex(index=features, columns=["raw_unified", "clean_master", "main_corpus"])
        image = ax.imshow(pivot.values * 100, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(3), ["Raw", "Clean master", "Main corpus"], rotation=25, ha="right")
        ax.set_yticks(range(len(features)), features)
        ax.set_title(source)
        for r in range(len(features)):
            for c in range(3):
                value = pivot.iloc[r, c]
                label = "n/a" if pd.isna(value) else f"{100*value:.2f}" if 0 < value < .01 else f"{100*value:.1f}"
                ax.text(c, r, label, ha="center", va="center", fontsize=8, color="white" if value > .65 else "black")
    fig.suptitle("Missing values by source and preparation stage (%)")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Missing (%)", fraction=.035, pad=.04)
    save(name, fig, "Raw language for Patent is absent; detected text language is recorded after normalization.")

    name = "figure_03_publication_trends"
    parts = []
    for population, frame in [("Original", raw), ("Main corpus", clean[clean.eligible_topic])]:
        counts = frame.dropna(subset=["year"]).groupby(["source", "year"]).size().reset_index(name="count")
        counts["population"] = population
        parts.append(counts)
    data = table(name, pd.concat(parts, ignore_index=True))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for (source, population), group in data.groupby(["source", "population"]):
        ax.plot(group.year, group["count"], color=COLORS[source], linestyle="-" if population == "Main corpus" else "--", label=f"{source}: {population}", linewidth=1.7)
    ax.set(xlabel="Publication year", ylabel="Documents", title="Publication trends before and after corpus selection")
    ax.legend()
    fig.text(.10, .01, "Patent: earliest valid publication date in PD per record; not priority year or counts of every family member.\nPublication: PY. Missing/unresolved years are omitted; coverage is limited to the supplied exports.", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, 1))
    save(name, fig, "Patent year is earliest valid publication date in PD; priority dates remain separate.")

    name = "figure_04_document_lengths"
    parts = []
    for population, frame in [("Original", raw), ("Main corpus", clean[clean.eligible_topic])]:
        for field in ["title", "abstract"]:
            part = frame[["doc_id", "source"]].copy()
            part["field"], part["population"] = field, population
            part["characters"] = frame[field].str.len().values
            parts.append(part)
    data = table(name, pd.concat(parts, ignore_index=True))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, field in enumerate(["title", "abstract"]):
        for col, population in enumerate(["Original", "Main corpus"]):
            ax = axes[row, col]
            for source in COLORS:
                values = data[(data.source == source) & (data.field == field) & (data.population == population)].characters
                ax.hist(values, bins=np.geomspace(1, max(2, data.characters.max() + 1), 45), histtype="step", linewidth=1.6, color=COLORS[source], label=source)
            ax.set(xscale="log", xlabel=f"{field.title()} length (characters; log scale)", ylabel="Documents", title=f"{population}: {field.title()}")
            ax.legend()
    fig.suptitle("Document length distributions (empty fields = 0, omitted on log axis)")
    fig.tight_layout()
    save(name, fig, "Source table includes zero lengths; the log-scale histograms omit zero values.")

    name = "figure_05_cleaning_summary"
    categories = ["STRUCTURE_REVIEW", "CONFIRMED_DUPLICATE", "RETRACTED", "OUT_OF_SCOPE", "RELEVANCE_REVIEW", "LANGUAGE_REVIEW", "TEXT_QUALITY_REVIEW", "MAIN_CORPUS"]
    complete_index = pd.MultiIndex.from_product([list(COLORS), categories], names=["source", "disposition"])
    counts = clean.groupby(["source", "disposition"]).size().reindex(complete_index, fill_value=0).reset_index(name="count")
    data = table(name, counts)
    pivot = data.pivot(index="disposition", columns="source", values="count").fillna(0).reindex(columns=list(COLORS))
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(pivot))
    left = np.zeros(len(pivot))
    for source in COLORS:
        ax.barh(y, pivot[source], left=left, label=source, color=COLORS[source])
        left += pivot[source].values
    for i, count in enumerate(left):
        ax.text(count + max(left) * .01, i, f"{int(count):,}", va="center", fontsize=9)
    ax.set(yticks=y, yticklabels=[v.replace("_", " ").title() for v in pivot.index], xlabel="Documents", title="Cleaning disposition: mutually exclusive counts")
    ax.set_xlim(0, max(left) * 1.15)
    ax.legend()
    repaired = int(clean.repair_status.eq("repaired").sum())
    unresolved = int(clean.structural_issue.sum())
    fig.text(.10, .01, f"Structural records repaired: {repaired}; unresolved: {unresolved}. Repaired rows follow their final disposition and are counted once.", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save(name, fig, "Precedence: structure, confirmed duplicate, retracted, out of scope, relevance review, language review, text quality review, main corpus.")

    name = "figure_06_domain_relevance"
    records = []
    for source in COLORS:
        for label in ["CORE", "RELATED", "OUT_OF_SCOPE", "UNCERTAIN"]:
            records.append(dict(source=source, relevance=label, count=int(((clean.source == source) & (clean.relevance == label)).sum())))
    data = table(name, pd.DataFrame(records))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = ["CORE", "RELATED", "OUT_OF_SCOPE", "UNCERTAIN"]
    x = np.arange(4)
    for offset, source in enumerate(COLORS):
        values = data[data.source == source].set_index("relevance").loc[labels, "count"]
        bars = ax.bar(x + (offset - .5) * .36, values, .36, label=source, color=COLORS[source])
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylabel="Documents", title="Domain relevance: heuristic labels, not expert ground truth")
    ax.set_ylim(0, data["count"].max() * 1.17)
    ax.legend()
    fig.tight_layout()
    save(name, fig, "All normalized records are shown; labels require expert review.")
    return manifest
