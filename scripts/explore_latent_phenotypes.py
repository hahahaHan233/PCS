from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from build_pcs_symptom_task import build_symptom_dataset
except Exception:  # pragma: no cover
    build_symptom_dataset = None


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset.csv"
SYMPTOM_DATASET = ROOT / "output" / "pcs_symptom_task" / "pcs_symptom_labeled_dataset.csv"
OUTPUT = ROOT / "output" / "latent_phenotypes"
LATEX_FIGURES = ROOT / "latex_demo_project" / "figures"
RANDOM_STATE = 42


NUMERIC_PROFILE_COLS = [
    "age",
    "bmi",
    "max_stone_diameter_mm",
    "common_bile_duct_diameter_mm",
    "alt",
    "ast",
    "alp",
    "ggt",
    "total_bilirubin",
    "total_bile_acid",
    "total_cholesterol",
    "triglyceride",
    "alpha_fetoprotein",
    "ca199",
]


DISPLAY_NAMES = {
    "age": "Age",
    "bmi": "BMI",
    "max_stone_diameter_mm": "Max stone diameter",
    "common_bile_duct_diameter_mm": "CBD diameter",
    "alt": "ALT",
    "ast": "AST",
    "alp": "ALP",
    "ggt": "GGT",
    "total_bilirubin": "Total bilirubin",
    "total_bile_acid": "Total bile acid",
    "total_cholesterol": "Total cholesterol",
    "triglyceride": "Triglyceride",
    "alpha_fetoprotein": "AFP",
    "ca199": "CA19-9",
}


SYMPTOM_DISPLAY = {
    "pain": "Pain",
    "constipation": "Constipation",
    "gi_dysfunction_dyspepsia": "Dyspepsia-related",
    "pcs_unknown": "Unknown",
    "no_pcs": "No PCS",
}


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_analysis_dataset() -> pd.DataFrame:
    if SYMPTOM_DATASET.exists():
        df = pd.read_csv(SYMPTOM_DATASET)
    elif build_symptom_dataset is not None:
        df = build_symptom_dataset()
    else:
        df = pd.read_csv(DATASET)
        df["pcs_symptom_category"] = np.where(df["pcs"] == 1, "pcs_unknown", "no_pcs")
        df["pcs_onset_time_raw"] = np.nan
    return df


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, pd.DataFrame]:
    drop_cols = [
        "source_row",
        "pcs",
        "pcs_onset_time_raw",
        "pcs_symptom_raw",
        "pcs_symptom_label",
        "pcs_symptom_category",
    ]
    x = df.drop(columns=[col for col in drop_cols if col in df.columns])
    categorical_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in x.columns if col not in categorical_cols]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor, x


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def copy_pdf_to_latex(stem: str) -> None:
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
    src = OUTPUT / f"{stem}.pdf"
    dst = LATEX_FIGURES / f"{stem}.pdf"
    if src.exists():
        dst.write_bytes(src.read_bytes())


def scatter_plot(
    coords: np.ndarray,
    hue: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    stem: str,
    palette: dict | None = None,
) -> None:
    plot_df = pd.DataFrame({xlabel: coords[:, 0], ylabel: coords[:, 1], "group": hue.astype(str).to_numpy()})
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    sns.scatterplot(
        data=plot_df,
        x=xlabel,
        y=ylabel,
        hue="group",
        palette=palette,
        s=42,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_title(title)
    ax.legend(title="", loc="best", fontsize=8)
    save_figure(fig, stem)


def choose_kmeans(x_transformed: np.ndarray, k_values: range = range(2, 6)) -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    best_labels = None
    best_score = -np.inf
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit_predict(x_transformed)
        score = silhouette_score(x_transformed, labels)
        rows.append({"k": k, "silhouette": score})
        if score > best_score:
            best_score = score
            best_labels = labels
    assert best_labels is not None
    return best_labels, pd.DataFrame(rows)


def cluster_profiles(df: pd.DataFrame, cluster_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for cluster, group in df.groupby(cluster_col):
        rows.append(
            {
                "cluster": int(cluster),
                "n": int(len(group)),
                "pcs_n": int(group["pcs"].sum()),
                "pcs_rate": group["pcs"].mean(),
                "pcs_onset_median_days": pd.to_numeric(group.loc[group["pcs"] == 1, "pcs_onset_time_raw"], errors="coerce").median(),
                "pcs_onset_iqr_days": (
                    pd.to_numeric(group.loc[group["pcs"] == 1, "pcs_onset_time_raw"], errors="coerce").quantile(0.75)
                    - pd.to_numeric(group.loc[group["pcs"] == 1, "pcs_onset_time_raw"], errors="coerce").quantile(0.25)
                ),
            }
        )
    cluster_summary = pd.DataFrame(rows).sort_values("cluster")

    symptom = (
        df.loc[df["pcs"] == 1]
        .groupby([cluster_col, "pcs_symptom_category"])
        .size()
        .rename("n")
        .reset_index()
    )
    symptom = symptom.rename(columns={cluster_col: "cluster"})
    totals = symptom.groupby("cluster")["n"].transform("sum")
    symptom["percent_within_cluster_pcs"] = symptom["n"] / totals * 100

    numeric_rows = []
    for cluster, group in df.groupby(cluster_col):
        row = {"cluster": int(cluster)}
        for col in NUMERIC_PROFILE_COLS:
            if col in group.columns:
                row[f"{col}_median"] = pd.to_numeric(group[col], errors="coerce").median()
        numeric_rows.append(row)
    numeric_profile = pd.DataFrame(numeric_rows).sort_values("cluster")
    return cluster_summary, symptom, numeric_profile


def plot_cluster_heatmap(df: pd.DataFrame, cluster_col: str) -> None:
    available = [col for col in NUMERIC_PROFILE_COLS if col in df.columns]
    medians = df.groupby(cluster_col)[available].median()
    z = (medians - medians.mean(axis=0)) / medians.std(axis=0, ddof=0).replace(0, np.nan)
    z = z.fillna(0)
    z.columns = [DISPLAY_NAMES.get(col, col) for col in z.columns]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.heatmap(z, cmap="vlag", center=0, linewidths=0.4, linecolor="white", cbar_kws={"label": "Cluster z-score"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Cluster")
    ax.set_title("Latent phenotype cluster profile")
    save_figure(fig, "figS5_cluster_heatmap")


def plot_cluster_pcs_rate(cluster_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.bar(cluster_summary["cluster"].astype(str), cluster_summary["pcs_rate"], color="#5b8cc0")
    for _, row in cluster_summary.iterrows():
        ax.text(str(int(row["cluster"])), row["pcs_rate"], f"{row['pcs_rate']:.1%}", ha="center", va="bottom")
    ax.set_ylim(0, max(0.35, cluster_summary["pcs_rate"].max() + 0.08))
    ax.set_xlabel("Cluster")
    ax.set_ylabel("PCS rate")
    ax.set_title("PCS occurrence rate by latent cluster")
    save_figure(fig, "figS6_cluster_pcs_rate")


def plot_symptom_by_cluster(symptom: pd.DataFrame) -> None:
    if symptom.empty:
        return
    plot_df = symptom.copy()
    plot_df["symptom"] = plot_df["pcs_symptom_category"].map(lambda x: SYMPTOM_DISPLAY.get(x, x))
    pivot = plot_df.pivot_table(
        index="cluster",
        columns="symptom",
        values="percent_within_cluster_pcs",
        fill_value=0,
        aggfunc="sum",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bottom = np.zeros(len(pivot))
    colors = sns.color_palette("Set2", n_colors=len(pivot.columns))
    for color, col in zip(colors, pivot.columns):
        values = pivot[col].to_numpy()
        ax.bar(pivot.index.astype(str), values, bottom=bottom, label=col, color=color)
        bottom += values
    ax.set_ylabel("Symptom distribution among PCS-positive patients (%)")
    ax.set_xlabel("Cluster")
    ax.set_title("PCS symptom subtype distribution by cluster")
    ax.legend(fontsize=8, loc="upper right")
    save_figure(fig, "figS7_cluster_symptom_distribution")


def write_report(
    cluster_summary: pd.DataFrame,
    symptom: pd.DataFrame,
    k_scores: pd.DataFrame,
    pca_variance: np.ndarray,
    pcs_positive_n: int,
) -> None:
    best_k = int(k_scores.sort_values("silhouette", ascending=False).iloc[0]["k"])
    best_score = float(k_scores["silhouette"].max())
    pcs_rate_range = (cluster_summary["pcs_rate"].min(), cluster_summary["pcs_rate"].max())
    onset_text = "not estimable"
    if cluster_summary["pcs_onset_median_days"].notna().any():
        onset_text = (
            f"{cluster_summary['pcs_onset_median_days'].min():.1f}-"
            f"{cluster_summary['pcs_onset_median_days'].max():.1f} days across clusters"
        )

    cluster_table = cluster_summary.copy()
    cluster_table["pcs_rate"] = cluster_table["pcs_rate"].map(lambda x: f"{x:.1%}")
    cluster_table["pcs_onset_median_days"] = cluster_table["pcs_onset_median_days"].map(lambda x: "NA" if pd.isna(x) else f"{x:.1f}")
    cluster_table["pcs_onset_iqr_days"] = cluster_table["pcs_onset_iqr_days"].map(lambda x: "NA" if pd.isna(x) else f"{x:.1f}")

    symptom_table = symptom.copy()
    if not symptom_table.empty:
        symptom_table["percent_within_cluster_pcs"] = symptom_table["percent_within_cluster_pcs"].map(lambda x: f"{x:.1f}")

    text = f"""# Latent Phenotype Exploration Report

## Purpose

This exploratory analysis evaluates whether preoperative/perioperative variables reveal latent clinical phenotypes related to PCS occurrence, PCS symptom subtype, or PCS onset time. The analysis is hypothesis-generating and does not provide causal evidence.

## Methods

- Predictors were processed using the same imputation, scaling, and one-hot encoding strategy used in the supervised models.
- Dimensionality reduction was performed using PCA and t-SNE.
- K-means clustering was evaluated for k = 2 to 5 using silhouette score.
- The best K-means solution was k = {best_k} with silhouette score = {best_score:.3f}.
- PCS-positive patients with available symptom labels: {pcs_positive_n}.

## PCA variance

- PC1 explained {pca_variance[0]:.1%} of transformed feature variance.
- PC2 explained {pca_variance[1]:.1%} of transformed feature variance.

## Cluster summary

{cluster_table.to_markdown(index=False)}

PCS rates ranged from {pcs_rate_range[0]:.1%} to {pcs_rate_range[1]:.1%}. Median PCS onset time was {onset_text}.

## Symptom subtype distribution among PCS-positive patients

{symptom_table.to_markdown(index=False) if not symptom_table.empty else 'No PCS-positive symptom labels available.'}

## Figures

![PCA by PCS status](figS4_pca_by_pcs.pdf)

![t-SNE by PCS status](figS4_tsne_by_pcs.pdf)

![PCS-positive t-SNE by symptom subtype](figS4_tsne_pcs_positive_by_symptom.pdf)

![Cluster heatmap](figS5_cluster_heatmap.pdf)

![Cluster PCS rate](figS6_cluster_pcs_rate.pdf)

![Cluster symptom distribution](figS7_cluster_symptom_distribution.pdf)

## Interpretation

These plots should be used to assess whether PCS status or symptom subtypes show visible separation in the preoperative feature space. If subtype separation remains weak, that supports the interpretation that symptom-subtype prediction is difficult with the currently available variables and sample size.
"""
    (OUTPUT / "latent_phenotype_report.md").write_text(text, encoding="utf-8")


def markdown_table_patch() -> None:
    # Keep pandas.to_markdown independent of the optional tabulate version.
    def _to_markdown(self: pd.DataFrame, index: bool = True, **_: object) -> str:
        df = self if index else self.reset_index(drop=True)
        columns = df.columns.tolist()
        lines = [
            "| " + " | ".join(map(str, columns)) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for _, row in df.iterrows():
            values = []
            for col in columns:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.3f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    pd.DataFrame.to_markdown = _to_markdown  # type: ignore[method-assign]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_plots()
    markdown_table_patch()

    df = load_analysis_dataset()
    preprocessor, predictors = build_preprocessor(df)
    x_transformed = preprocessor.fit_transform(predictors)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(x_transformed)
    tsne = TSNE(n_components=2, perplexity=25, learning_rate="auto", init="pca", random_state=RANDOM_STATE)
    tsne_coords = tsne.fit_transform(x_transformed)

    pcs_hue = df["pcs"].map({0: "No PCS", 1: "PCS"})
    scatter_plot(pca_coords, pcs_hue, "PCA visualization by PCS status", "PC1", "PC2", "figS4_pca_by_pcs")
    scatter_plot(tsne_coords, pcs_hue, "t-SNE visualization by PCS status", "t-SNE 1", "t-SNE 2", "figS4_tsne_by_pcs")

    pcs_pos = (df["pcs"] == 1) & (df["pcs_symptom_category"] != "pcs_unknown")
    if pcs_pos.sum() >= 10:
        symptom_hue = df.loc[pcs_pos, "pcs_symptom_category"].map(lambda x: SYMPTOM_DISPLAY.get(x, x))
        scatter_plot(
            tsne_coords[pcs_pos.to_numpy()],
            symptom_hue,
            "t-SNE among PCS-positive patients by symptom subtype",
            "t-SNE 1",
            "t-SNE 2",
            "figS4_tsne_pcs_positive_by_symptom",
        )

    kmeans_labels, k_scores = choose_kmeans(x_transformed)
    agglomerative_labels = AgglomerativeClustering(n_clusters=len(np.unique(kmeans_labels))).fit_predict(x_transformed)
    df["latent_cluster"] = kmeans_labels
    df["hierarchical_cluster"] = agglomerative_labels

    cluster_summary, symptom, numeric_profile = cluster_profiles(df, "latent_cluster")
    plot_cluster_heatmap(df, "latent_cluster")
    plot_cluster_pcs_rate(cluster_summary)
    plot_symptom_by_cluster(symptom)

    df.to_csv(OUTPUT / "latent_phenotype_assignments.csv", index=False, encoding="utf-8-sig")
    k_scores.to_csv(OUTPUT / "kmeans_silhouette_scores.csv", index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(OUTPUT / "cluster_pcs_rate.csv", index=False, encoding="utf-8-sig")
    symptom.to_csv(OUTPUT / "cluster_symptom_distribution.csv", index=False, encoding="utf-8-sig")
    numeric_profile.to_csv(OUTPUT / "cluster_numeric_profile.csv", index=False, encoding="utf-8-sig")

    write_report(cluster_summary, symptom, k_scores, pca.explained_variance_ratio_, int(pcs_pos.sum()))
    for stem in [
        "figS4_pca_by_pcs",
        "figS4_tsne_by_pcs",
        "figS4_tsne_pcs_positive_by_symptom",
        "figS5_cluster_heatmap",
        "figS6_cluster_pcs_rate",
        "figS7_cluster_symptom_distribution",
    ]:
        copy_pdf_to_latex(stem)

    metadata = {
        "n_rows": int(len(df)),
        "n_transformed_features": int(x_transformed.shape[1]),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "kmeans_silhouette_scores": k_scores.to_dict(orient="records"),
        "selected_k": int(k_scores.sort_values("silhouette", ascending=False).iloc[0]["k"]),
    }
    (OUTPUT / "latent_phenotype_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved latent phenotype report to {OUTPUT / 'latent_phenotype_report.md'}")
    print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()
