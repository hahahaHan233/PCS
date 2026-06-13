from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset.csv"
OUTPUT = ROOT / "output" / "pcs_association_figures"

NUMERIC_COLS = [
    "age",
    "height_cm",
    "weight_kg",
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

CATEGORICAL_COLS = [
    "sex",
    "occupation",
    "education_level",
    "smoking",
    "alcohol_use",
    "symptom_duration",
    "abdominal_pain",
    "pain_frequency",
    "radiating_pain",
    "meal_related_pain",
    "hypertension",
    "hyperlipidemia",
    "diabetes",
    "anxiety_depression",
    "prior_abdominal_surgery",
    "gallbladder_wall_thickening",
    "stone_number",
    "stone_location",
    "fatty_liver",
    "gallbladder_atrophy",
]

DISPLAY_NAMES = {
    "age": "Age",
    "height_cm": "Height",
    "weight_kg": "Weight",
    "bmi": "BMI",
    "max_stone_diameter_mm": "Maximum stone diameter",
    "common_bile_duct_diameter_mm": "Common bile duct diameter",
    "alt": "ALT",
    "ast": "AST",
    "alp": "ALP",
    "ggt": "GGT",
    "total_bilirubin": "Total bilirubin",
    "total_bile_acid": "Total bile acid",
    "total_cholesterol": "Total cholesterol",
    "triglyceride": "Triglyceride",
    "alpha_fetoprotein": "Alpha-fetoprotein",
    "ca199": "CA19-9",
    "sex": "Sex",
    "occupation": "Occupation",
    "education_level": "Education level",
    "smoking": "Smoking",
    "alcohol_use": "Alcohol use",
    "symptom_duration": "Symptom duration",
    "abdominal_pain": "Abdominal pain",
    "pain_frequency": "Pain frequency",
    "radiating_pain": "Radiating pain",
    "meal_related_pain": "Meal-related pain",
    "hypertension": "Hypertension",
    "hyperlipidemia": "Hyperlipidemia",
    "diabetes": "Diabetes",
    "anxiety_depression": "Anxiety/depression",
    "prior_abdominal_surgery": "Prior abdominal surgery",
    "gallbladder_wall_thickening": "Gallbladder wall thickening",
    "stone_number": "Stone number",
    "stone_location": "Stone location",
    "fatty_liver": "Fatty liver",
    "gallbladder_atrophy": "Gallbladder atrophy",
}

LAB_DISPLAY = {
    "alt": "ALT",
    "ast": "AST",
    "alp": "ALP",
    "ggt": "GGT",
    "total_bilirubin": "TBil",
    "total_bile_acid": "TBA",
    "total_cholesterol": "TC",
    "triglyceride": "TG",
    "alpha_fetoprotein": "AFP",
    "ca199": "CA19-9",
}


def display_level(level: object) -> str:
    if pd.isna(level):
        return "Missing"
    if isinstance(level, (int, float, np.integer, np.floating)) and float(level) in (0.0, 1.0):
        return "Yes" if int(level) == 1 else "No"
    text = str(level).replace("_", " ")
    replacements = {
        "very short": "very short",
        "junior high or below": "junior high or below",
        "college or above": "college or above",
        "gallbladder fundus": "gallbladder fundus",
        "gallbladder body": "gallbladder body",
        "gallbladder neck": "gallbladder neck",
        "cystic duct": "cystic duct",
        "distal common bile duct": "distal common bile duct",
        "left intrahepatic bile duct": "left intrahepatic bile duct",
        "multiple sites": "multiple sites",
        "no stone": "no stone",
    }
    return replacements.get(text, text)


def configure_style() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )


def save_pdf(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


def p_text(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def rank_biserial_from_u(u: float, n1: int, n0: int) -> float:
    return 2 * u / (n1 * n0) - 1


def numeric_associations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_COLS:
        x1 = pd.to_numeric(df.loc[df["pcs"] == 1, col], errors="coerce").dropna()
        x0 = pd.to_numeric(df.loc[df["pcs"] == 0, col], errors="coerce").dropna()
        if len(x1) < 3 or len(x0) < 3:
            continue
        test = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        rb = rank_biserial_from_u(test.statistic, len(x1), len(x0))
        rows.append(
            {
                "variable": col,
                "label": DISPLAY_NAMES[col],
                "n_pcs": len(x1),
                "n_no_pcs": len(x0),
                "median_pcs": x1.median(),
                "q1_pcs": x1.quantile(0.25),
                "q3_pcs": x1.quantile(0.75),
                "median_no_pcs": x0.median(),
                "q1_no_pcs": x0.quantile(0.25),
                "q3_no_pcs": x0.quantile(0.75),
                "rank_biserial": rb,
                "p_value": test.pvalue,
                "neg_log10_p": -math.log10(max(test.pvalue, 1e-300)),
            }
        )
    out = pd.DataFrame(rows)
    out["p_fdr"] = fdr_bh(out["p_value"].to_numpy())
    return out.sort_values("p_value")


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty(n)
    result[order] = adjusted
    return result


def odds_ratio_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float]:
    # Haldane-Anscombe correction for sparse clinical strata.
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = math.log((aa * dd) / (bb * cc))
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo = math.exp(log_or - 1.96 * se)
    hi = math.exp(log_or + 1.96 * se)
    fisher = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    p = fisher.pvalue if hasattr(fisher, "pvalue") else fisher[1]
    return math.exp(log_or), lo, hi, p


def categorical_associations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_pos = int((df["pcs"] == 1).sum())
    total_neg = int((df["pcs"] == 0).sum())
    for col in CATEGORICAL_COLS:
        values = df[col].dropna()
        if values.empty:
            continue
        counts = df.groupby(col, dropna=True)["pcs"].agg(["count", "sum"]).sort_values("count", ascending=False)
        if counts.empty:
            continue
        reference = counts.index[0]
        for level, row in counts.iterrows():
            a = int(row["sum"])
            b = int(row["count"] - row["sum"])
            c = total_pos - a
            d = total_neg - b
            odds_ratio, ci_low, ci_high, p = odds_ratio_ci(a, b, c, d)
            rows.append(
                {
                    "variable": col,
                    "label": DISPLAY_NAMES.get(col, col),
                    "level": display_level(level),
                    "reference_level": display_level(reference),
                    "n": int(row["count"]),
                    "pcs_n": a,
                    "pcs_rate": a / row["count"] * 100,
                    "odds_ratio": odds_ratio,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p,
                }
            )
    out = pd.DataFrame(rows)
    out["p_fdr"] = fdr_bh(out["p_value"].to_numpy())
    return out.sort_values("p_value")


def plot_numeric_lollipop(stats_df: pd.DataFrame) -> None:
    plot_df = stats_df.sort_values("rank_biserial")
    colors = np.where(plot_df["rank_biserial"] >= 0, "#B24A48", "#3F6F9F")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hlines(y=np.arange(len(plot_df)), xmin=0, xmax=plot_df["rank_biserial"], color="#B8B8B8", lw=1)
    ax.scatter(plot_df["rank_biserial"], np.arange(len(plot_df)), s=32 + 18 * plot_df["neg_log10_p"].clip(0, 5), c=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Rank-biserial effect size (PCS vs no PCS)")
    ax.set_title("Continuous predictors associated with PCS")
    ax.grid(axis="x", color="#E6E6E6", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        x = row["rank_biserial"]
        ha = "left" if x >= 0 else "right"
        offset = 0.018 if x >= 0 else -0.018
        ax.text(x + offset, i, f"p={p_text(row['p_value'])}", va="center", ha=ha, fontsize=6.5, color="#333333")
    save_pdf(fig, "figure_numeric_pcs_association_lollipop.pdf")


def plot_categorical_forest(cat_df: pd.DataFrame) -> None:
    plot_df = cat_df[(cat_df["n"] >= 8) & np.isfinite(cat_df["odds_ratio"])].copy()
    plot_df = plot_df.sort_values("p_value").head(24)
    plot_df = plot_df.sort_values("odds_ratio")
    plot_df["term"] = plot_df["label"] + ": " + plot_df["level"].astype(str)
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(7.4, max(4.8, 0.22 * len(plot_df))))
    xerr = np.vstack([plot_df["odds_ratio"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["odds_ratio"]])
    ax.errorbar(plot_df["odds_ratio"], y, xerr=xerr, fmt="o", color="#2F5D7C", ecolor="#7E9AAF", elinewidth=1, capsize=2, markersize=4)
    ax.axvline(1, color="#222222", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["term"])
    ax.set_xlabel("Odds ratio for PCS (log scale)")
    ax.set_title("Categorical strata associated with PCS")
    ax.grid(axis="x", which="both", color="#E6E6E6", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xmax = max(12, plot_df["ci_high"].replace([np.inf], np.nan).quantile(0.9))
    ax.set_xlim(max(0.05, plot_df["ci_low"].min() * 0.7), xmax * 1.25)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(ax.get_xlim()[1], i, f"{row['pcs_n']}/{row['n']}  p={p_text(row['p_value'])}", ha="right", va="center", fontsize=6.3)
    save_pdf(fig, "figure_categorical_pcs_odds_ratio_forest.pdf")


def plot_lab_small_multiples(df: pd.DataFrame, numeric_stats: pd.DataFrame) -> None:
    lab_cols = [c for c in ["ggt", "ast", "alt", "alp", "total_bile_acid", "total_bilirubin", "ca199", "alpha_fetoprotein"] if c in df.columns]
    melted = df[["pcs"] + lab_cols].melt(id_vars="pcs", var_name="marker", value_name="value").dropna()
    melted["PCS status"] = melted["pcs"].map({0: "No PCS", 1: "PCS"})
    melted["marker_label"] = melted["marker"].map(LAB_DISPLAY)
    melted["log_value"] = np.log1p(pd.to_numeric(melted["value"], errors="coerce").clip(lower=0))

    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.2), sharey=False)
    axes = axes.ravel()
    palette = {"No PCS": "#597A9D", "PCS": "#C65A54"}
    for ax, col in zip(axes, lab_cols):
        d = melted[melted["marker"] == col]
        sns.violinplot(data=d, x="PCS status", y="log_value", palette=palette, inner=None, cut=0, linewidth=0.7, ax=ax)
        sns.boxplot(data=d, x="PCS status", y="log_value", width=0.28, showcaps=True, showfliers=False, boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.7}, whiskerprops={"linewidth": 0.7}, medianprops={"color": "#111111", "linewidth": 0.8}, ax=ax)
        sns.stripplot(data=d, x="PCS status", y="log_value", color="#222222", alpha=0.18, size=1.5, jitter=0.18, ax=ax)
        pval = numeric_stats.loc[numeric_stats["variable"] == col, "p_value"]
        ax.set_title(f"{LAB_DISPLAY.get(col, col)}  p={p_text(float(pval.iloc[0])) if len(pval) else 'NA'}")
        ax.set_xlabel("")
        ax.set_ylabel("log(1 + value)" if ax in axes[::4] else "")
        ax.tick_params(axis="x", rotation=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(lab_cols) :]:
        ax.axis("off")
    fig.suptitle("Preoperative laboratory markers stratified by PCS status", y=1.02, fontsize=10)
    save_pdf(fig, "figure_laboratory_distributions_by_pcs.pdf")


def plot_pcs_rate_heatmap(df: pd.DataFrame) -> None:
    selected = ["symptom_duration", "pain_frequency", "stone_location", "stone_number", "gallbladder_wall_thickening", "radiating_pain"]
    rows = []
    for var in selected:
        for level, group in df.groupby(var, dropna=True):
            if len(group) >= 5:
                rows.append(
                    {
                        "variable": DISPLAY_NAMES[var],
                        "level": display_level(level),
                        "pcs_rate": group["pcs"].mean() * 100,
                        "n": len(group),
                    }
                )
    heat = pd.DataFrame(rows)
    heat["cell"] = heat["pcs_rate"].map(lambda x: f"{x:.1f}%")
    pivot = heat.pivot(index="level", columns="variable", values="pcs_rate")
    annot = heat.pivot(index="level", columns="variable", values="cell")
    fig, ax = plt.subplots(figsize=(7.2, max(4.8, 0.18 * len(pivot))))
    sns.heatmap(pivot, cmap=sns.light_palette("#B24A48", as_cmap=True), annot=annot, fmt="", linewidths=0.4, linecolor="white", cbar_kws={"label": "Observed PCS rate (%)"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("PCS event rates across clinically interpretable strata")
    save_pdf(fig, "figure_pcs_rate_strata_heatmap.pdf")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    df = pd.read_csv(DATASET)

    numeric_stats = numeric_associations(df)
    categorical_stats = categorical_associations(df)
    numeric_stats.to_csv(OUTPUT / "numeric_pcs_association_statistics.csv", index=False, encoding="utf-8-sig")
    categorical_stats.to_csv(OUTPUT / "categorical_pcs_association_statistics.csv", index=False, encoding="utf-8-sig")

    plot_numeric_lollipop(numeric_stats)
    plot_categorical_forest(categorical_stats)
    plot_lab_small_multiples(df, numeric_stats)
    plot_pcs_rate_heatmap(df)
    write_latex_snippet()

    print(f"Saved association statistics and PDF figures to {OUTPUT}")
    print("Top continuous variables:")
    print(numeric_stats[["variable", "rank_biserial", "p_value", "p_fdr"]].head(8).to_string(index=False))
    print("Top categorical strata:")
    print(categorical_stats[["variable", "level", "n", "pcs_rate", "odds_ratio", "p_value", "p_fdr"]].head(8).to_string(index=False))


def write_latex_snippet() -> None:
    latex = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.92\textwidth]{output/pcs_association_figures/figure_numeric_pcs_association_lollipop.pdf}
    \caption{Univariable association between continuous predictors and PCS. Effect sizes are expressed as rank-biserial statistics from the Mann--Whitney U test, where positive values indicate higher distributions among patients with PCS. Marker size reflects the strength of statistical evidence on the $-\log_{10}(p)$ scale.}
    \label{fig:pcs_numeric_association}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.92\textwidth]{output/pcs_association_figures/figure_categorical_pcs_odds_ratio_forest.pdf}
    \caption{Categorical strata associated with PCS in univariable analysis. Points denote odds ratios and horizontal bars denote 95\% confidence intervals with Haldane--Anscombe correction for sparse cells. The dashed vertical line indicates an odds ratio of 1.}
    \label{fig:pcs_categorical_forest}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.92\textwidth]{output/pcs_association_figures/figure_laboratory_distributions_by_pcs.pdf}
    \caption{Distribution of selected preoperative laboratory markers stratified by PCS status. Values are displayed on a $\log(1+x)$ scale to improve visualization of skewed laboratory distributions.}
    \label{fig:pcs_laboratory_distribution}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.92\textwidth]{output/pcs_association_figures/figure_pcs_rate_strata_heatmap.pdf}
    \caption{Observed PCS event rates across selected clinically interpretable categorical strata. Cells show the observed PCS proportion within each stratum.}
    \label{fig:pcs_rate_strata}
\end{figure}
"""
    (OUTPUT / "association_figures_latex.tex").write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    main()
