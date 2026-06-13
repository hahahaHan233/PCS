from __future__ import annotations

import json
import math
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

try:
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception as exc:  # pragma: no cover
    raise RuntimeError("statsmodels is required for manuscript statistical analyses") from exc


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset.csv"
SYMPTOM_DATASET = ROOT / "output" / "pcs_symptom_task" / "pcs_symptom_labeled_dataset.csv"
MODELING_OOF = ROOT / "output" / "modeling" / "cross_validated_oof_predictions.csv"
OUTPUT = ROOT / "output" / "manuscript_experiments"
TABLE_DIR = OUTPUT / "tables"
FIG_DIR = OUTPUT / "figures"
LATEX_DIR = ROOT / "latex_demo_project"
LATEX_TABLE_DIR = LATEX_DIR / "tables"
LATEX_FIG_DIR = LATEX_DIR / "figures" / "pdf"
RANDOM_STATE = 42


DISPLAY = {
    "age": "Age",
    "sex": "Sex",
    "occupation": "Occupation",
    "height_cm": "Height",
    "weight_kg": "Weight",
    "bmi": "BMI",
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
    "max_stone_diameter_mm": "Max stone diameter",
    "stone_number": "Stone number",
    "stone_location": "Stone location",
    "common_bile_duct_diameter_mm": "CBD diameter",
    "fatty_liver": "Fatty liver",
    "gallbladder_atrophy": "Gallbladder atrophy",
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

GROUPS = {
    "Demographics": ["age", "sex", "occupation", "height_cm", "weight_kg", "bmi", "education_level"],
    "Lifestyle": ["smoking", "alcohol_use"],
    "Symptoms": ["symptom_duration", "abdominal_pain", "pain_frequency", "radiating_pain", "meal_related_pain"],
    "Medical history": [
        "hypertension",
        "hyperlipidemia",
        "diabetes",
        "anxiety_depression",
        "prior_abdominal_surgery",
    ],
    "Imaging": [
        "gallbladder_wall_thickening",
        "max_stone_diameter_mm",
        "stone_number",
        "stone_location",
        "common_bile_duct_diameter_mm",
        "fatty_liver",
        "gallbladder_atrophy",
    ],
    "Laboratory": [
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
    ],
}

BASELINE_SELECTED = [
    "age",
    "sex",
    "bmi",
    "symptom_duration",
    "abdominal_pain",
    "pain_frequency",
    "stone_location",
    "common_bile_duct_diameter_mm",
    "alt",
    "ast",
    "alp",
    "ggt",
    "total_bilirubin",
    "total_bile_acid",
    "ca199",
]

MULTIVARIABLE_CANDIDATES = [
    "ggt",
    "ca199",
    "total_bilirubin",
    "total_bile_acid",
    "symptom_duration",
    "pain_frequency",
    "age",
]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close()


def rf_model(n_estimators: int = 300, class_weight: str | None = "balanced_subsample") -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_leaf=4,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def preprocessor_for(x: pd.DataFrame, numeric_strategy: str = "median") -> ColumnTransformer:
    categorical = x.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in x.columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy=numeric_strategy)), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]),
                categorical,
            ),
        ]
    )


def cv_predict_rf(
    df: pd.DataFrame,
    features: list[str],
    n_estimators: int = 300,
    numeric_strategy: str = "median",
    class_weight: str | None = "balanced_subsample",
) -> np.ndarray:
    x = df[features].copy()
    y = df["pcs"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probs = np.zeros(len(df))
    for train_idx, test_idx in cv.split(x, y):
        pre = preprocessor_for(x.iloc[train_idx], numeric_strategy=numeric_strategy)
        pipe = Pipeline([("preprocess", pre), ("model", rf_model(n_estimators, class_weight=class_weight))])
        pipe.fit(x.iloc[train_idx], y[train_idx])
        probs[test_idx] = pipe.predict_proba(x.iloc[test_idx])[:, 1]
    return probs


def metric_row(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "average_precision": average_precision_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "brier": brier_score_loss(y, prob),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_simple_latex_table(
    df: pd.DataFrame,
    path: Path,
    columns: list[str],
    align: str | None = None,
    max_rows: int | None = None,
) -> None:
    out = df[columns].copy()
    if max_rows:
        out = out.head(max_rows)
    align = align or ("l" + "c" * (len(columns) - 1))
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(latex_escape(c) for c in columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in out.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def table1(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y = df["pcs"].astype(int)
    for col in [c for c in df.columns if c != "pcs"]:
        series = df[col]
        label = DISPLAY.get(col, col)
        if pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() > 2:
            pcs = pd.to_numeric(df.loc[y == 1, col], errors="coerce").dropna()
            no = pd.to_numeric(df.loc[y == 0, col], errors="coerce").dropna()
            if len(pcs) and len(no):
                p = stats.mannwhitneyu(pcs, no, alternative="two-sided").pvalue
            else:
                p = np.nan
            rows.append(
                {
                    "variable": col,
                    "label": label,
                    "level": "",
                    "PCS": f"{pcs.median():.2f} ({pcs.quantile(0.25):.2f}-{pcs.quantile(0.75):.2f})",
                    "No PCS": f"{no.median():.2f} ({no.quantile(0.25):.2f}-{no.quantile(0.75):.2f})",
                    "p_value": p,
                    "P": format_p(p),
                }
            )
        else:
            tab = pd.crosstab(series.astype("object").fillna("Missing"), y)
            if 0 not in tab.columns:
                tab[0] = 0
            if 1 not in tab.columns:
                tab[1] = 0
            try:
                if tab.shape == (2, 2):
                    p = stats.fisher_exact(tab[[0, 1]].to_numpy())[1]
                else:
                    p = stats.chi2_contingency(tab[[0, 1]].to_numpy())[1]
            except Exception:
                p = np.nan
            for level, counts in tab.sort_index().iterrows():
                rows.append(
                    {
                        "variable": col,
                        "label": label,
                        "level": str(level),
                        "PCS": f"{int(counts[1])} ({counts[1] / max(1, int((y == 1).sum())) * 100:.1f}%)",
                        "No PCS": f"{int(counts[0])} ({counts[0] / max(1, int((y == 0).sum())) * 100:.1f}%)",
                        "p_value": p,
                        "P": format_p(p),
                    }
                )
    pvals = [r["p_value"] for r in rows]
    valid = np.array([not pd.isna(p) for p in pvals])
    fdr = np.full(len(rows), np.nan)
    if valid.any():
        fdr[valid] = multipletests(np.array(pvals, dtype=float)[valid], method="fdr_bh")[1]
    for idx, q in enumerate(fdr):
        rows[idx]["FDR"] = format_p(q)
        rows[idx]["fdr_value"] = q
    return pd.DataFrame(rows)


def standardized_design(
    df: pd.DataFrame,
    variables: list[str],
    drop_first: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    x_parts = []
    for var in variables:
        s = df[var]
        if pd.api.types.is_numeric_dtype(s) and s.dropna().nunique() > 2:
            vals = pd.to_numeric(s, errors="coerce")
            vals = vals.fillna(vals.median())
            sd = vals.std(ddof=0)
            x_parts.append(pd.DataFrame({var: (vals - vals.mean()) / sd if sd else vals * 0}))
        else:
            vals = s.astype("object").fillna("Missing")
            dummies = pd.get_dummies(vals, prefix=var, drop_first=drop_first, dtype=float)
            x_parts.append(dummies)
    x = pd.concat(x_parts, axis=1) if x_parts else pd.DataFrame(index=df.index)
    return x.astype(float), df["pcs"].astype(int)


def fit_logit(x: pd.DataFrame, y: pd.Series):
    x_const = sm.add_constant(x, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.Logit(y, x_const).fit(disp=False, maxiter=200)


def logistic_analyses(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    univ_rows = []
    predictors = [c for c in df.columns if c != "pcs"]
    for var in predictors:
        x, y = standardized_design(df, [var])
        if x.empty:
            continue
        try:
            model = fit_logit(x, y)
            for term in x.columns:
                coef = model.params[term]
                conf = model.conf_int().loc[term]
                p = model.pvalues[term]
                univ_rows.append(
                    {
                        "variable": var,
                        "term": term,
                        "label": DISPLAY.get(var, var),
                        "OR": math.exp(coef),
                        "CI_low": math.exp(conf[0]),
                        "CI_high": math.exp(conf[1]),
                        "p_value": p,
                    }
                )
        except Exception as exc:
            univ_rows.append(
                {
                    "variable": var,
                    "term": var,
                    "label": DISPLAY.get(var, var),
                    "OR": np.nan,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                    "p_value": np.nan,
                    "note": str(exc),
                }
            )
    univ = pd.DataFrame(univ_rows)
    valid = univ["p_value"].notna()
    univ["fdr_value"] = np.nan
    if valid.any():
        univ.loc[valid, "fdr_value"] = multipletests(univ.loc[valid, "p_value"], method="fdr_bh")[1]
    univ["OR (95% CI)"] = univ.apply(
        lambda r: "NA" if pd.isna(r["OR"]) else f"{r['OR']:.2f} ({r['CI_low']:.2f}-{r['CI_high']:.2f})",
        axis=1,
    )
    univ["P"] = univ["p_value"].map(format_p)
    univ["FDR"] = univ["fdr_value"].map(format_p)

    x_multi, y = standardized_design(df, MULTIVARIABLE_CANDIDATES)
    # VIF pruning keeps the multivariable model readable with only 70 events.
    keep = x_multi.columns.tolist()
    while len(keep) > 2:
        vif_df = pd.DataFrame(
            {
                "term": keep,
                "VIF": [variance_inflation_factor(x_multi[keep].to_numpy(), i) for i in range(len(keep))],
            }
        )
        worst = vif_df.sort_values("VIF", ascending=False).iloc[0]
        if worst["VIF"] <= 5:
            break
        keep.remove(worst["term"])
    model = fit_logit(x_multi[keep], y)
    multi_rows = []
    for term in keep:
        coef = model.params[term]
        conf = model.conf_int().loc[term]
        p = model.pvalues[term]
        source = term.split("_")[0] if "_" in term else term
        multi_rows.append(
            {
                "term": term,
                "label": term,
                "source_variable": source,
                "OR": math.exp(coef),
                "CI_low": math.exp(conf[0]),
                "CI_high": math.exp(conf[1]),
                "p_value": p,
                "OR (95% CI)": f"{math.exp(coef):.2f} ({math.exp(conf[0]):.2f}-{math.exp(conf[1]):.2f})",
                "P": format_p(p),
            }
        )
    multi = pd.DataFrame(multi_rows).sort_values("p_value")
    vif_final = pd.DataFrame(
        {
            "term": keep,
            "VIF": [variance_inflation_factor(x_multi[keep].to_numpy(), i) for i in range(len(keep))],
        }
    )
    return univ.sort_values("p_value"), multi, vif_final


def load_oof_or_compute(df: pd.DataFrame) -> np.ndarray:
    if MODELING_OOF.exists():
        oof = pd.read_csv(MODELING_OOF)
        col = "Random Forest probability"
        if col in oof.columns and len(oof) == len(df):
            return oof[col].to_numpy()
    features = [c for c in df.columns if c != "pcs"]
    return cv_predict_rf(df, features)


def risk_stratification(df: pd.DataFrame, prob: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["pcs"].astype(int).to_numpy()
    strata = pd.cut(prob, bins=[-0.001, 0.10, 0.30, 1.001], labels=["Low (<10%)", "Intermediate (10-30%)", "High (>=30%)"])
    rows = []
    for stratum, idx in pd.Series(strata).groupby(strata).groups.items():
        idx = np.asarray(list(idx))
        rows.append(
            {
                "risk_group": str(stratum),
                "n": len(idx),
                "pcs_n": int(y[idx].sum()),
                "observed_rate": y[idx].mean() if len(idx) else np.nan,
                "mean_predicted_risk": prob[idx].mean() if len(idx) else np.nan,
            }
        )
    stratum_df = pd.DataFrame(rows)
    threshold_rows = []
    for threshold in [0.10, 0.20, 0.30]:
        row = metric_row(y, prob, threshold)
        row["threshold"] = threshold
        row["tp_per_100"] = row["tp"] / len(y) * 100
        row["fp_per_100"] = row["fp"] / len(y) * 100
        row["fn_per_100"] = row["fn"] / len(y) * 100
        row["tn_per_100"] = row["tn"] / len(y) * 100
        threshold_rows.append(row)
    threshold_df = pd.DataFrame(threshold_rows)
    return stratum_df, threshold_df


def plot_risk_strata(stratum_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(stratum_df))
    ax.bar(x, stratum_df["observed_rate"], color="#4f81bd", label="Observed PCS rate")
    ax.plot(x, stratum_df["mean_predicted_risk"], marker="o", color="#c0504d", label="Mean predicted risk")
    for i, row in stratum_df.iterrows():
        ax.text(i, row["observed_rate"] + 0.02, f"{row['pcs_n']}/{row['n']}\n{row['observed_rate']:.1%}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(stratum_df["risk_group"], rotation=15, ha="right")
    ax.set_ylim(0, min(1.0, max(stratum_df["observed_rate"].max(), stratum_df["mean_predicted_risk"].max()) + 0.18))
    ax.set_ylabel("PCS rate")
    ax.set_title("Observed PCS rate by model-defined risk strata")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(FIG_DIR / "risk_stratification_bar.pdf")


def calibration_slope_intercept(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    p = np.clip(prob, 1e-5, 1 - 1e-5)
    x = sm.add_constant(logit(p), has_constant="add")
    try:
        model = sm.Logit(y, x).fit(disp=False, maxiter=100)
        return float(model.params[0]), float(model.params[1])
    except Exception:
        return np.nan, np.nan


def bootstrap_validation(df: pd.DataFrame, n_boot: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [c for c in df.columns if c != "pcs"]
    x = df[features]
    y = df["pcs"].astype(int).to_numpy()
    apparent_prob = load_oof_or_compute(df)
    apparent = metric_row(y, apparent_prob)
    app_intercept, app_slope = calibration_slope_intercept(y, apparent_prob)
    apparent["calibration_intercept"] = app_intercept
    apparent["calibration_slope"] = app_slope

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    pred_matrix = np.zeros((len(df), n_boot), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        if len(np.unique(y[idx])) < 2:
            continue
        pre = preprocessor_for(x.iloc[idx])
        pipe = Pipeline([("preprocess", pre), ("model", rf_model(200))])
        pipe.fit(x.iloc[idx], y[idx])
        prob_boot = pipe.predict_proba(x.iloc[idx])[:, 1]
        prob_orig = pipe.predict_proba(x)[:, 1]
        pred_matrix[:, b] = prob_orig
        boot_m = metric_row(y[idx], prob_boot)
        orig_m = metric_row(y, prob_orig)
        boot_i, boot_s = calibration_slope_intercept(y[idx], prob_boot)
        orig_i, orig_s = calibration_slope_intercept(y, prob_orig)
        rows.append(
            {
                "bootstrap": b + 1,
                "auc_boot": boot_m["auc"],
                "auc_original": orig_m["auc"],
                "brier_boot": boot_m["brier"],
                "brier_original": orig_m["brier"],
                "calibration_intercept_boot": boot_i,
                "calibration_intercept_original": orig_i,
                "calibration_slope_boot": boot_s,
                "calibration_slope_original": orig_s,
            }
        )
    boot = pd.DataFrame(rows)
    summary_rows = []
    for metric in ["auc", "brier", "calibration_intercept", "calibration_slope"]:
        original_col = f"{metric}_original"
        low, high = ci(boot[original_col].to_numpy())
        apparent_value = apparent.get(metric, np.nan)
        if metric in ["auc", "brier"]:
            optimism = (boot[f"{metric}_boot"] - boot[f"{metric}_original"]).mean()
            corrected = apparent_value - optimism
        else:
            optimism = (boot[f"{metric}_boot"] - boot[f"{metric}_original"]).mean()
            corrected = apparent_value - optimism
        summary_rows.append(
            {
                "metric": metric,
                "apparent_oof": apparent_value,
                "bootstrap_original_mean": boot[original_col].mean(),
                "bootstrap_original_95ci_low": low,
                "bootstrap_original_95ci_high": high,
                "mean_optimism": optimism,
                "optimism_corrected": corrected,
            }
        )
    instability = np.nanstd(pred_matrix, axis=1)
    summary_rows.append(
        {
            "metric": "prediction_instability_sd",
            "apparent_oof": np.nan,
            "bootstrap_original_mean": np.nanmean(instability),
            "bootstrap_original_95ci_low": np.nanpercentile(instability, 2.5),
            "bootstrap_original_95ci_high": np.nanpercentile(instability, 97.5),
            "mean_optimism": np.nan,
            "optimism_corrected": np.nanmedian(instability),
        }
    )
    return boot, pd.DataFrame(summary_rows)


def feature_ablation(df: pd.DataFrame) -> pd.DataFrame:
    configs = {
        "Demographics": GROUPS["Demographics"],
        "Symptoms": GROUPS["Symptoms"],
        "Imaging": GROUPS["Imaging"],
        "Laboratory": GROUPS["Laboratory"],
        "Symptoms + Laboratory": GROUPS["Symptoms"] + GROUPS["Laboratory"],
        "All variables": [c for c in df.columns if c != "pcs"],
    }
    y = df["pcs"].astype(int).to_numpy()
    rows = []
    for name, features in configs.items():
        prob = cv_predict_rf(df, [f for f in features if f in df.columns], n_estimators=250)
        row = metric_row(y, prob)
        row["feature_set"] = name
        row["n_features"] = len(features)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def plot_feature_ablation(ablation: pd.DataFrame) -> None:
    plot_df = ablation.sort_values("auc")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.barh(plot_df["feature_set"], plot_df["auc"], color="#6aa84f")
    for _, row in plot_df.iterrows():
        ax.text(row["auc"] + 0.01, row["feature_set"], f"{row['auc']:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.45, min(1.0, plot_df["auc"].max() + 0.08))
    ax.set_xlabel("OOF AUC")
    ax.set_title("Feature-domain ablation for PCS prediction")
    fig.tight_layout()
    save_fig(FIG_DIR / "feature_ablation_auc.pdf")


def calibration_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    x = df.drop(columns=["pcs"])
    y = df["pcs"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probs = {name: np.zeros(len(df)) for name in ["Raw RF", "Platt scaling", "Isotonic calibration"]}
    for train_idx, test_idx in cv.split(x, y):
        pre = preprocessor_for(x.iloc[train_idx])
        base = Pipeline([("preprocess", pre), ("model", rf_model(250))])
        base.fit(x.iloc[train_idx], y[train_idx])
        probs["Raw RF"][test_idx] = base.predict_proba(x.iloc[test_idx])[:, 1]

        for method, label in [("sigmoid", "Platt scaling"), ("isotonic", "Isotonic calibration")]:
            calibrated = Pipeline(
                [
                    ("preprocess", preprocessor_for(x.iloc[train_idx])),
                    (
                        "model",
                        CalibratedClassifierCV(
                            estimator=rf_model(250),
                            method=method,
                            cv=3,
                        ),
                    ),
                ]
            )
            calibrated.fit(x.iloc[train_idx], y[train_idx])
            probs[label][test_idx] = calibrated.predict_proba(x.iloc[test_idx])[:, 1]
    rows = []
    for label, prob in probs.items():
        row = metric_row(y, prob)
        intercept, slope = calibration_slope_intercept(y, prob)
        row["calibration_method"] = label
        row["calibration_intercept"] = intercept
        row["calibration_slope"] = slope
        rows.append(row)
    return pd.DataFrame(rows).sort_values("brier"), probs


def plot_calibration_comparison(df: pd.DataFrame, probs: dict[str, np.ndarray]) -> None:
    y = df["pcs"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", label="Ideal")
    for label, prob in probs.items():
        frac, mean = calibration_curve(y, prob, n_bins=6, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=f"{label} (Brier={brier_score_loss(y, prob):.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed PCS rate")
    ax.set_title("Calibration method comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(FIG_DIR / "calibration_method_comparison.pdf")


def shap_stability(df: pd.DataFrame) -> pd.DataFrame:
    if shap is None:
        return pd.DataFrame({"feature": ["SHAP package unavailable"], "top10_frequency": [np.nan]})
    x = df.drop(columns=["pcs"])
    y = df["pcs"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    counts: dict[str, int] = {}
    mean_ranks: dict[str, list[int]] = {}
    for train_idx, test_idx in cv.split(x, y):
        pre = preprocessor_for(x.iloc[train_idx])
        model = rf_model(250)
        x_train_trans = pre.fit_transform(x.iloc[train_idx])
        x_test_trans = pre.transform(x.iloc[test_idx])
        model.fit(x_train_trans, y[train_idx])
        names = pre.get_feature_names_out()
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_test_trans)
        if isinstance(values, list):
            values = values[1]
        elif getattr(values, "ndim", 0) == 3:
            values = values[:, :, 1]
        arr = np.abs(values).mean(axis=0)
        agg: dict[str, float] = {}
        for name, val in zip(names, arr):
            raw = name.split("__", 1)[-1]
            original = raw
            for col in x.columns:
                if raw == col or raw.startswith(f"{col}_"):
                    original = col
                    break
            agg[original] = agg.get(original, 0.0) + float(val)
        ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (feature, _) in enumerate(ranked[:10], start=1):
            counts[feature] = counts.get(feature, 0) + 1
            mean_ranks.setdefault(feature, []).append(rank)
    rows = []
    for feature, freq in counts.items():
        rows.append(
            {
                "feature": feature,
                "label": DISPLAY.get(feature, feature),
                "top10_frequency": freq / 5,
                "folds_in_top10": freq,
                "mean_rank_when_top10": np.mean(mean_ranks[feature]),
            }
        )
    return pd.DataFrame(rows).sort_values(["top10_frequency", "mean_rank_when_top10"], ascending=[False, True])


def sensitivity_analyses(df: pd.DataFrame) -> pd.DataFrame:
    y = df["pcs"].astype(int).to_numpy()
    features = [c for c in df.columns if c != "pcs"]
    rows = []
    configs = [("Base RF", df.copy(), "median", "balanced_subsample")]
    wins = df.copy()
    for col in features:
        if pd.api.types.is_numeric_dtype(wins[col]) and wins[col].dropna().nunique() > 2:
            lo, hi = wins[col].quantile([0.01, 0.99])
            wins[col] = wins[col].clip(lo, hi)
    configs.append(("Winsorized numeric 1-99%", wins, "median", "balanced_subsample"))
    configs.append(("Mean numeric imputation", df.copy(), "mean", "balanced_subsample"))
    configs.append(("No class weight", df.copy(), "median", None))
    for name, data, impute, class_weight in configs:
        prob = cv_predict_rf(data, features, n_estimators=250, numeric_strategy=impute, class_weight=class_weight)
        row = metric_row(y, prob)
        row["analysis"] = name
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def subgroup_performance(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    y = df["pcs"].astype(int).to_numpy()
    subgroups: dict[str, np.ndarray] = {
        "Female": df["sex"].astype(str).eq("female").to_numpy(),
        "Male": df["sex"].astype(str).eq("male").to_numpy(),
        "Age <60": (df["age"] < 60).to_numpy(),
        "Age >=60": (df["age"] >= 60).to_numpy(),
        "Abdominal pain": (df["abdominal_pain"] == 1).to_numpy(),
        "No abdominal pain": (df["abdominal_pain"] == 0).to_numpy(),
        "High GGT": (df["ggt"] >= df["ggt"].median()).to_numpy(),
        "Low GGT": (df["ggt"] < df["ggt"].median()).to_numpy(),
    }
    rows = []
    for name, mask in subgroups.items():
        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
            continue
        row = metric_row(y[mask], prob[mask])
        row["subgroup"] = name
        row["n"] = int(mask.sum())
        row["pcs_n"] = int(y[mask].sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def onset_exploration(prob: np.ndarray) -> pd.DataFrame:
    if not SYMPTOM_DATASET.exists():
        return pd.DataFrame()
    s = pd.read_csv(SYMPTOM_DATASET)
    if "pcs_onset_time_raw" not in s.columns or len(s) != len(prob):
        return pd.DataFrame()
    s["risk_probability"] = prob
    s["risk_group"] = pd.cut(prob, bins=[-0.001, 0.10, 0.30, 1.001], labels=["Low (<10%)", "Intermediate (10-30%)", "High (>=30%)"])
    s["onset_days"] = pd.to_numeric(s["pcs_onset_time_raw"], errors="coerce")
    pcs = s.loc[(s["pcs"] == 1) & s["onset_days"].notna()].copy()
    rows = []
    for group, g in pcs.groupby("risk_group", observed=False):
        if len(g) == 0:
            continue
        rows.append(
            {
                "risk_group": str(group),
                "pcs_with_onset_n": len(g),
                "onset_median_days": g["onset_days"].median(),
                "onset_iqr_days": g["onset_days"].quantile(0.75) - g["onset_days"].quantile(0.25),
                "mean_predicted_risk": g["risk_probability"].mean(),
            }
        )
    if len(pcs) >= 5:
        rho, p = stats.spearmanr(pcs["risk_probability"], pcs["onset_days"])
        rows.append(
            {
                "risk_group": "Spearman risk vs onset",
                "pcs_with_onset_n": len(pcs),
                "onset_median_days": rho,
                "onset_iqr_days": p,
                "mean_predicted_risk": pcs["risk_probability"].mean(),
            }
        )
    return pd.DataFrame(rows)


def save_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        for name, sheet in sheets.items():
            sheet.to_excel(writer, sheet_name=name[:31], index=False)


def format_outputs(
    baseline: pd.DataFrame,
    univ: pd.DataFrame,
    multi: pd.DataFrame,
    vif: pd.DataFrame,
    stratum: pd.DataFrame,
    threshold: pd.DataFrame,
    boot_summary: pd.DataFrame,
    ablation: pd.DataFrame,
    calib: pd.DataFrame,
    shap_stab: pd.DataFrame,
    sensitivity: pd.DataFrame,
    subgroup: pd.DataFrame,
    onset: pd.DataFrame,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)

    save_excel(
        OUTPUT / "manuscript_experiment_results.xlsx",
        {
            "table1": baseline,
            "univariate_logistic": univ,
            "multivariable_logistic": multi,
            "vif": vif,
            "risk_stratification": stratum,
            "thresholds": threshold,
            "bootstrap_summary": boot_summary,
            "feature_ablation": ablation,
            "calibration": calib,
            "shap_stability": shap_stab,
            "sensitivity": sensitivity,
            "subgroup": subgroup,
            "onset": onset,
        },
    )
    for name, df in {
        "baseline_table": baseline,
        "univariate_logistic": univ,
        "multivariable_logistic": multi,
        "multivariable_vif": vif,
        "risk_stratification": stratum,
        "clinical_threshold_table": threshold,
        "bootstrap_summary": boot_summary,
        "feature_ablation": ablation,
        "calibration_comparison": calib,
        "shap_stability": shap_stab,
        "sensitivity_analyses": sensitivity,
        "subgroup_performance": subgroup,
        "onset_exploration": onset,
    }.items():
        df.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    baseline_show = baseline.loc[baseline["variable"].isin(BASELINE_SELECTED)].copy()
    baseline_show["Characteristic"] = baseline_show.apply(
        lambda r: DISPLAY.get(r["variable"], r["variable"]) if not r["level"] else f"{DISPLAY.get(r['variable'], r['variable'])}: {r['level']}",
        axis=1,
    )
    baseline_show = baseline_show[["Characteristic", "PCS", "No PCS", "P", "FDR"]].head(24)
    write_simple_latex_table(baseline_show, LATEX_TABLE_DIR / "table1_baseline_condensed.tex", baseline_show.columns.tolist(), align="lcccc")

    univ_show = univ[["label", "term", "OR (95% CI)", "P", "FDR"]].head(10).copy()
    univ_show.columns = ["Variable", "Term", "OR (95% CI)", "P", "FDR"]
    write_simple_latex_table(univ_show, LATEX_TABLE_DIR / "table2_univariate_top.tex", univ_show.columns.tolist(), align="llccc")

    multi_show = multi[["label", "OR (95% CI)", "P"]].copy()
    multi_show.columns = ["Predictor", "Adjusted OR (95% CI)", "P"]
    write_simple_latex_table(multi_show, LATEX_TABLE_DIR / "table2_multivariable.tex", multi_show.columns.tolist(), align="lcc")

    risk_show = stratum.copy()
    risk_show["Observed PCS rate"] = risk_show["observed_rate"].map(lambda v: f"{v:.1%}")
    risk_show["Mean predicted risk"] = risk_show["mean_predicted_risk"].map(lambda v: f"{v:.1%}")
    risk_show = risk_show[["risk_group", "n", "pcs_n", "Observed PCS rate", "Mean predicted risk"]]
    risk_show.columns = ["Risk group", "N", "PCS cases", "Observed PCS rate", "Mean predicted risk"]
    write_simple_latex_table(risk_show, LATEX_TABLE_DIR / "table4_risk_stratification.tex", risk_show.columns.tolist(), align="lcccc")

    th_show = threshold.copy()
    th_show["Threshold"] = th_show["threshold"].map(lambda v: f"{v:.0%}")
    th_show["Sensitivity"] = th_show["sensitivity"].map(lambda v: f"{v:.3f}")
    th_show["Specificity"] = th_show["specificity"].map(lambda v: f"{v:.3f}")
    th_show["PPV"] = th_show["precision"].map(lambda v: f"{v:.3f}")
    th_show["TP/100"] = th_show["tp_per_100"].map(lambda v: f"{v:.1f}")
    th_show["FP/100"] = th_show["fp_per_100"].map(lambda v: f"{v:.1f}")
    th_show["FN/100"] = th_show["fn_per_100"].map(lambda v: f"{v:.1f}")
    th_show = th_show[["Threshold", "Sensitivity", "Specificity", "PPV", "TP/100", "FP/100", "FN/100"]]
    write_simple_latex_table(th_show, LATEX_TABLE_DIR / "table_clinical_thresholds.tex", th_show.columns.tolist(), align="ccccccc")

    boot_show = boot_summary.copy()
    for col in ["apparent_oof", "bootstrap_original_mean", "bootstrap_original_95ci_low", "bootstrap_original_95ci_high", "optimism_corrected"]:
        boot_show[col] = boot_show[col].map(lambda v: "NA" if pd.isna(v) else f"{v:.3f}")
    boot_show = boot_show[["metric", "apparent_oof", "bootstrap_original_mean", "bootstrap_original_95ci_low", "bootstrap_original_95ci_high", "optimism_corrected"]]
    boot_show.columns = ["Metric", "OOF apparent", "Bootstrap mean", "CI low", "CI high", "Optimism-corrected"]
    write_simple_latex_table(boot_show, LATEX_TABLE_DIR / "table_bootstrap_validation.tex", boot_show.columns.tolist(), align="lccccc")

    abl_show = ablation.copy()
    for col in ["auc", "average_precision", "brier", "sensitivity", "specificity"]:
        abl_show[col] = abl_show[col].map(lambda v: f"{v:.3f}")
    abl_show = abl_show[["feature_set", "auc", "average_precision", "brier", "sensitivity", "specificity"]]
    abl_show.columns = ["Feature set", "AUC", "AP", "Brier", "Sensitivity", "Specificity"]
    write_simple_latex_table(abl_show, LATEX_TABLE_DIR / "table_feature_ablation.tex", abl_show.columns.tolist(), align="lccccc")

    cal_show = calib.copy()
    for col in ["auc", "brier", "calibration_intercept", "calibration_slope"]:
        cal_show[col] = cal_show[col].map(lambda v: f"{v:.3f}")
    cal_show = cal_show[["calibration_method", "auc", "brier", "calibration_intercept", "calibration_slope"]]
    cal_show.columns = ["Calibration method", "AUC", "Brier", "Intercept", "Slope"]
    write_simple_latex_table(cal_show, LATEX_TABLE_DIR / "table_calibration_comparison.tex", cal_show.columns.tolist(), align="lcccc")

    shap_show = shap_stab.copy().head(10)
    if "top10_frequency" in shap_show.columns:
        shap_show["top10_frequency"] = shap_show["top10_frequency"].map(lambda v: "NA" if pd.isna(v) else f"{v:.1%}")
        shap_show["mean_rank_when_top10"] = shap_show["mean_rank_when_top10"].map(lambda v: "NA" if pd.isna(v) else f"{v:.1f}")
    shap_show = shap_show[["label", "top10_frequency", "folds_in_top10", "mean_rank_when_top10"]] if "folds_in_top10" in shap_show.columns else shap_show
    shap_show.columns = ["Feature", "Top-10 frequency", "Folds", "Mean rank"]
    write_simple_latex_table(shap_show, LATEX_TABLE_DIR / "table_shap_stability.tex", shap_show.columns.tolist(), align="lccc")

    sens_show = sensitivity.copy()
    for col in ["auc", "brier", "sensitivity", "specificity"]:
        sens_show[col] = sens_show[col].map(lambda v: f"{v:.3f}")
    sens_show = sens_show[["analysis", "auc", "brier", "sensitivity", "specificity"]]
    sens_show.columns = ["Analysis", "AUC", "Brier", "Sensitivity", "Specificity"]
    write_simple_latex_table(sens_show, LATEX_TABLE_DIR / "table_sensitivity.tex", sens_show.columns.tolist(), align="lcccc")

    sub_show = subgroup.copy()
    for col in ["auc", "brier", "sensitivity", "specificity"]:
        sub_show[col] = sub_show[col].map(lambda v: f"{v:.3f}")
    sub_show = sub_show[["subgroup", "n", "pcs_n", "auc", "brier", "sensitivity", "specificity"]]
    sub_show.columns = ["Subgroup", "N", "PCS", "AUC", "Brier", "Sensitivity", "Specificity"]
    write_simple_latex_table(sub_show, LATEX_TABLE_DIR / "table_subgroup_performance.tex", sub_show.columns.tolist(), align="lcccccc")

    if not onset.empty:
        onset_show = onset.copy()
        for col in ["onset_median_days", "onset_iqr_days", "mean_predicted_risk"]:
            onset_show[col] = onset_show[col].map(lambda v: "NA" if pd.isna(v) else f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}")
        onset_show.columns = ["Risk group", "N", "Median onset/rho", "IQR/p", "Mean risk"]
        write_simple_latex_table(onset_show, LATEX_TABLE_DIR / "table_onset_exploration.tex", onset_show.columns.tolist(), align="lcccc")

    for fig in ["risk_stratification_bar.pdf", "feature_ablation_auc.pdf", "calibration_method_comparison.pdf"]:
        src = FIG_DIR / fig
        if src.exists():
            shutil.copy2(src, LATEX_FIG_DIR / fig)


def write_report(summary: dict[str, object]) -> None:
    text = f"""# Manuscript Enhancement Experiments

## Purpose

This report collects the additional experiments requested to strengthen the clinical and computational evidence of the PCS manuscript.

## Key outputs

- Full workbook: `output/manuscript_experiments/manuscript_experiment_results.xlsx`
- Condensed LaTeX tables: `latex_demo_project/tables/`
- New PDF figures copied to: `latex_demo_project/figures/pdf/`

## Main findings

- Table 1 and logistic regression outputs were generated to provide a conventional clinical-statistical basis.
- Risk stratification was based on Random Forest out-of-fold probabilities with low, intermediate, and high risk strata.
- Bootstrap internal validation used {summary['bootstrap_n']} resamples.
- Feature ablation compared demographics, symptoms, imaging, laboratory, symptoms+laboratory, and all-variable models.
- Calibration comparison evaluated raw Random Forest, Platt scaling, and isotonic calibration.
- SHAP stability was assessed across cross-validation folds where the SHAP package was available.
- Sensitivity analyses compared winsorized numeric values, mean imputation, and no class weighting.
- Subgroup analyses evaluated model behavior across sex, age, abdominal pain, and GGT-defined strata.
- PCS onset analysis is descriptive only because censoring and follow-up windows are not fully specified.

## Caution

These analyses remain internally validated. External validation and prospective assessment are still required before clinical deployment.
"""
    (OUTPUT / "manuscript_experiments_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_plots()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET)
    y = df["pcs"].astype(int).to_numpy()
    prob = load_oof_or_compute(df)

    baseline = table1(df)
    univ, multi, vif = logistic_analyses(df)
    stratum, threshold = risk_stratification(df, prob)
    plot_risk_strata(stratum)
    boot, boot_summary = bootstrap_validation(df, n_boot=500)
    boot.to_csv(TABLE_DIR / "bootstrap_resamples.csv", index=False, encoding="utf-8-sig")
    ablation = feature_ablation(df)
    plot_feature_ablation(ablation)
    calib, calib_probs = calibration_comparison(df)
    plot_calibration_comparison(df, calib_probs)
    shap_stab = shap_stability(df)
    sensitivity = sensitivity_analyses(df)
    subgroup = subgroup_performance(df, prob)
    onset = onset_exploration(prob)

    format_outputs(
        baseline,
        univ,
        multi,
        vif,
        stratum,
        threshold,
        boot_summary,
        ablation,
        calib,
        shap_stab,
        sensitivity,
        subgroup,
        onset,
    )
    write_report({"bootstrap_n": 500})
    metadata = {
        "n": int(len(df)),
        "pcs_positive": int(y.sum()),
        "pcs_negative": int((1 - y).sum()),
        "bootstrap_resamples": 500,
        "risk_thresholds": [0.10, 0.20, 0.30],
        "risk_strata": ["<0.10", "0.10-0.30", ">=0.30"],
    }
    (OUTPUT / "manuscript_experiments_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved manuscript enhancement outputs to {OUTPUT}")
    print(boot_summary.to_string(index=False))
    print(ablation[["feature_set", "auc", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
