from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset.csv"
OUTPUT = ROOT / "output" / "modeling"
RANDOM_STATE = 42
N_SPLITS = 5


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    x = df.drop(columns=["pcs"])
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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )


def build_models(y: pd.Series) -> dict[str, object]:
    neg, pos = np.bincount(y)
    scale_pos_weight = neg / pos

    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "Gaussian NB": GaussianNB(),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=220,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return models


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def fold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Average precision": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Brier score": brier_score_loss(y_true, y_prob),
    }


def cross_validate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    x = df.drop(columns=["pcs"])
    y = df["pcs"].astype(int)
    preprocessor = build_preprocessor(df)
    models = build_models(y)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_rows: list[dict[str, object]] = []
    oof_predictions: dict[str, np.ndarray] = {}

    for model_name, estimator in models.items():
        oof_prob = np.zeros(len(df), dtype=float)
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
            pipeline = Pipeline(
                steps=[
                    ("preprocess", clone(preprocessor)),
                    ("model", clone(estimator)),
                ]
            )
            pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
            prob = pipeline.predict_proba(x.iloc[test_idx])[:, 1]
            oof_prob[test_idx] = prob
            row = {"model": model_name, "fold": fold_id}
            row.update(fold_metrics(y.iloc[test_idx].to_numpy(), prob))
            fold_rows.append(row)

        oof_predictions[model_name] = oof_prob

    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    for model_name, group in fold_df.groupby("model", sort=False):
        oof = oof_predictions[model_name]
        row = {"model": model_name}
        for metric in [
            "AUC",
            "Average precision",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "Precision",
            "F1",
            "Brier score",
        ]:
            row[f"{metric} mean"] = group[metric].mean()
            row[f"{metric} sd"] = group[metric].std(ddof=1)
        row["OOF AUC"] = roc_auc_score(y, oof)
        row["OOF average precision"] = average_precision_score(y, oof)
        row["OOF Brier score"] = brier_score_loss(y, oof)
        tn, fp, fn, tp = confusion_matrix(y, (oof >= 0.5).astype(int), labels=[0, 1]).ravel()
        row["OOF TN"] = tn
        row["OOF FP"] = fp
        row["OOF FN"] = fn
        row["OOF TP"] = tp
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["OOF AUC", "OOF Brier score"], ascending=[False, True]
    )
    return fold_df, summary_df, oof_predictions


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_roc(y: pd.Series, oof_predictions: dict[str, np.ndarray], path: Path) -> None:
    plt.figure(figsize=(7.2, 5.4))
    for model_name, prob in oof_predictions.items():
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#8a8f99", lw=1.2, linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Cross-validated ROC curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_pr(y: pd.Series, oof_predictions: dict[str, np.ndarray], path: Path) -> None:
    prevalence = y.mean()
    plt.figure(figsize=(7.2, 5.4))
    for model_name, prob in oof_predictions.items():
        precision, recall, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        plt.plot(recall, precision, lw=2, label=f"{model_name} (AP={ap:.3f})")
    plt.axhline(prevalence, color="#8a8f99", lw=1.2, linestyle="--", label=f"Prevalence={prevalence:.3f}")
    plt.xlabel("Recall / sensitivity")
    plt.ylabel("Precision")
    plt.title("Cross-validated precision-recall curves")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_calibration(y: pd.Series, oof_predictions: dict[str, np.ndarray], path: Path) -> None:
    plt.figure(figsize=(7.2, 5.4))
    plt.plot([0, 1], [0, 1], color="#8a8f99", lw=1.2, linestyle="--", label="Ideal")
    for model_name, prob in oof_predictions.items():
        frac_pos, mean_pred = calibration_curve(y, prob, n_bins=6, strategy="quantile")
        brier = brier_score_loss(y, prob)
        plt.plot(mean_pred, frac_pos, marker="o", lw=2, label=f"{model_name} (Brier={brier:.3f})")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed PCS rate")
    plt.title("Cross-validated calibration curves")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def net_benefit(y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y)
    values = []
    for threshold in thresholds:
        pred = prob >= threshold
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        values.append(tp / n - fp / n * threshold / (1 - threshold))
    return np.asarray(values)


def plot_dca(y: pd.Series, oof_predictions: dict[str, np.ndarray], path: Path) -> None:
    y_arr = y.to_numpy()
    thresholds = np.linspace(0.01, 0.80, 80)
    prevalence = y_arr.mean()
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)

    plt.figure(figsize=(7.2, 5.4))
    for model_name, prob in oof_predictions.items():
        plt.plot(thresholds, net_benefit(y_arr, prob, thresholds), lw=2, label=model_name)
    plt.plot(thresholds, treat_all, color="#8a8f99", linestyle="--", lw=1.2, label="Treat all")
    plt.axhline(0, color="#444b55", linestyle=":", lw=1.2, label="Treat none")
    plt.ylim(bottom=min(-0.05, np.nanmin(treat_all) - 0.02), top=0.35)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision curve analysis based on cross-validated predictions")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_metric_heatmap(summary: pd.DataFrame, path: Path) -> None:
    metrics = [
        "AUC mean",
        "Average precision mean",
        "Sensitivity mean",
        "Specificity mean",
        "F1 mean",
        "Brier score mean",
    ]
    plot_df = summary.set_index("model")[metrics].copy()
    display_df = plot_df.copy()
    display_df["Brier score mean"] = 1 - display_df["Brier score mean"]
    display_df = display_df.rename(columns={"Brier score mean": "1 - Brier mean"})

    plt.figure(figsize=(8.2, 4.8))
    im = plt.imshow(display_df, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(display_df.columns)), display_df.columns, rotation=35, ha="right")
    plt.yticks(np.arange(len(display_df.index)), display_df.index)
    for i in range(display_df.shape[0]):
        for j in range(display_df.shape[1]):
            plt.text(j, i, f"{display_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Cross-validated model performance overview")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_outputs(fold_df: pd.DataFrame, summary_df: pd.DataFrame, oof: dict[str, np.ndarray], df: pd.DataFrame) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_plots()
    y = df["pcs"].astype(int)

    fold_df.to_csv(OUTPUT / "model_performance_by_fold.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT / "model_performance_summary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT / "model_performance.xlsx") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        fold_df.to_excel(writer, sheet_name="folds", index=False)

    oof_df = pd.DataFrame({"pcs": y})
    for model_name, prob in oof.items():
        oof_df[f"{model_name} probability"] = prob
    oof_df.to_csv(OUTPUT / "cross_validated_oof_predictions.csv", index=False, encoding="utf-8-sig")

    plot_roc(y, oof, OUTPUT / "roc_curve.png")
    plot_pr(y, oof, OUTPUT / "pr_curve.png")
    plot_calibration(y, oof, OUTPUT / "calibration_curve.png")
    plot_dca(y, oof, OUTPUT / "dca_curve.png")
    plot_metric_heatmap(summary_df, OUTPUT / "model_metric_heatmap.png")

    metadata = {
        "dataset": str(DATASET),
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1] - 1),
        "outcome": "pcs",
        "pcs_positive": int(y.sum()),
        "pcs_negative": int((1 - y).sum()),
        "pcs_prevalence": float(y.mean()),
        "cv": f"{N_SPLITS}-fold stratified cross-validation",
        "random_state": RANDOM_STATE,
        "models": list(oof.keys()),
    }
    (OUTPUT / "modeling_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(summary_df, metadata)


def fmt_mean_sd(row: pd.Series, metric: str, higher_is_better: bool = True) -> str:
    mean = row[f"{metric} mean"]
    sd = row[f"{metric} sd"]
    arrow = "↑" if higher_is_better else "↓"
    return f"{mean:.3f} ± {sd:.3f} {arrow}"


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(summary_df: pd.DataFrame, metadata: dict[str, object]) -> None:
    report_rows = []
    for _, row in summary_df.iterrows():
        report_rows.append(
            {
                "Model": row["model"],
                "AUC": fmt_mean_sd(row, "AUC"),
                "Average precision": fmt_mean_sd(row, "Average precision"),
                "Sensitivity": fmt_mean_sd(row, "Sensitivity"),
                "Specificity": fmt_mean_sd(row, "Specificity"),
                "F1": fmt_mean_sd(row, "F1"),
                "Brier score": fmt_mean_sd(row, "Brier score", higher_is_better=False),
                "OOF AUC": f"{row['OOF AUC']:.3f}",
                "OOF Brier": f"{row['OOF Brier score']:.3f}",
                "OOF confusion matrix": f"TN={int(row['OOF TN'])}, FP={int(row['OOF FP'])}, FN={int(row['OOF FN'])}, TP={int(row['OOF TP'])}",
            }
        )

    table = markdown_table(report_rows)
    best = summary_df.iloc[0]
    text = f"""# PCS Machine Learning Model Report

## Dataset and validation

- Dataset: `{DATASET.name}`
- Sample size: {metadata["n_rows"]}; PCS positive: {metadata["pcs_positive"]}; PCS negative: {metadata["pcs_negative"]}; prevalence: {metadata["pcs_prevalence"]:.1%}
- Predictors: {metadata["n_features"]} preoperative/perioperative clinical variables
- Validation: {metadata["cv"]}, random state = {metadata["random_state"]}
- Preprocessing: median imputation and standardization for numeric variables; most-frequent imputation and one-hot encoding for categorical variables
- Probability estimates and curves are based on out-of-fold predictions from cross-validation.

## Performance summary

{table}

## Current best model

The top-ranked model by out-of-fold AUC is **{best["model"]}** with OOF AUC = **{best["OOF AUC"]:.3f}**, OOF average precision = **{best["OOF average precision"]:.3f}**, and OOF Brier score = **{best["OOF Brier score"]:.3f}**.

Because PCS prevalence is imbalanced, the report emphasizes sensitivity, precision-recall performance, calibration, and decision curve analysis in addition to AUC.

## Curves

![ROC curves](roc_curve.png)

![Precision-recall curves](pr_curve.png)

![Calibration curves](calibration_curve.png)

![Decision curve analysis](dca_curve.png)

![Metric heatmap](model_metric_heatmap.png)

## Output files

- `model_performance.xlsx`: summary and fold-level performance
- `model_performance_summary.csv`: model-level summary
- `model_performance_by_fold.csv`: fold-level metrics
- `cross_validated_oof_predictions.csv`: out-of-fold predicted probabilities
- `modeling_metadata.json`: reproducibility metadata

## Notes for manuscript writing

- These results are internally validated only and should be described as cross-validated performance.
- Final threshold selection should be clinically justified; the current confusion matrices use threshold = 0.5 for comparability.
- External validation is still required before clinical deployment.
"""
    (OUTPUT / "model_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    df = pd.read_csv(DATASET)
    fold_df, summary_df, oof = cross_validate_models(df)
    save_outputs(fold_df, summary_df, oof, df)
    print(f"Saved model report to {OUTPUT / 'model_report.md'}")
    print(summary_df[["model", "OOF AUC", "OOF average precision", "OOF Brier score"]].to_string(index=False))


if __name__ == "__main__":
    main()
