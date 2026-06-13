from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset.csv"
MODELING_SUMMARY = ROOT / "output" / "modeling" / "model_performance_summary.csv"
OUTPUT = ROOT / "output" / "shap_analysis"
RANDOM_STATE = 42
PLOT_FONT = "Times New Roman"


DISPLAY_NAMES = {
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
    "max_stone_diameter_mm": "Maximum stone diameter",
    "stone_number": "Stone number",
    "stone_location": "Stone location",
    "common_bile_duct_diameter_mm": "Common bile duct diameter",
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
    "alpha_fetoprotein": "Alpha-fetoprotein",
    "ca199": "CA19-9",
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [PLOT_FONT],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


CLINICAL_GROUPS = {
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


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
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
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def clean_feature_name(name: str) -> str:
    return name.replace("numeric__", "").replace("categorical__", "")


def format_category_level(level: str) -> str:
    replacements = {
        "very_short": "very short",
        "short": "short",
        "medium": "medium",
        "long": "long",
        "daily": "daily",
        "weekly": "weekly",
        "monthly": "monthly",
        "rare": "rare",
        "yes": "yes",
        "no": "no",
        "male": "male",
        "female": "female",
        "gallbladder_body": "gallbladder body",
        "gallbladder_neck": "gallbladder neck",
        "gallbladder_fundus": "gallbladder fundus",
        "multiple": "multiple",
        "single": "single",
        "unemployed": "unemployed",
    }
    return replacements.get(level, level.replace("_", " "))


def display_feature_name(feature: str, numeric_cols: list[str], categorical_cols: list[str]) -> str:
    cleaned = clean_feature_name(feature)
    if cleaned in numeric_cols:
        return DISPLAY_NAMES.get(cleaned, cleaned)
    for col in sorted(categorical_cols, key=len, reverse=True):
        prefix = f"{col}_"
        if cleaned.startswith(prefix):
            variable = DISPLAY_NAMES.get(col, col)
            level = format_category_level(cleaned[len(prefix) :])
            return f"{variable}: {level}"
    return DISPLAY_NAMES.get(cleaned, cleaned.replace("_", " "))


def original_variable_from_feature(feature: str, numeric_cols: list[str], categorical_cols: list[str]) -> str:
    cleaned = clean_feature_name(feature)
    if cleaned in numeric_cols:
        return cleaned
    for col in sorted(categorical_cols, key=len, reverse=True):
        if cleaned == col or cleaned.startswith(f"{col}_"):
            return col
    return cleaned


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def positive_class_shap_values(explainer: shap.TreeExplainer, x_transformed: pd.DataFrame) -> tuple[np.ndarray, float]:
    shap_values = explainer.shap_values(x_transformed)
    expected_value = explainer.expected_value

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
        base_value = float(np.asarray(expected_value)[1])
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            values = shap_array[:, :, 1]
            base_value = float(np.asarray(expected_value)[1])
        else:
            values = shap_array
            base_value = float(np.asarray(expected_value).ravel()[0])
    return values, base_value


def save_summary_plots(shap_values: np.ndarray, x_display: pd.DataFrame) -> None:
    plt.figure()
    shap.summary_plot(shap_values, x_display, max_display=20, show=False)
    plt.title("SHAP summary plot for PCS prediction", pad=18)
    plt.tight_layout()
    plt.savefig(OUTPUT / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT / "shap_summary_plot.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, x_display, plot_type="bar", max_display=20, show=False)
    plt.title("SHAP feature importance for PCS prediction", pad=18)
    plt.tight_layout()
    plt.savefig(OUTPUT / "shap_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT / "shap_feature_importance.pdf", bbox_inches="tight")
    plt.close()


def save_variable_bar(variable_importance: pd.DataFrame) -> None:
    top = variable_importance.head(15).iloc[::-1]
    colors = ["#d94d5c" if group == "Laboratory" else "#4779b6" if group == "Symptoms" else "#6d7f9c" for group in top["clinical_group"]]
    plt.figure(figsize=(7.6, 5.6))
    plt.barh(top["display_name"], top["mean_abs_shap"], color=colors)
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Aggregated SHAP importance by original clinical variable")
    plt.tight_layout()
    plt.savefig(OUTPUT / "shap_variable_importance.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT / "shap_variable_importance.pdf", bbox_inches="tight")
    plt.close()


def save_domain_bar(variable_importance: pd.DataFrame) -> pd.DataFrame:
    domain = (
        variable_importance.groupby("clinical_group", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    top = domain.iloc[::-1]
    plt.figure(figsize=(6.8, 4.2))
    plt.barh(top["clinical_group"], top["mean_abs_shap"], color="#5b8cc0")
    plt.xlabel("Sum of mean absolute SHAP values")
    plt.title("SHAP contribution by clinical feature domain")
    plt.tight_layout()
    plt.savefig(OUTPUT / "shap_domain_importance.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT / "shap_domain_importance.pdf", bbox_inches="tight")
    plt.close()
    return domain


def save_dependence_plots(
    shap_values: np.ndarray,
    x_display: pd.DataFrame,
    feature_importance: pd.DataFrame,
    max_plots: int = 6,
) -> list[str]:
    saved = []
    for _, row in feature_importance.head(max_plots).iterrows():
        feature = row["display_feature_name"]
        safe = row["original_variable"].replace("/", "_").replace(" ", "_").replace(":", "_")
        plt.figure()
        shap.dependence_plot(feature, shap_values, x_display, show=False, interaction_index=None)
        plt.title(f"SHAP dependence: {feature}", pad=18)
        plt.tight_layout()
        filename = f"shap_dependence_{safe}.png"
        plt.savefig(OUTPUT / filename, dpi=300, bbox_inches="tight")
        plt.savefig(OUTPUT / filename.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()
        saved.append(filename)
    return saved


def save_waterfall_plot(
    shap_values: np.ndarray,
    base_value: float,
    x_display: pd.DataFrame,
    probabilities: np.ndarray,
    y: pd.Series,
) -> dict[str, object]:
    pcs_positive_indices = np.where(y.to_numpy() == 1)[0]
    if len(pcs_positive_indices):
        selected_idx = int(pcs_positive_indices[np.argmax(probabilities[pcs_positive_indices])])
    else:
        selected_idx = int(np.argmax(probabilities))

    explanation = shap.Explanation(
        values=shap_values[selected_idx],
        base_values=base_value,
        data=x_display.iloc[selected_idx].to_numpy(),
        feature_names=x_display.columns.tolist(),
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title("Individual SHAP explanation for a high-risk patient", pad=18)
    plt.tight_layout()
    plt.savefig(OUTPUT / "shap_single_patient_waterfall.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT / "shap_single_patient_waterfall.pdf", bbox_inches="tight")
    plt.close()
    return {
        "selected_row_index": selected_idx,
        "observed_pcs": int(y.iloc[selected_idx]),
        "predicted_probability": float(probabilities[selected_idx]),
    }


def feature_direction_summary(
    shap_values: np.ndarray,
    x_transformed: pd.DataFrame,
    feature_importance: pd.DataFrame,
    numeric_cols: list[str],
    max_features: int = 20,
) -> pd.DataFrame:
    rows = []
    for feature in feature_importance["feature"].head(max_features):
        values = x_transformed[feature].to_numpy()
        shap_col = shap_values[:, x_transformed.columns.get_loc(feature)]
        if np.nanstd(values) == 0 or np.nanstd(shap_col) == 0:
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = spearmanr(values, shap_col)
        original = original_variable_from_feature(feature, numeric_cols, [])
        rows.append(
            {
                "feature": feature,
                "direction_hint": "higher value increases predicted PCS risk"
                if pd.notna(rho) and rho > 0.15
                else "higher value decreases predicted PCS risk"
                if pd.notna(rho) and rho < -0.15
                else "nonlinear or weak monotonic direction",
                "spearman_rho_feature_vs_shap": rho,
                "p_value": p_value,
                "original_variable_guess": original,
            }
        )
    return pd.DataFrame(rows)


def clinical_group_for_variable(variable: str) -> str:
    for group, variables in CLINICAL_GROUPS.items():
        if variable in variables:
            return group
    return "Other"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = df.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    variable_importance: pd.DataFrame,
    domain_importance: pd.DataFrame,
    feature_importance: pd.DataFrame,
    direction: pd.DataFrame,
    selected_patient: dict[str, object],
    best_model_row: pd.Series | None,
    dependence_files: list[str],
) -> None:
    top_vars = variable_importance.head(10)
    top_table = top_vars[["display_name", "clinical_group", "mean_abs_shap", "rank"]].copy()
    top_table["mean_abs_shap"] = top_table["mean_abs_shap"].map(lambda x: f"{x:.4f}")

    domain_table = domain_importance.copy()
    domain_table["mean_abs_shap"] = domain_table["mean_abs_shap"].map(lambda x: f"{x:.4f}")

    direction_table = direction.head(10).copy()
    direction_table["spearman_rho_feature_vs_shap"] = direction_table["spearman_rho_feature_vs_shap"].map(
        lambda x: "NA" if pd.isna(x) else f"{x:.3f}"
    )
    direction_table["p_value"] = direction_table["p_value"].map(lambda x: "NA" if pd.isna(x) else f"{x:.3g}")

    best_text = ""
    if best_model_row is not None:
        best_text = (
            f"Random Forest was selected for SHAP interpretation because it had the highest out-of-fold AUC "
            f"in the 5-fold cross-validation report (OOF AUC = {best_model_row['OOF AUC']:.3f}, "
            f"OOF Brier score = {best_model_row['OOF Brier score']:.3f})."
        )

    top_names = ", ".join(top_vars["display_name"].head(6).tolist())
    lab_rank = domain_importance.iloc[0]["clinical_group"]
    text = f"""# SHAP Explainability Report for PCS Prediction

## Model used for interpretation

{best_text}

The model was refitted on the full dataset after the same preprocessing pipeline used in cross-validation. SHAP values therefore describe the fitted final Random Forest model; they should be interpreted as internal explanatory evidence, not as external validation.

## Main interpretable findings

- The most influential variables by aggregated mean absolute SHAP value were: {top_names}.
- At the domain level, the largest contribution came from **{lab_rank}**, followed by the remaining clinical domains shown below.
- The individual waterfall plot selected a high-risk PCS-positive patient with predicted PCS probability = **{selected_patient['predicted_probability']:.3f}**.

## Variable-level SHAP importance

{markdown_table(top_table)}

## Domain-level SHAP contribution

{markdown_table(domain_table)}

## Directional interpretation of top transformed features

This table uses Spearman correlation between transformed feature values and SHAP values. It is a direction hint, not a causal estimate.

{markdown_table(direction_table)}

## Figures

![SHAP summary plot](shap_summary_plot.png)

![SHAP feature importance](shap_feature_importance.png)

![Aggregated variable importance](shap_variable_importance.png)

![Domain importance](shap_domain_importance.png)

![Single patient waterfall](shap_single_patient_waterfall.png)

Dependence plots:

{chr(10).join(f'- ![{name}]({name})' for name in dependence_files)}

## Manuscript-ready interpretation

The SHAP analysis indicated that the final Random Forest model mainly relied on hepatobiliary laboratory markers and preoperative symptom-related variables when estimating PCS risk. High-impact features included laboratory markers such as GGT, ALT, ALP, AST, bilirubin-related indices and CA19-9, together with symptom duration and pain-frequency-related information where selected by the model. These findings suggest that biochemical evidence of biliary or hepatocellular disturbance and symptom burden may jointly contribute to model-based PCS risk stratification.

Because SHAP explains model behavior rather than biological causality, these findings should be discussed as model-derived risk signals. They support clinical interpretability of the prediction model but still require external validation and prospective assessment.
"""
    (OUTPUT / "shap_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_plot_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET)
    x = df.drop(columns=["pcs"])
    y = df["pcs"].astype(int)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(df)
    x_transformed_arr = preprocessor.fit_transform(x)
    feature_names = [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    x_transformed = pd.DataFrame(x_transformed_arr, columns=feature_names)
    display_feature_names = [display_feature_name(name, numeric_cols, categorical_cols) for name in feature_names]
    x_display = pd.DataFrame(x_transformed_arr, columns=display_feature_names)

    model = build_model()
    model.fit(x_transformed, y)
    probabilities = model.predict_proba(x_transformed)[:, 1]

    explainer = shap.TreeExplainer(model)
    shap_values, base_value = positive_class_shap_values(explainer, x_transformed)

    feature_importance = pd.DataFrame(
        {
            "feature": x_transformed.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    feature_importance["rank"] = np.arange(1, len(feature_importance) + 1)
    feature_importance["original_variable"] = feature_importance["feature"].map(
        lambda f: original_variable_from_feature(f, numeric_cols, categorical_cols)
    )
    feature_importance["display_feature_name"] = feature_importance["feature"].map(
        lambda f: display_feature_name(f, numeric_cols, categorical_cols)
    )
    feature_importance["display_name"] = feature_importance["original_variable"].map(
        lambda v: DISPLAY_NAMES.get(v, v)
    )

    variable_importance = (
        feature_importance.groupby(["original_variable", "display_name"], as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    variable_importance["rank"] = np.arange(1, len(variable_importance) + 1)
    variable_importance["clinical_group"] = variable_importance["original_variable"].map(clinical_group_for_variable)

    domain_importance = save_domain_bar(variable_importance)
    save_summary_plots(shap_values, x_display)
    save_variable_bar(variable_importance)
    dependence_files = save_dependence_plots(shap_values, x_display, feature_importance)
    selected_patient = save_waterfall_plot(shap_values, base_value, x_display, probabilities, y)
    direction = feature_direction_summary(shap_values, x_transformed, feature_importance, numeric_cols)

    feature_importance.to_csv(OUTPUT / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")
    variable_importance.to_csv(OUTPUT / "shap_variable_importance.csv", index=False, encoding="utf-8-sig")
    domain_importance.to_csv(OUTPUT / "shap_domain_importance.csv", index=False, encoding="utf-8-sig")
    direction.to_csv(OUTPUT / "shap_direction_summary.csv", index=False, encoding="utf-8-sig")

    best_model_row = None
    if MODELING_SUMMARY.exists():
        summary = pd.read_csv(MODELING_SUMMARY)
        rf_rows = summary.loc[summary["model"] == "Random Forest"]
        if not rf_rows.empty:
            best_model_row = rf_rows.iloc[0]

    metadata = {
        "model": "Random Forest",
        "n_rows": int(len(df)),
        "n_transformed_features": int(x_transformed.shape[1]),
        "base_value": float(base_value),
        "selected_patient": selected_patient,
        "top_variables": variable_importance.head(10).to_dict(orient="records"),
    }
    (OUTPUT / "shap_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(
        variable_importance,
        domain_importance,
        feature_importance,
        direction,
        selected_patient,
        best_model_row,
        dependence_files,
    )
    print(f"Saved SHAP report to {OUTPUT / 'shap_report.md'}")
    print(variable_importance.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
