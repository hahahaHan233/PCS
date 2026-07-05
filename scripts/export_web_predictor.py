from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from train_ml_models import DATASET, OUTPUT, RANDOM_STATE, build_models, build_preprocessor


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SCHEMA = ROOT / "output" / "dataset_schema_mapping.csv"
MODEL_JS = DOCS / "model.js"


VALUE_LABELS = {
    "0": "否",
    "1": "是",
    "male": "男",
    "female": "女",
    "mental_labor": "脑力劳动",
    "manual_labor": "体力劳动",
    "unemployed": "无业/未就业",
    "junior_high_or_below": "初中及以下",
    "high_school": "高中/中专",
    "college_or_above": "大专及以上",
    "none": "无",
    "very_short": "很短",
    "short": "短",
    "medium": "中等",
    "long": "长",
    "daily": "每日",
    "weekly": "每周",
    "monthly": "每月",
    "occasional": "偶发",
    "single": "单发",
    "multiple": "多发",
    "no_stone": "未见结石",
    "gallbladder_body": "胆囊体部",
    "gallbladder_fundus": "胆囊底部",
    "gallbladder_neck": "胆囊颈部",
    "cystic_duct": "胆囊管",
    "left_intrahepatic_bile_duct": "左肝内胆管",
    "distal_common_bile_duct": "胆总管远端",
}


def as_jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def export_tree(estimator) -> dict[str, object]:
    tree = estimator.tree_
    values = tree.value[:, 0, :]
    totals = values.sum(axis=1)
    positive_probability = np.divide(
        values[:, 1],
        totals,
        out=np.zeros_like(totals, dtype=float),
        where=totals != 0,
    )
    return {
        "childrenLeft": tree.children_left.astype(int).tolist(),
        "childrenRight": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "probability": positive_probability.astype(float).tolist(),
    }


def build_field_metadata(df: pd.DataFrame) -> list[dict[str, object]]:
    schema = pd.read_csv(SCHEMA)
    labels = {
        row["english_name"]: row["original_name"]
        for _, row in schema.iterrows()
        if row.get("included_in_dataset_csv") is True or str(row.get("included_in_dataset_csv")).lower() == "true"
    }

    fields: list[dict[str, object]] = []
    for column in df.drop(columns=["pcs"]).columns:
        series = df[column]
        non_null = series.dropna()
        is_numeric = pd.api.types.is_numeric_dtype(series)
        unique_values = sorted(non_null.unique().tolist()) if len(non_null) else []
        is_binary = is_numeric and set(unique_values).issubset({0, 1})

        if is_numeric and not is_binary:
            quantiles = non_null.quantile([0.01, 0.5, 0.99]) if len(non_null) else pd.Series([0, 0, 0])
            fields.append(
                {
                    "name": column,
                    "label": labels.get(column, column),
                    "kind": "number",
                    "default": as_jsonable(quantiles.loc[0.5]),
                    "min": as_jsonable(quantiles.loc[0.01]),
                    "max": as_jsonable(quantiles.loc[0.99]),
                }
            )
        else:
            options = []
            for value in unique_values:
                key = str(as_jsonable(value))
                options.append({"value": as_jsonable(value), "label": VALUE_LABELS.get(key, key)})
            fields.append(
                {
                    "name": column,
                    "label": labels.get(column, column),
                    "kind": "select",
                    "default": as_jsonable(non_null.mode().iloc[0]) if len(non_null) else None,
                    "options": options,
                }
            )
    return fields


def main() -> None:
    df = pd.read_csv(DATASET)
    x = df.drop(columns=["pcs"])
    y = df["pcs"].astype(int)

    preprocessor = build_preprocessor(df)
    model = build_models(y)["Random Forest"]
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(x, y)

    fitted_preprocessor = pipeline.named_steps["preprocess"]
    numeric_cols = list(fitted_preprocessor.transformers_[0][2])
    categorical_cols = list(fitted_preprocessor.transformers_[1][2])
    numeric_pipeline = fitted_preprocessor.named_transformers_["numeric"]
    categorical_pipeline = fitted_preprocessor.named_transformers_["categorical"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    one_hot = categorical_pipeline.named_steps["onehot"]

    artifact = {
        "modelType": "RandomForestClassifier",
        "createdBy": "scripts/export_web_predictor.py",
        "randomState": RANDOM_STATE,
        "trainedRows": int(len(df)),
        "positiveLabel": "PCS",
        "threshold": 0.5,
        "performance": {
            "oofAuc": 0.917,
            "oofAveragePrecision": 0.842,
            "oofBrier": 0.088,
            "validation": "5-fold stratified cross-validation from output/modeling/model_report.md",
        },
        "preprocessing": {
            "numeric": {
                "columns": numeric_cols,
                "median": numeric_pipeline.named_steps["imputer"].statistics_.astype(float).tolist(),
                "mean": numeric_pipeline.named_steps["scaler"].mean_.astype(float).tolist(),
                "scale": numeric_pipeline.named_steps["scaler"].scale_.astype(float).tolist(),
            },
            "categorical": {
                "columns": categorical_cols,
                "fill": [as_jsonable(value) for value in categorical_imputer.statistics_.tolist()],
                "categories": [[as_jsonable(value) for value in category.tolist()] for category in one_hot.categories_],
            },
        },
        "fields": build_field_metadata(df),
        "forest": {
            "nEstimators": len(pipeline.named_steps["model"].estimators_),
            "trees": [export_tree(estimator) for estimator in pipeline.named_steps["model"].estimators_],
        },
    }

    DOCS.mkdir(parents=True, exist_ok=True)
    js = "window.PCS_MODEL = "
    js += json.dumps(artifact, ensure_ascii=False, separators=(",", ":"))
    js += ";\n"
    MODEL_JS.write_text(js, encoding="utf-8")
    print(f"Saved browser model to {MODEL_JS}")


if __name__ == "__main__":
    main()
