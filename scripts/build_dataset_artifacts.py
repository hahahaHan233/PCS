from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "数据（终）.xlsx"
DATASET_CSV = ROOT / "dataset.csv"
OUTPUT = ROOT / "output"
FIG_DIR = OUTPUT / "dataset_insights"

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
    "gallbladder_wall_thickening": "Gallbladder wall thickening",
    "fatty_liver": "Fatty liver",
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
    "pcs": "PCS",
}


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def display_level(level: object) -> str:
    if pd.isna(level):
        return "Missing"
    if isinstance(level, (int, float, np.integer, np.floating)) and float(level) in (0.0, 1.0):
        return "Yes" if int(level) == 1 else "No"
    return str(level).replace("_", " ")


SCHEMA = {
    "姓名": {"en": "patient_name", "role": "identifier", "include_model": False, "type": "identifier"},
    "住院号": {"en": "hospital_id", "role": "identifier", "include_model": False, "type": "identifier"},
    "年龄": {"en": "age", "role": "predictor", "include_model": True, "type": "numeric"},
    "性别": {"en": "sex", "role": "predictor", "include_model": True, "type": "categorical"},
    "职业": {"en": "occupation", "role": "predictor", "include_model": True, "type": "categorical"},
    "身高": {"en": "height_cm", "role": "predictor", "include_model": True, "type": "numeric"},
    "体重": {"en": "weight_kg", "role": "predictor", "include_model": True, "type": "numeric"},
    "BMI": {"en": "bmi", "role": "predictor", "include_model": True, "type": "numeric"},
    "教育水平": {"en": "education_level", "role": "predictor", "include_model": True, "type": "categorical"},
    "是否吸烟": {"en": "smoking", "role": "predictor", "include_model": True, "type": "binary"},
    "是否饮酒": {"en": "alcohol_use", "role": "predictor", "include_model": True, "type": "binary"},
    "症状持续时间": {"en": "symptom_duration", "role": "predictor", "include_model": True, "type": "ordinal_categorical"},
    "有否腹痛": {"en": "abdominal_pain", "role": "predictor", "include_model": True, "type": "binary"},
    "疼痛频率": {"en": "pain_frequency", "role": "predictor", "include_model": True, "type": "ordinal_categorical"},
    "是否有放射痛": {"en": "radiating_pain", "role": "predictor", "include_model": True, "type": "binary"},
    "是否与进食有关": {"en": "meal_related_pain", "role": "predictor", "include_model": True, "type": "binary"},
    "是否合并高血压": {"en": "hypertension", "role": "predictor", "include_model": True, "type": "binary"},
    "是否合并高血脂": {"en": "hyperlipidemia", "role": "predictor", "include_model": True, "type": "binary"},
    "是否合并糖尿病": {"en": "diabetes", "role": "predictor", "include_model": True, "type": "binary"},
    "是否焦虑/抑郁": {"en": "anxiety_depression", "role": "predictor", "include_model": True, "type": "binary"},
    "是否既往腹部手术史": {"en": "prior_abdominal_surgery", "role": "predictor", "include_model": True, "type": "binary"},
    "胆囊壁是否增厚": {"en": "gallbladder_wall_thickening", "role": "predictor", "include_model": True, "type": "binary"},
    "最大结石直径": {"en": "max_stone_diameter_mm", "role": "predictor", "include_model": True, "type": "numeric"},
    "结石数量": {"en": "stone_number", "role": "predictor", "include_model": True, "type": "categorical"},
    "结石位置": {"en": "stone_location", "role": "predictor", "include_model": True, "type": "categorical"},
    "胆总管直径": {"en": "common_bile_duct_diameter_mm", "role": "predictor", "include_model": True, "type": "numeric"},
    "是否有脂肪肝": {"en": "fatty_liver", "role": "predictor", "include_model": True, "type": "binary"},
    "胆囊是否萎缩": {"en": "gallbladder_atrophy", "role": "predictor", "include_model": True, "type": "binary"},
    "谷丙转氨酶": {"en": "alt", "role": "predictor", "include_model": True, "type": "numeric"},
    "谷草转氨酶": {"en": "ast", "role": "predictor", "include_model": True, "type": "numeric"},
    "碱性磷酸酶": {"en": "alp", "role": "predictor", "include_model": True, "type": "numeric"},
    "谷氨酰转肽酶": {"en": "ggt", "role": "predictor", "include_model": True, "type": "numeric"},
    "总胆红素": {"en": "total_bilirubin", "role": "predictor", "include_model": True, "type": "numeric"},
    "总胆汁酸": {"en": "total_bile_acid", "role": "predictor", "include_model": True, "type": "numeric"},
    "总胆固醇": {"en": "total_cholesterol", "role": "predictor", "include_model": True, "type": "numeric"},
    "甘油三酯": {"en": "triglyceride", "role": "predictor", "include_model": True, "type": "numeric"},
    "甲胎蛋白": {"en": "alpha_fetoprotein", "role": "predictor", "include_model": True, "type": "numeric"},
    "CA199": {"en": "ca199", "role": "predictor", "include_model": True, "type": "numeric"},
    "是否发生PCS": {"en": "pcs", "role": "outcome", "include_model": True, "type": "binary"},
    "PCS发生时间": {"en": "pcs_onset_time_days", "role": "post_outcome", "include_model": False, "type": "numeric"},
    "PCS症状类型": {"en": "pcs_symptom_type", "role": "post_outcome", "include_model": False, "type": "categorical"},
}


VALUE_MAPS = {
    "sex": {"男": "male", "女": "female"},
    "occupation": {"脑力劳动": "mental_labor", "体力劳动": "manual_labor", "无业": "unemployed", "退休": "retired"},
    "education_level": {
        "初中及以下": "junior_high_or_below",
        "高中": "high_school",
        "高中/中专": "high_school_or_technical",
        "大学及以上": "college_or_above",
        "大专及以上": "college_or_above",
    },
    "symptom_duration": {"无": "none", "极短期": "very_short", "短期": "short", "中期": "medium", "长期": "long"},
    "pain_frequency": {"无": "none", "每日": "daily", "每周": "weekly", "没周": "weekly", "每月": "monthly", "偶发": "occasional", "偶尔": "occasional"},
    "stone_number": {"无结石": "no_stone", "单发": "single", "多发": "multiple"},
    "stone_location": {
        "无": "none",
        "胆囊底部": "gallbladder_fundus",
        "胆囊体部": "gallbladder_body",
        "胆囊颈部": "gallbladder_neck",
        "胆囊管": "cystic_duct",
        "胆总管下段": "distal_common_bile_duct",
        "左肝内胆管": "left_intrahepatic_bile_duct",
        "多部位": "multiple_sites",
    },
}

BINARY_VALUES = {"是": 1, "否": 0, "有": 1, "无": 0, "男": 1, "女": 0}


def flatten_headers(raw: pd.DataFrame) -> pd.DataFrame:
    headers = list(raw.iloc[1])
    headers[0] = "姓名"
    headers[1] = "住院号"
    data = raw.iloc[2:].copy()
    data.columns = headers
    data = data.dropna(how="all")
    return data


def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        info = next((v for v in SCHEMA.values() if v["en"] == col), None)
        if info and info["type"] == "binary":
            out[col] = out[col].map(BINARY_VALUES).astype("Int64")
        elif col in VALUE_MAPS:
            out[col] = out[col].map(VALUE_MAPS[col]).fillna(out[col])
        elif info and info["type"] == "numeric":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def save_schema_mapping(original_columns: list[str]) -> pd.DataFrame:
    rows = []
    for idx, cn in enumerate(original_columns, start=1):
        info = SCHEMA[cn]
        rows.append(
            {
                "source_order": idx,
                "original_name": cn,
                "english_name": info["en"],
                "type": info["type"],
                "role": info["role"],
                "included_in_dataset_csv": info["include_model"],
            }
        )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(OUTPUT / "dataset_schema_mapping.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "dataset_schema_mapping.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return mapping


def numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "variable": col,
                "n": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "mean": s.mean(),
                "sd": s.std(),
                "median": s.median(),
                "q1": s.quantile(0.25),
                "q3": s.quantile(0.75),
                "min": s.min(),
                "max": s.max(),
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cat_cols:
        counts = df[col].value_counts(dropna=False)
        for value, count in counts.items():
            label = "missing" if pd.isna(value) else value
            rows.append({"variable": col, "level": label, "n": int(count), "percent": count / len(df) * 100})
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(name: str) -> None:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def make_figures(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    configure_plotting()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    outcome = df["pcs"].map({0: "No PCS", 1: "PCS"})
    plt.figure(figsize=(4.2, 3.4))
    ax = sns.countplot(x=outcome, order=["No PCS", "PCS"], palette=["#4C78A8", "#E45756"])
    total = len(df)
    for p in ax.patches:
        n = int(p.get_height())
        ax.annotate(f"{n}\n({n / total:.1%})", (p.get_x() + p.get_width() / 2, n), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Number of patients")
    ax.set_title("Outcome distribution")
    savefig("fig1_outcome_distribution.png")

    missing = df.isna().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0]
    missing.index = [display_name(col) for col in missing.index]
    plt.figure(figsize=(7.2, max(3.2, 0.25 * len(missing))))
    ax = sns.barplot(x=missing.values, y=missing.index, color="#72B7B2")
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("")
    ax.set_title("Variables with missing values")
    savefig("fig2_missingness_bar.png")

    demo = df[["pcs", "age", "bmi"]].melt(id_vars="pcs", var_name="variable", value_name="value")
    demo["variable"] = demo["variable"].map(display_name)
    demo["PCS status"] = demo["pcs"].map({0: "No PCS", 1: "PCS"})
    plt.figure(figsize=(6.4, 3.6))
    ax = sns.boxplot(data=demo, x="variable", y="value", hue="PCS status", palette=["#4C78A8", "#E45756"], showfliers=False)
    sns.stripplot(data=demo, x="variable", y="value", hue="PCS status", dodge=True, alpha=0.22, size=2, palette=["#4C78A8", "#E45756"], ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.set_title("Demographic features by PCS status")
    savefig("fig3_demographics_by_pcs.png")

    lab_cols = ["alt", "ast", "alp", "ggt", "total_bilirubin", "total_bile_acid", "ca199"]
    labs = df[["pcs"] + lab_cols].melt(id_vars="pcs", var_name="marker", value_name="value").dropna()
    labs["marker"] = labs["marker"].map(display_name)
    labs["PCS status"] = labs["pcs"].map({0: "No PCS", 1: "PCS"})
    labs["log1p_value"] = np.log1p(labs["value"].clip(lower=0))
    plt.figure(figsize=(9.2, 4.2))
    ax = sns.boxplot(data=labs, x="marker", y="log1p_value", hue="PCS status", palette=["#4C78A8", "#E45756"], showfliers=False)
    ax.set_xlabel("")
    ax.set_ylabel("log(1 + value)")
    ax.set_title("Preoperative laboratory markers by PCS status")
    ax.legend(title="")
    savefig("fig4_lab_markers_by_pcs.png")

    rate_vars = ["sex", "smoking", "alcohol_use", "abdominal_pain", "radiating_pain", "gallbladder_wall_thickening", "fatty_liver"]
    rows = []
    for var in rate_vars:
        temp = df.groupby(var, dropna=False)["pcs"].agg(["mean", "count"]).reset_index()
        for _, row in temp.iterrows():
            if row["count"] >= 5:
                rows.append({"variable": display_name(var), "level": display_level(row[var]), "pcs_rate": row["mean"] * 100, "n": int(row["count"])})
    rates = pd.DataFrame(rows)
    rates["label"] = rates["variable"] + ": " + rates["level"] + " (n=" + rates["n"].astype(str) + ")"
    rates = rates.sort_values("pcs_rate", ascending=True)
    plt.figure(figsize=(8.2, max(4.2, 0.26 * len(rates))))
    ax = sns.barplot(data=rates, x="pcs_rate", y="label", color="#F58518")
    ax.set_xlabel("Observed PCS rate (%)")
    ax.set_ylabel("")
    ax.set_title("PCS rate across selected categorical strata")
    savefig("fig5_categorical_pcs_rates.png")

    corr_cols = [c for c in numeric_cols if df[c].notna().sum() >= 50] + ["pcs"]
    corr = df[corr_cols].corr(method="spearman")
    corr = corr.rename(index=display_name, columns=display_name)
    plt.figure(figsize=(9.5, 7.2))
    ax = sns.heatmap(corr, cmap="vlag", center=0, square=False, linewidths=0.25, cbar_kws={"label": "Spearman r"})
    ax.set_title("Spearman correlation among numeric variables")
    savefig("fig6_numeric_correlation_heatmap.png")


def make_latex(df: pd.DataFrame, mapping: pd.DataFrame) -> None:
    n = len(df)
    pcs_n = int(df["pcs"].sum())
    no_pcs_n = n - pcs_n
    pcs_rate = pcs_n / n * 100
    age = df["age"].agg(["mean", "std"])
    bmi = df["bmi"].agg(["mean", "std"])
    female_n = int((df["sex"] == "female").sum())
    female_rate = female_n / n * 100
    predictors = int((mapping["included_in_dataset_csv"] & (mapping["role"] == "predictor")).sum())
    numeric_predictors = int((mapping["included_in_dataset_csv"] & (mapping["role"] == "predictor") & (mapping["type"] == "numeric")).sum())
    binary_predictors = int((mapping["included_in_dataset_csv"] & (mapping["role"] == "predictor") & (mapping["type"] == "binary")).sum())
    cat_predictors = predictors - numeric_predictors - binary_predictors
    missing_cols = int((df.isna().sum() > 0).sum())

    latex = rf"""\subsection{{Dataset characteristics}}

The final analytic dataset was derived from a single de-identified spreadsheet and contained {n} patients undergoing cholecystectomy-related assessment. Direct identifiers, including patient name and hospital admission number, were removed before analysis. Variables recorded after outcome ascertainment, including PCS onset time and PCS symptom type, were excluded from the modelling dataset to reduce outcome leakage. The resulting dataset included {predictors} candidate predictors: {numeric_predictors} continuous predictors, {binary_predictors} binary predictors, and {cat_predictors} multi-level categorical or ordinal predictors. The binary outcome was post-cholecystectomy syndrome (PCS).

PCS occurred in {pcs_n} of {n} patients ({pcs_rate:.1f}\%), whereas {no_pcs_n} patients ({100 - pcs_rate:.1f}\%) did not develop PCS. The mean age was {age['mean']:.1f} $\pm$ {age['std']:.1f} years and the mean body mass index was {bmi['mean']:.1f} $\pm$ {bmi['std']:.1f} kg/m$^2$. Female patients accounted for {female_n} cases ({female_rate:.1f}\%). Missingness was present in {missing_cols} variables and was handled within the cross-validation pipeline to avoid information leakage.

\begin{{figure}}[htbp]
    \centering
    \includegraphics[width=0.48\textwidth]{{output/dataset_insights/fig1_outcome_distribution.png}}
    \includegraphics[width=0.48\textwidth]{{output/dataset_insights/fig2_missingness_bar.png}}
    \caption{{Outcome distribution and missingness profile of the de-identified PCS dataset.}}
    \label{{fig:pcs_dataset_quality}}
\end{{figure}}

\begin{{figure}}[htbp]
    \centering
    \includegraphics[width=0.48\textwidth]{{output/dataset_insights/fig3_demographics_by_pcs.png}}
    \includegraphics[width=0.48\textwidth]{{output/dataset_insights/fig4_lab_markers_by_pcs.png}}
    \caption{{Distribution of demographic and preoperative laboratory features stratified by PCS status. Laboratory measurements are displayed on a log-transformed scale to reduce the influence of extreme values.}}
    \label{{fig:pcs_feature_distributions}}
\end{{figure}}

\begin{{figure}}[htbp]
    \centering
    \includegraphics[width=0.48\textwidth]{{output/dataset_insights/fig5_categorical_pcs_rates.png}}
    \includegraphics[width=0.48\textwidth]{{output/dataset_insights/fig6_numeric_correlation_heatmap.png}}
    \caption{{Observed PCS rates across selected categorical strata and Spearman correlation structure among continuous variables.}}
    \label{{fig:pcs_exploratory_analysis}}
\end{{figure}}
"""
    (OUTPUT / "dataset_distribution_latex.tex").write_text(latex, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    raw = pd.read_excel(SOURCE, header=None)
    data_cn = flatten_headers(raw)
    original_columns = list(data_cn.columns)
    missing_schema = sorted(set(original_columns) - set(SCHEMA))
    if missing_schema:
        raise ValueError(f"Missing schema definitions: {missing_schema}")

    mapping = save_schema_mapping(original_columns)
    rename_map = {cn: info["en"] for cn, info in SCHEMA.items()}
    data_en = data_cn.rename(columns=rename_map)
    data_en = normalize_values(data_en)

    included_cols = [SCHEMA[cn]["en"] for cn in original_columns if SCHEMA[cn]["include_model"]]
    dataset = data_en[included_cols].copy()
    dataset.to_csv(DATASET_CSV, index=False, encoding="utf-8-sig")

    numeric_cols = [info["en"] for info in SCHEMA.values() if info["include_model"] and info["type"] == "numeric"]
    cat_cols = [info["en"] for info in SCHEMA.values() if info["include_model"] and info["type"] != "numeric"]
    numeric_summary(dataset, numeric_cols).to_csv(OUTPUT / "dataset_numeric_summary.csv", index=False, encoding="utf-8-sig")
    categorical_summary(dataset, cat_cols).to_csv(OUTPUT / "dataset_categorical_summary.csv", index=False, encoding="utf-8-sig")

    make_figures(dataset, numeric_cols)
    make_latex(dataset, mapping)

    report = {
        "source_file": str(SOURCE),
        "dataset_csv": str(DATASET_CSV),
        "n_rows": int(dataset.shape[0]),
        "n_columns": int(dataset.shape[1]),
        "pcs_positive": int(dataset["pcs"].sum()),
        "pcs_negative": int((dataset["pcs"] == 0).sum()),
        "dropped_columns": ["patient_name", "hospital_id", "pcs_onset_time_days", "pcs_symptom_type"],
        "figure_directory": str(FIG_DIR),
    }
    (OUTPUT / "dataset_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
