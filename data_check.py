# -*- coding: utf-8 -*-
"""数据特征检查脚本：读入Excel，列名中英映射，分布/缺失/异常值检查，目标：是否发生PCS"""

import pandas as pd
import numpy as np

# ========== 1. 读入数据 ==========
df = pd.read_excel('./data/数据2.12.xlsx', header=1)

# ========== 2. 列名中英映射 ==========
COL_RENAME = {
    'Unnamed: 0': '姓名',
    'Unnamed: 1': '住院号',
    '年龄': 'age',
    '性别': 'gender',
    '职业': 'occupation',
    '身高': 'height',
    '体重': 'weight',
    'BMI': 'bmi',
    '教育水平': 'education',
    '是否吸烟': 'smoking',
    '是否饮酒': 'drinking',
    '症状持续时间': 'symptom_duration',
    '主要症状': 'main_symptom',
    '疼痛频率': 'pain_frequency',
    '是否有放射痛': 'radiating_pain',
    '是否与进食有关': 'food_related',
    '是否合并高血压': 'hypertension',
    '是否合并高血脂': 'hyperlipidemia',
    '是否合并糖尿病': 'diabetes',
    '是否焦虑/抑郁': 'anxiety_depression',
    '是否既往腹部手术史': 'prior_abdominal_surgery',
    '胆囊壁是否增厚': 'gallbladder_wall_thickening',
    '最大结石直径': 'max_stone_diameter',
    '结石数量': 'stone_count',
    '结石位置': 'stone_location',
    '胆总管直径': 'common_bile_duct_diameter',
    '术后胆总管': 'postoperative_cbd',
    '是否有脂肪肝': 'fatty_liver',
    '胆囊是否萎缩': 'gallbladder_atrophy',
    '手术时间': 'surgery_duration',
    '预估出血量': 'estimated_blood_loss',
    '是否术中并发症': 'intraoperative_complications',
    '术后病理诊断': 'postoperative_pathology',
    '谷丙转氨酶': 'ALT',
    '谷草转氨酶': 'AST',
    '碱性磷酸酶': 'ALP',
    '谷氨酰转肽酶': 'GGT',
    '总胆红素': 'total_bilirubin',
    '总胆汁酸': 'total_bile_acid',
    '血淀粉酶': 'blood_amylase',
    '总胆固醇': 'total_cholesterol',
    '甘油三酯': 'triglyceride',
    '甲胎蛋白': 'AFP',
    'CA199': 'CA199',
    '是否发生PCS': 'PCS_occurred',   # 目标变量
    'PCS发生时间': 'PCS_time',
    'PCS症状类型': 'PCS_symptom_type',
}
df = df.rename(columns=COL_RENAME)

# ========== 3. 基本概况 ==========
print("=" * 50, "数据概况", "=" * 50)
print(f"行数: {len(df)}, 列数: {len(df.columns)}")
print(df.dtypes)
print()

# ========== 4. 缺失值 ==========
print("=" * 50, "缺失值", "=" * 50)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"缺失数": missing, "缺失率%": missing_pct})
print(missing_df[missing_df["缺失数"] > 0].sort_values("缺失数", ascending=False))
if missing.sum() == 0:
    print("无缺失值")
print()

# ========== 5. 目标变量分布（是否发生PCS） ==========
print("=" * 50, "目标变量 PCS_occurred 分布", "=" * 50)
if "PCS_occurred" in df.columns:
    print(df["PCS_occurred"].value_counts(dropna=False))
    print(df["PCS_occurred"].value_counts(normalize=True, dropna=False).round(4))
print()

# ========== 6. 数值列分布与异常值（IQR法） ==========
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("=" * 50, "数值列 分布与异常值(IQR)", "=" * 50)
for col in numeric_cols[:20]:  # 先展示前20列，可去掉切片看全部
    s = df[col].dropna()
    if len(s) == 0:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    print(f"{col}: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}, 异常数={n_out}")
if len(numeric_cols) > 20:
    print(f"... 共 {len(numeric_cols)} 个数值列，仅展示前20列")
print()

# ========== 7. 分类/对象列取值分布（抽样） ==========
obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
print("=" * 50, "分类列 取值数量（前15列）", "=" * 50)
for col in obj_cols[:15]:
    n_unique = df[col].nunique()
    print(f"{col}: {n_unique} 个不同取值")
if len(obj_cols) > 15:
    print(f"... 共 {len(obj_cols)} 个非数值列")
print()

# ========== 8. 简要汇总 ==========
print("=" * 50, "简要汇总", "=" * 50)
print("列名已按字典重命名为英文；目标变量: PCS_occurred")
print("数值列:", len(numeric_cols), "| 非数值列:", len(obj_cols))
