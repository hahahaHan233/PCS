# -*- coding: utf-8 -*-
"""
相关性分析与降维可视化脚本
- 特征与「是否发生PCS」的关联性可视化
- 特征标准化 + PCA / t-SNE 2D 降维图（按是否发生PCS着色）
"""
# 避免 Windows 下 joblib/loky 检测物理核心数时的警告（系统找不到指定文件）
import os
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 中文字体与负号
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'SimHei', 'Microsoft YaHei', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 读入数据与列名映射 ==========
DATA_PATH = Path(__file__).resolve().parent / 'data' / '数据4.16.xlsx'
COL_RENAME = {
    'Unnamed: 0': '姓名', 'Unnamed: 1': '住院号', '年龄': 'age', '性别': 'gender',
    '职业': 'occupation', '身高': 'height', '体重': 'weight', 'BMI': 'bmi',
    '教育水平': 'education', '是否吸烟': 'smoking', '是否饮酒': 'drinking',
    '症状持续时间': 'symptom_duration', '主要症状': 'main_symptom', '疼痛频率': 'pain_frequency',
    '是否有放射痛': 'radiating_pain', '是否与进食有关': 'food_related',
    '是否合并高血压': 'hypertension', '是否合并高血脂': 'hyperlipidemia', '是否合并糖尿病': 'diabetes',
    '是否焦虑/抑郁': 'anxiety_depression', '是否既往腹部手术史': 'prior_abdominal_surgery',
    '胆囊壁是否增厚': 'gallbladder_wall_thickening', '最大结石直径': 'max_stone_diameter',
    '结石数量': 'stone_count', '结石位置': 'stone_location', '胆总管直径': 'common_bile_duct_diameter',
    '术后胆总管': 'postoperative_cbd', '是否有脂肪肝': 'fatty_liver', '胆囊是否萎缩': 'gallbladder_atrophy',
    '手术时间': 'surgery_duration', '预估出血量': 'estimated_blood_loss',
    '是否术中并发症': 'intraoperative_complications', '术后病理诊断': 'postoperative_pathology',
    '谷丙转氨酶': 'ALT', '谷草转氨酶': 'AST', '碱性磷酸酶': 'ALP', '谷氨酰转肽酶': 'GGT',
    '总胆红素': 'total_bilirubin', '总胆汁酸': 'total_bile_acid', '血淀粉酶': 'blood_amylase',
    '总胆固醇': 'total_cholesterol', '甘油三酯': 'triglyceride', '甲胎蛋白': 'AFP', 'CA199': 'CA199',
    '是否发生PCS': 'PCS_occurred', 'PCS发生时间': 'PCS_time', 'PCS症状类型': 'PCS_symptom_type',
}
df = pd.read_excel(DATA_PATH, header=1).rename(columns=COL_RENAME)

# ========== 2. 目标变量编码（是否发生PCS -> 0/1） ==========
target_col = 'PCS_occurred'
if target_col not in df.columns:
    raise KeyError(f"未找到列: {target_col}")
# 统一为数值：是/1/True -> 1，否/0/False -> 0
y_raw = df[target_col].astype(str).str.strip().str.lower()
y = np.where(y_raw.isin(['1', '是', 'yes', 'true', '发生']), 1, 0)
df['PCS_label'] = y
labels = df['PCS_label']

# ========== 3. 特征列：仅数值型，排除标识与目标相关 ==========
exclude = {'姓名', '住院号', 'PCS_occurred', 'PCS_label', 'PCS_time', 'PCS_symptom_type'}
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
# 若数值列不足，尝试将部分分类列转为数值
for c in ['gender', 'smoking', 'drinking', 'hypertension', 'hyperlipidemia', 'diabetes',
          'radiating_pain', 'food_related', 'anxiety_depression', 'prior_abdominal_surgery',
          'gallbladder_wall_thickening', 'fatty_liver', 'gallbladder_atrophy',
          'intraoperative_complications']:
    if c in df.columns and c not in numeric_cols and df[c].dtype == object:
        df[c] = pd.to_numeric(df[c], errors='coerce')
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

# 去掉全空或方差为0的列
X_df = df[numeric_cols].copy()
X_df = X_df.loc[:, X_df.notna().any(axis=0) & (X_df.std(skipna=True) > 0)]
numeric_cols = X_df.columns.tolist()

# 特征缺失值用各列中位数填充；仅保留目标变量非缺失的行
X_df = X_df.fillna(X_df.median())
valid = pd.notna(labels)
X_clean = X_df.loc[valid].astype(float)
y_clean = np.asarray(labels[valid], dtype=int)
print(f"有效样本数: {len(y_clean)}, 特征数: {len(numeric_cols)}")
print(f"PCS 分布: 未发生={np.sum(y_clean==0)}, 发生={np.sum(y_clean==1)}")

# ========== 4. 标准化 ==========
scaler = StandardScaler()
X_std = scaler.fit_transform(X_clean)

# ========== 5. 与「是否发生PCS」的相关性 ==========
corr_with_pcs = pd.Series({
    col: np.corrcoef(X_clean[col].values, y_clean)[0, 1]
    for col in numeric_cols
})
corr_with_pcs = corr_with_pcs.iloc[np.argsort(np.abs(corr_with_pcs.values))]

# 绘图：特征与 PCS 的相关系数条形图
fig1, ax1 = plt.subplots(figsize=(10, max(6, len(corr_with_pcs) * 0.22)))
colors = ['#e74c3c' if c < 0 else '#3498db' for c in corr_with_pcs.values]
ax1.barh(range(len(corr_with_pcs)), corr_with_pcs.values, color=colors, edgecolor='gray', linewidth=0.5)
ax1.set_yticks(range(len(corr_with_pcs)))
ax1.set_yticklabels(corr_with_pcs.index, fontsize=9)
ax1.axvline(0, color='black', linewidth=0.8)
ax1.set_xlabel('与「是否发生PCS」的相关系数')
ax1.set_title('各特征与是否发生PCS的相关性')
ax1.set_xlim(-1.05, 1.05)
ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
out_dir = Path(__file__).resolve().parent / 'output'
out_dir.mkdir(exist_ok=True)
fig1.savefig(out_dir / 'correlation_with_PCS.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"已保存: {out_dir / 'correlation_with_PCS.png'}")

# ========== 6. PCA 2D 降维 ==========
n_components = min(2, X_std.shape[0], X_std.shape[1])
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_std)
explained = pca.explained_variance_ratio_ * 100

n0, n1 = np.sum(y_clean == 0), np.sum(y_clean == 1)
fig2, ax2 = plt.subplots(figsize=(8, 6))
mask0 = y_clean == 0
mask1 = y_clean == 1
ax2.scatter(X_pca[mask0, 0], X_pca[mask0, 1], c='#3498db', alpha=0.7, edgecolors='k', linewidths=0.3, s=50, label=f'0 (否), n={n0}')
ax2.scatter(X_pca[mask1, 0], X_pca[mask1, 1], c='#e67e22', alpha=0.7, edgecolors='k', linewidths=0.3, s=50, label=f'1 (是), n={n1}')
ax2.set_xlabel(f'PC1 ({explained[0]:.1f}%)')
ax2.set_ylabel(f'PC2 ({explained[1]:.1f}%)')
ax2.set_title('PCA 2D 降维（按是否发生PCS着色）')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
fig2.savefig(out_dir / 'pca_2d_by_PCS.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"已保存: {out_dir / 'pca_2d_by_PCS.png'}")

# ========== 7. t-SNE 2D 降维 ==========
# 样本多时先做 PCA 到 50 维再 t-SNE 以加速
if X_std.shape[1] > 50:
    X_for_tsne = PCA(n_components=50, random_state=42).fit_transform(X_std)
else:
    X_for_tsne = X_std
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_for_tsne) - 1), max_iter=1000)
X_tsne = tsne.fit_transform(X_for_tsne)

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(X_tsne[mask0, 0], X_tsne[mask0, 1], c='#3498db', alpha=0.7, edgecolors='k', linewidths=0.3, s=50, label=f'0 (否), n={n0}')
ax3.scatter(X_tsne[mask1, 0], X_tsne[mask1, 1], c='#e67e22', alpha=0.7, edgecolors='k', linewidths=0.3, s=50, label=f'1 (是), n={n1}')
ax3.set_xlabel('t-SNE 1')
ax3.set_ylabel('t-SNE 2')
ax3.set_title('t-SNE 2D 降维（按是否发生PCS着色）')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
fig3.savefig(out_dir / 'tsne_2d_by_PCS.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"已保存: {out_dir / 'tsne_2d_by_PCS.png'}")

# ========== 8. 可选：相关性热力图（特征子集 + PCS） ==========
# 取与 PCS 相关性较强的若干特征 + PCS 做热力图
n_top = min(20, len(corr_with_pcs))
top_cols = corr_with_pcs.tail(n_top).index.tolist()
heat_df = X_clean[top_cols].copy()
heat_df['PCS_occurred'] = y_clean
corr_matrix = heat_df.corr()
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, square=True,
            linewidths=0.5, ax=ax4, vmin=-1, vmax=1)
ax4.set_title('特征与是否发生PCS 相关性热力图（高相关特征子集）')
plt.tight_layout()
fig4.savefig(out_dir / 'correlation_heatmap_PCS.png', dpi=150, bbox_inches='tight')
plt.close(fig4)
print(f"已保存: {out_dir / 'correlation_heatmap_PCS.png'}")

print("\n全部可视化已完成，结果保存在 output/ 目录。")
