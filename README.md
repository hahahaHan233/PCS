# PCS 风险预测与可解释机器学习项目

本项目围绕胆囊疾病患者胆囊切除术后综合征（post-cholecystectomy syndrome, PCS）的发生风险，构建基于多源临床数据的机器学习预测模型，并结合 SHAP、校准曲线、决策曲线分析、症状亚型探索和潜在表型分析，为医学与计算机交叉论文提供可复现的实验基础。

推荐论文定位：

> 基于多源术前/围手术期临床数据和可解释机器学习的 PCS 发生风险预测、模型解释与临床风险分层研究。

## 当前数据

- 原始数据：`data/数据（终）.xlsx`
- 建模数据：`dataset.csv`
- 样本量：311 例
- 结局变量：是否发生 PCS
- PCS 阳性：70 例
- PCS 阴性：241 例
- PCS 发生率：22.5%
- 建模预测变量：36 个术前/围手术期变量
- 排除变量：姓名、住院号等身份识别变量，以及 `PCS发生时间`、`PCS症状类型` 等结局后变量

变量体系包括人口学资料、生活方式、临床症状、既往病史、影像学特征和实验室指标。数据字典见 `output/dataset_schema_mapping.csv`。

## 已完成分析

### 数据整理与探索

- 已从原始 Excel 构建标准化建模数据 `dataset.csv`
- 已生成变量映射、描述统计、缺失值概览和初步分布图
- 已完成 PCS 与非 PCS 组的初步关联分析
- 已生成实验室指标、分类变量和 PCS 发生率相关图表

主要探索发现：

- PCS 组中 GGT、ALT、ALP、AST、CA19-9、总胆红素、总胆汁酸等肝胆实验室指标明显更高。
- 症状持续时间、疼痛频率、腹痛和结石位置等分类变量与 PCS 发生存在一定关联。

### 机器学习模型

已比较以下模型：

- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM
- Gaussian Naive Bayes
- XGBoost
- Soft Voting Ensemble
- Hard Voting Ensemble
- Stacking Ensemble
- Weighted Soft Voting Ensemble

验证策略：

- 5 折分层交叉验证
- 基于 out-of-fold prediction 计算模型性能、ROC、PR、校准曲线和 DCA
- 数值变量使用中位数插补和标准化
- 分类变量使用众数插补和 one-hot encoding
- 类别不平衡通过 `class_weight` 或模型内权重参数处理

当前最佳单模型为 Random Forest：

| 指标 | 数值 |
| --- | --- |
| OOF AUC | 0.917 |
| OOF average precision | 0.842 |
| OOF Brier score | 0.088 |
| Sensitivity | 0.714 |
| Specificity | 0.946 |
| OOF confusion matrix | TN=228, FP=13, FN=20, TP=50 |

集成架构验证显示，weighted soft voting 的 OOF AUC 为 0.915，接近但未超过 Random Forest；soft voting 在 Brier score 上更好。因此当前建议将 Random Forest 作为主模型，将集成模型作为架构验证或补充实验。

结果目录：

- `output/modeling/model_report.md`
- `output/modeling/model_performance.xlsx`
- `output/modeling/model_performance_summary.csv`
- `output/modeling/model_performance_by_fold.csv`
- `output/modeling/cross_validated_oof_predictions.csv`
- `output/modeling/roc_curve.pdf`
- `output/modeling/pr_curve.pdf`
- `output/modeling/calibration_curve.pdf`
- `output/modeling/dca_curve.pdf`

### SHAP 可解释性分析

已对最终 Random Forest 模型进行 SHAP 解释。模型主要依赖肝胆实验室指标和术前症状相关变量。

Top 10 原始变量：

| 排名 | 变量 | 临床类别 | mean abs SHAP |
| --- | --- | --- | --- |
| 1 | GGT | Laboratory | 0.0825 |
| 2 | ALT | Laboratory | 0.0761 |
| 3 | AST | Laboratory | 0.0481 |
| 4 | ALP | Laboratory | 0.0420 |
| 5 | CA19-9 | Laboratory | 0.0400 |
| 6 | Total bilirubin | Laboratory | 0.0340 |
| 7 | Symptom duration | Symptoms | 0.0206 |
| 8 | Total bile acid | Laboratory | 0.0195 |
| 9 | Pain frequency | Symptoms | 0.0179 |
| 10 | Age | Demographics | 0.0075 |

领域层面贡献：

| 临床类别 | mean abs SHAP |
| --- | --- |
| Laboratory | 0.3573 |
| Symptoms | 0.0425 |
| Demographics | 0.0289 |
| Imaging | 0.0147 |
| Medical history | 0.0030 |
| Lifestyle | 0.0024 |

结果目录：

- `output/shap_analysis/shap_report.md`
- `output/shap_analysis/shap_summary_plot.pdf`
- `output/shap_analysis/shap_feature_importance.pdf`
- `output/shap_analysis/shap_variable_importance.pdf`
- `output/shap_analysis/shap_domain_importance.pdf`
- `output/shap_analysis/shap_single_patient_waterfall.pdf`
- `output/shap_analysis/shap_direction_summary.csv`

### PCS 症状亚型探索

已将原始 `PCS症状类型` 自由文本转化为结构化标签。该任务是 PCS 阳性患者内部的探索性补充分析，不作为 PCS 二分类预测模型的输入。

构建类别：

| 类别 | 含义 | 样本数 |
| --- | --- | --- |
| `no_pcs` | 未发生 PCS | 241 |
| `pain` | 腹痛表型 | 24 |
| `constipation` | 便秘表型 | 19 |
| `gi_dysfunction_dyspepsia` | 消化不良/胃肠功能紊乱相关表型 | 26 |
| `pcs_unknown` | PCS 阳性但症状文本缺失 | 1 |

症状亚型多分类结果较弱，当前最优模型 XGBoost 的 OOF macro F1 为 0.381，macro one-vs-rest AUC 为 0.529。因此该部分应作为 hypothesis-generating analysis，而不是论文主结论。

结果目录：

- `output/pcs_symptom_task/pcs_symptom_prediction_report.md`
- `output/pcs_symptom_task/pcs_symptom_labeled_dataset.csv`
- `output/pcs_symptom_task/pcs_symptom_label_mapping.csv`
- `output/pcs_symptom_task/pcs_symptom_prediction_outputs.xlsx`
- `output/pcs_symptom_task/pcs_symptom_class_distribution.pdf`
- `output/pcs_symptom_task/pcs_symptom_confusion_matrix.pdf`
- `output/pcs_symptom_task/pcs_symptom_ovr_roc.pdf`

### 潜在表型与聚类探索

已基于术前/围手术期变量进行 PCA、t-SNE 和 K-means 聚类探索。该分析用于观察 PCS 风险和症状亚型是否在术前变量空间中存在潜在结构。

当前结果：

- K-means 最优解为 k=2
- silhouette score = 0.219
- Cluster 0：45 例，PCS 43 例，PCS 发生率 95.6%
- Cluster 1：266 例，PCS 27 例，PCS 发生率 10.2%
- 两个 cluster 的 PCS 中位发生时间约为 43.0-86.0 天

由于 silhouette score 较低，该部分应作为探索性表型分析，不提供因果证据，也不作为主模型验证证据。

结果目录：

- `output/latent_phenotypes/latent_phenotype_report.md`
- `output/latent_phenotypes/latent_phenotype_assignments.csv`
- `output/latent_phenotypes/kmeans_silhouette_scores.csv`
- `output/latent_phenotypes/cluster_pcs_rate.csv`
- `output/latent_phenotypes/cluster_symptom_distribution.csv`
- `output/latent_phenotypes/figS4_pca_by_pcs.pdf`
- `output/latent_phenotypes/figS4_tsne_by_pcs.pdf`
- `output/latent_phenotypes/figS5_cluster_heatmap.pdf`

## 论文图表状态

已完成：

- Table 3：模型性能比较
- Figure 2：ROC 曲线
- Figure 3：校准曲线
- Figure 4：DCA 曲线
- Figure 5：SHAP summary plot
- Figure 6：SHAP feature importance
- Supplementary Figure：症状类别分布、混淆矩阵、one-vs-rest ROC
- Supplementary Figure：PCA / t-SNE 降维图
- Supplementary Figure：潜在表型聚类热图
- Supplementary Table：症状标签映射、聚类发生率与症状分布

尚需完成：

- Figure 1：病例筛选流程图或研究流程图
- Table 1：PCS 组与非 PCS 组基线特征
- Table 2：单因素和多因素 Logistic 回归
- Table 4：风险分层结果
- Figure 7：风险分层可视化
- Supplementary Table：变量单位、参考范围、缺失比例、模型超参数

LaTeX demo 位于 `latex_demo_project/`，已优先使用 PDF 图件。

## 复现流程

建议从项目根目录运行：

```powershell
python scripts/build_dataset_artifacts.py
python scripts/plot_pcs_associations.py
python scripts/train_ml_models.py
python scripts/explain_best_model_shap.py
python scripts/build_pcs_symptom_task.py
python scripts/explore_latent_phenotypes.py
python scripts/export_figures_to_pdf.py
```

如需编译当前 LaTeX demo：

```powershell
cd latex_demo_project
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## 主要脚本

| 脚本 | 功能 | 主要输出 |
| --- | --- | --- |
| `scripts/build_dataset_artifacts.py` | 从原始 Excel 构建 `dataset.csv`、数据字典和描述统计 | `dataset.csv`, `output/dataset_*` |
| `scripts/plot_pcs_associations.py` | PCS 组间初步关联分析和图表 | `output/pcs_association_figures/` |
| `scripts/train_ml_models.py` | 训练和验证机器学习模型、集成模型、ROC/PR/校准/DCA | `output/modeling/` |
| `scripts/explain_best_model_shap.py` | 对最佳模型进行 SHAP 解释 | `output/shap_analysis/` |
| `scripts/build_pcs_symptom_task.py` | 症状文本标签化和 PCS 阳性人群内症状亚型预测 | `output/pcs_symptom_task/` |
| `scripts/explore_latent_phenotypes.py` | PCA、t-SNE、K-means 潜在表型探索 | `output/latent_phenotypes/` |
| `scripts/export_figures_to_pdf.py` | 将关键图件复制/转换到 LaTeX 图件目录 | `latex_demo_project/figures/` |

## 依赖环境

核心 Python 包：

- pandas
- numpy
- scipy
- scikit-learn
- xgboost
- shap
- matplotlib
- seaborn
- openpyxl

如果缺少依赖，可按本地环境安装：

```powershell
python -m pip install pandas numpy scipy scikit-learn xgboost shap matplotlib seaborn openpyxl
```

## 投稿前最需要补强的实验

为了提高医学与计算机交叉论文的接受概率，建议按优先级补充：

1. 正式 Table 1：PCS 组与非 PCS 组基线特征比较。
2. 单因素和多因素 Logistic 回归：报告 OR、95% CI、P 值，并与 SHAP 结果对照。
3. 风险分层：基于模型预测概率划分低/中/高风险组，报告各层实际 PCS 发生率、PPV、NPV、敏感度和特异度。
4. Bootstrap 或重复交叉验证：补充 AUC、Brier score、校准斜率和校准截距的置信区间。
5. 特征组消融实验：比较仅人口学、仅症状、仅影像、仅实验室、症状+实验室、全部变量的性能。
6. 阈值决策表：在 10%、20%、30% 等临床阈值下报告每 100 名患者中的真阳性、假阳性和漏诊数。
7. 校准增强：比较原始模型、Platt scaling 和 isotonic calibration。
8. SHAP 稳定性：用交叉验证或 bootstrap 评估 top 特征排名稳定性。
9. 敏感性分析：实验室极端值截尾、缺失值策略、类别不平衡处理方式。
10. 亚组性能：按性别、年龄、腹痛状态或实验室异常程度报告模型性能。

其中最建议优先完成 `Table 1 + Logistic 回归 + 风险分层 + bootstrap/消融实验`。这几项最能回应医学审稿人对临床可信度、模型稳定性和实际应用价值的关注。

## 推荐论文主线

建议把主文聚焦在：

1. PCS 术前/围手术期风险识别的临床需求。
2. 多源临床变量融合建模。
3. Random Forest 在内部验证中获得较好的区分度、校准度和净获益。
4. SHAP 显示肝胆实验室指标和症状负担是主要模型驱动因素。
5. 风险分层可为术后随访强度和健康宣教提供量化参考。

症状亚型预测和潜在表型聚类建议作为补充分析，表述为探索性发现。

## 当前局限

- 单中心或单数据来源，外部推广性有限。
- 样本量较小，PCS 阳性仅 70 例，复杂模型存在过拟合风险。
- 当前主要为内部交叉验证，尚缺少外部验证和前瞻性验证。
- PCS 诊断标准、随访时间窗、纳入排除标准和伦理信息仍需在论文中明确。
- 手术方式、术中情况、术后用药、并发症、心理量表等潜在影响因素尚未充分纳入。
- 症状亚型预测样本量较小，仅适合作为探索性补充。
- 聚类和降维分析不能提供因果证据。

## 推荐表述

本研究基于多源术前/围手术期临床数据，构建并内部验证 PCS 发生风险预测模型；通过校准曲线、决策曲线分析和 SHAP 可解释性方法评估模型的临床可用性与关键驱动因素；进一步探索 PCS 症状异质性和潜在临床表型，为胆囊疾病患者术后个体化随访和风险分层提供量化参考。
