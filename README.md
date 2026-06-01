# 基于 PCS 的科研论文设计说明

## 1. 项目定位

本项目基于 `data/数据（终）.xlsx` 中的临床数据，围绕胆囊疾病患者术后 PCS（post-cholecystectomy syndrome，胆囊切除术后综合征）的发生风险开展研究。当前数据包含 311 例患者，结局变量为“是否发生 PCS”，其中 PCS 发生 70 例，未发生 241 例，发生率约 22.5%。

本课题建议定位为：

> 基于多源临床数据和可解释机器学习的 PCS 发生风险预测模型构建与临床应用研究。

该定位兼具医学意义和计算机技术特色：医学上关注 PCS 的早期识别与风险分层，计算机技术上体现数据融合、机器学习建模、模型评价、可解释性分析和临床辅助决策。

## 2. 推荐论文题目

可根据论文偏重选择题目：

1. 基于机器学习的胆囊切除术后综合征发生风险预测模型构建及可解释性分析
2. 融合多源临床数据的 PCS 风险分层模型构建与 SHAP 解释研究
3. 胆囊疾病患者术后 PCS 发生的危险因素分析及智能预测模型研究
4. 基于术前临床特征的 PCS 发生风险预测模型构建与验证
5. 面向个体化随访的 PCS 智能风险评估模型研究

其中最推荐第 1 个或第 2 个，因为能同时突出临床问题、机器学习方法和可解释人工智能。

## 3. 论文需要包含的核心内容

### 3.1 背景与意义

论文引言部分需要说明：

- 胆囊切除术是胆囊结石、胆囊炎等疾病的重要治疗方式。
- 部分患者术后仍会出现腹痛、腹胀、消化不良、便秘、胃肠功能紊乱等症状，即 PCS。
- PCS 影响患者术后生活质量，也增加随访和诊疗负担。
- 传统研究多集中于 PCS 的症状描述、发生率和危险因素分析。
- 临床上更需要在术前或围手术期识别高风险患者，以便开展个体化随访、健康宣教和早期干预。
- 机器学习适合处理多维临床变量，可用于构建风险预测模型；SHAP 等可解释 AI 方法可以提升模型透明度和临床可信度。

### 3.2 数据来源与变量体系

需要描述数据来源、纳入排除标准和变量类型。

当前数据主要包括：

- 人口学资料：年龄、性别、职业、身高、体重、BMI、教育水平。
- 生活方式：是否吸烟、是否饮酒。
- 临床症状：症状持续时间、有否腹痛、疼痛频率、是否有放射痛、是否与进食有关。
- 既往病史：高血压、高血脂、糖尿病、焦虑/抑郁、既往腹部手术史。
- 影像学特征：胆囊壁是否增厚、最大结石直径、结石数量、结石位置、胆总管直径、是否有脂肪肝、胆囊是否萎缩。
- 实验室指标：ALT、AST、ALP、GGT、总胆红素、总胆汁酸、总胆固醇、甘油三酯、甲胎蛋白、CA199。
- 结局变量：是否发生 PCS、PCS 发生时间、PCS 症状类型。

### 3.3 数据预处理

需要说明：

- 缺失值统计与处理方式。
- 数值型变量异常值识别，可采用箱线图、IQR 法或医学参考范围辅助判断。
- 分类变量编码，如 One-Hot Encoding。
- 数值变量标准化或归一化。
- 训练集和测试集划分，建议采用分层抽样。
- 类别不平衡处理。当前 PCS 发生率约 22.5%，可使用 `class_weight`、阈值调整、SMOTE 或保持原始比例并报告灵敏度/特异度。

### 3.4 统计学分析

传统统计分析仍然需要保留，作为医学论文的基础。

建议包括：

- PCS 组与非 PCS 组的基线特征比较。
- 连续变量根据分布选择 t 检验或 Mann-Whitney U 检验。
- 分类变量采用卡方检验或 Fisher 精确检验。
- 单因素 Logistic 回归筛选候选变量。
- 多因素 Logistic 回归分析独立危险因素。
- 报告 OR 值、95% CI 和 P 值。

### 3.5 机器学习建模

建议至少比较以下模型：

- Logistic Regression：作为传统基线模型。
- Random Forest：适合非线性关系和变量重要性分析。
- XGBoost 或 LightGBM：适合结构化临床数据，预测性能通常较强。
- SVM：可作为对比模型。
- KNN 或朴素贝叶斯：可选，不是必须。

模型评价指标建议包括：

- AUC
- Accuracy
- Sensitivity / Recall
- Specificity
- Precision
- F1-score
- Calibration curve
- Brier score
- Decision curve analysis（DCA）

医学论文中不能只看准确率。由于 PCS 样本占比不高，灵敏度、AUC、校准曲线和 DCA 更重要。

### 3.6 SHAP 可解释性分析

SHAP 用于解释机器学习模型的预测结果。它可以说明每个特征对 PCS 预测风险的贡献方向和贡献大小。

建议输出：

- SHAP summary plot：展示总体上最重要的变量。
- SHAP bar plot：展示平均绝对 SHAP 值的特征重要性排序。
- SHAP dependence plot：观察某个指标升高时 PCS 风险如何变化。
- 单例患者 SHAP force plot 或 waterfall plot：解释某个高风险患者为什么被模型判定为高风险。

在本项目中，初步探索显示 GGT、ALT、ALP、总胆红素、AST、总胆汁酸、CA199 等实验室指标与 PCS 发生关系较明显，后续可通过 SHAP 验证其在模型中的贡献。

### 3.7 风险分层与临床转化

建议将模型预测概率转化为风险分层：

- 低风险：常规随访。
- 中风险：加强术后饮食、消化功能和症状监测。
- 高风险：重点随访，必要时提前干预。

这样可以让论文从“模型构建”进一步走向“临床辅助决策”，提升应用价值。

## 4. 创新点在哪里

### 创新点 1：研究视角前移

传统 PCS 研究多关注术后症状总结或危险因素分析。本研究将研究视角前移至术前或围手术期，尝试在 PCS 发生前识别高风险患者。

可写为：

> 本研究将 PCS 研究由术后被动描述前移至术前主动预测，为高风险患者的个体化随访和早期干预提供依据。

### 创新点 2：多源临床数据融合

本研究不是只分析某一个实验室指标，而是融合人口学资料、生活方式、症状、影像学特征和实验室指标。

可写为：

> 本研究构建了包含人口学、症状学、影像学和实验室指标的多维特征体系，较单一变量分析更全面地反映患者 PCS 发生风险。

### 创新点 3：机器学习预测模型

传统 Logistic 回归可以解释危险因素，但对复杂非线性关系和变量交互的捕捉能力有限。机器学习模型可以作为补充。

可写为：

> 本研究比较多种机器学习算法在 PCS 风险预测中的性能，探索非线性模型在 PCS 预测中的应用价值。

### 创新点 4：可解释人工智能

医学场景不能只追求模型准确率，还需要解释模型为什么这样预测。

可写为：

> 本研究引入 SHAP 方法解释模型预测结果，量化关键变量对 PCS 发生风险的贡献方向和大小，提高模型透明度和临床可理解性。

### 创新点 5：风险分层应用

论文不止停留在“发现因素”和“建模型”，而是将模型输出转化为风险分层。

可写为：

> 本研究基于模型预测概率构建 PCS 风险分层策略，为术后随访强度、健康宣教和早期干预提供量化参考。

## 5. 还需要完成哪些工作

### 5.1 数据层面

- 核对所有变量含义，形成数据字典。
- 明确 PCS 的诊断标准和随访时间范围。
- 明确“PCS发生时间”的单位，是天、周还是月。
- 核查 PCS 症状类型是否允许多标签，例如同一患者是否同时存在腹痛和消化不良。
- 明确纳入排除标准。
- 核查缺失值和异常值。
- 确认是否存在术后治疗、用药、并发症等未纳入变量。

### 5.2 方法层面

- 完成描述性统计和组间比较。
- 完成单因素和多因素 Logistic 回归。
- 构建机器学习模型并进行交叉验证。
- 比较模型性能。
- 使用 SHAP 解释最佳模型。
- 完成风险分层。
- 如果样本量允许，可做内部验证，如 Bootstrap 或重复交叉验证。

### 5.3 论文层面

- 明确研究类型：回顾性队列研究或回顾性病例对照研究。
- 完成伦理审批或说明数据来源合规性。
- 按 TRIPOD / TRIPOD+AI 思路报告预测模型研究。
- 讨论样本量、单中心数据、外部验证缺失等局限性。
- 提出未来可进行多中心外部验证和临床决策系统开发。

### 5.4 程序与结果输出

建议建立如下输出：

- `output/baseline_table.xlsx`：PCS 组与非 PCS 组基线表。
- `output/logistic_regression_results.xlsx`：Logistic 回归结果。
- `output/model_performance.xlsx`：各模型评价指标。
- `output/roc_curve.png`：ROC 曲线。
- `output/calibration_curve.png`：校准曲线。
- `output/dca_curve.png`：决策曲线。
- `output/shap_summary_plot.png`：SHAP 总结图。
- `output/shap_feature_importance.png`：SHAP 重要性排序图。
- `output/shap_single_patient.png`：单例患者解释图。

## 6. 推荐论文结构

### 摘要

- 目的：构建 PCS 发生风险预测模型并解释关键影响因素。
- 方法：回顾性收集临床资料，建立 Logistic 回归和机器学习模型，采用 SHAP 解释模型。
- 结果：报告样本量、PCS 发生率、最佳模型性能、关键预测变量。
- 结论：模型可用于 PCS 高风险患者识别和风险分层。

### 引言

- PCS 的临床问题。
- 现有研究不足。
- 机器学习和可解释 AI 的价值。
- 本研究目的。

### 资料与方法

- 研究对象。
- 变量定义。
- 结局定义。
- 数据预处理。
- 统计分析。
- 机器学习建模。
- SHAP 分析。
- 模型评价。

### 结果

- 基线特征。
- PCS 发生情况。
- 单因素和多因素分析。
- 机器学习模型性能比较。
- 最佳模型的 SHAP 解释。
- 风险分层结果。

### 讨论

- 主要发现。
- 与既往研究比较。
- 关键变量的临床解释。
- 机器学习模型的临床意义。
- 创新点。
- 局限性。
- 未来研究方向。

### 结论

- 简明总结模型价值和临床意义。

## 7. 相关文献方向

### 7.1 PCS 发生率、症状和危险因素

需要重点查找：

- PCS 的定义、发生率和症状谱。
- 胆囊切除术后腹痛、消化不良、腹胀、便秘等症状的随访研究。
- PCS 相关危险因素，如胆总管结石、胆道功能异常、术前症状、肝胆酶学异常等。

可检索关键词：

- `post-cholecystectomy syndrome incidence risk factors`
- `postcholecystectomy syndrome symptoms abdominal pain dyspepsia`
- `risk factors for post-cholecystectomy syndrome`
- `胆囊切除术后综合征 危险因素`
- `胆囊切除术后综合征 发生率`

### 7.2 机器学习临床预测模型

需要重点查找：

- 机器学习预测腹腔镜胆囊切除术后并发症的研究。
- 机器学习在肝胆外科或消化系统疾病风险预测中的应用。
- 结构化临床数据预测模型的建模和验证方法。

可检索关键词：

- `machine learning prediction cholecystectomy complications`
- `machine learning clinical prediction model hepatobiliary surgery`
- `XGBoost clinical prediction model surgery complications`
- `machine learning post-operative complications prediction`

### 7.3 SHAP 与可解释 AI

需要重点查找：

- SHAP 原始论文。
- TreeSHAP 论文。
- SHAP 在医学预测模型中的应用综述或实例。

可检索关键词：

- `SHAP explainable artificial intelligence clinical prediction model`
- `SHAP medical machine learning prediction model`
- `TreeSHAP feature attribution tree ensembles`
- `explainable artificial intelligence medicine SHAP`

### 7.4 预测模型报告规范

需要重点查找：

- TRIPOD 预测模型报告规范。
- TRIPOD+AI 或 AI 临床预测模型报告规范。
- PROBAST 或 PROBAST-AI 风险偏倚评价工具。

可检索关键词：

- `TRIPOD clinical prediction model reporting guideline`
- `TRIPOD AI prediction model reporting guideline`
- `PROBAST AI prediction model risk of bias`
- `clinical prediction model machine learning reporting guideline`

## 8. 已查到的关键参考文献

以下文献可作为后续精读起点：

1. Isherwood J, Oakland K, Khanna A. [A systematic review of the aetiology and management of post cholecystectomy syndrome](https://www.sciencedirect.com/science/article/pii/S1479666X18300453). 可用于 PCS 病因和处理综述。
2. StatPearls. [Postcholecystectomy Syndrome](https://www.ncbi.nlm.nih.gov/books/NBK539902/). 可用于 PCS 定义、症状和临床背景。
3. Incidence risk and risk factors for postcholecystectomy syndrome: [A systematic review and meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC12908842/). 可用于 PCS 发生率和危险因素背景。
4. Leghari S, et al. [Predictions of postoperative and perioperative complications of laparoscopic cholecystectomy using machine learning algorithms: systematic review](https://bmcsurg.biomedcentral.com/articles/10.1186/s12893-025-03035-z). 可用于证明机器学习已被用于胆囊切除术后并发症预测，但 PCS 专项预测仍有研究空间。
5. Lundberg SM, Lee SI. [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html). NeurIPS 2017. SHAP 的经典原始论文。
6. Lundberg SM, et al. [Consistent individualized feature attribution for tree ensembles](https://www.nature.com/articles/s42256-019-0138-9). TreeSHAP 相关论文，适合解释 XGBoost、LightGBM、Random Forest。
7. Collins GS, et al. [TRIPOD Statement](https://www.tripod-statement.org/). 预测模型研究报告规范。
8. TRIPOD+AI Statement: [updated guidance for reporting clinical prediction models that use regression or machine learning methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/). 人工智能预测模型报告规范，可作为机器学习医学论文写作参照。
9. PROBAST+AI: [assessing quality, risk of bias and applicability of prediction models based on regression or AI methods](https://www.bmj.com/content/388/bmj-2024-082505). 可用于论文方法学质量控制。

后续正式写论文时，建议使用 PubMed、Web of Science、CNKI、Google Scholar 进一步补充近 5 年文献，并优先选择系统综述、指南、临床队列研究和医学 AI 预测模型研究。

## 9. 建议技术路线

```text
数据整理
  -> 缺失值与异常值处理
  -> PCS 组/非 PCS 组基线比较
  -> 单因素分析
  -> 多因素 Logistic 回归
  -> 机器学习模型构建
  -> 模型性能评价
  -> SHAP 可解释性分析
  -> 风险分层
  -> 临床意义与论文撰写
```

## 10. 最推荐的论文表述

本研究的核心创新可概括为：

> 本研究基于多源临床数据，构建 PCS 发生风险预测模型，并引入 SHAP 可解释人工智能方法揭示关键风险因素对模型预测的贡献，进一步将预测概率转化为风险分层策略，为胆囊疾病患者术后个体化随访和早期干预提供参考。
