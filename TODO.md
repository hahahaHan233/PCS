# PCS 医学期刊论文项目 TODO

> 项目主题：基于多源临床数据构建胆囊切除术后综合征（post-cholecystectomy syndrome, PCS）发生风险预测模型，并结合可解释机器学习用于风险因素识别和临床风险分层。

## 0. 当前项目状态

- [x] 原始数据已整理为 `dataset.csv`
- [x] 当前样本量：311 例
- [x] 结局变量：是否发生 PCS
- [x] PCS 阳性：70 例；PCS 阴性：241 例；发生率约 22.5%
- [x] 已生成数据字典：`output/dataset_schema_mapping.csv`
- [x] 已生成描述统计输出：`output/dataset_numeric_summary.csv`、`output/dataset_categorical_summary.csv`
- [x] 已生成初步数据可视化：`output/dataset_insights/`
- [x] 已生成 PCS 相关性探索图表和统计表：`output/pcs_association_figures/`
- [ ] 完成正式论文用基线表
- [ ] 完成单因素和多因素 Logistic 回归
- [x] 完成机器学习模型训练、验证和性能比较
- [x] 完成 SHAP 可解释性分析
- [x] 完成校准曲线、决策曲线分析
- [x] 补充 PCS 症状类型文本标签化和症状发生类别预测任务
- [ ] 完成风险分层
- [ ] 完成论文初稿、图表编号和投稿格式整理

## 1. 整体处理工作流

### 1.1 数据核查与数据字典

- [x] 明确纳入建模数据中的变量：人口学、生活方式、临床症状、既往病史、影像学特征、实验室指标和 PCS 结局
- [x] 排除身份识别变量：姓名、住院号
- [x] 排除结局后变量：PCS 发生时间、PCS 症状类型
- [ ] 核对原始 Excel 中所有中文变量名、英文变量名和临床含义是否一致
- [ ] 补充每个变量的单位、取值范围和临床参考范围
- [ ] 明确 PCS 的诊断标准、判定来源和随访时间窗
- [ ] 明确 PCS 发生时间的单位，以及是否只用于描述性分析
- [ ] 明确 PCS 症状类型是否允许多标签记录
- [ ] 明确纳入标准、排除标准和病例筛选流程
- [ ] 输出正式数据字典表，供论文 Supplementary Table 使用

### 1.2 数据清洗与预处理

- [x] 生成缺失值概览图：`fig2_missingness_bar.png`
- [x] 生成数值型变量和分类变量描述统计
- [ ] 逐项核查缺失比例较高变量，决定删除、插补或保留原始缺失
- [ ] 核查 ALT、AST、ALP、GGT、胆红素、胆汁酸、CA19-9 等实验室指标的极端值
- [ ] 根据临床参考范围或箱线图/IQR 方法标记可能异常值
- [ ] 明确异常值处理原则：保留、截尾、winsorize 或敏感性分析
- [ ] 对分类变量进行规范编码，并保留原始中文标签和英文建模标签的映射
- [ ] 对连续变量进行建模前标准化或归一化
- [ ] 建立可复现的预处理 pipeline，避免训练集和测试集之间数据泄漏

### 1.3 描述性统计与组间比较

- [x] 已完成 PCS 与非 PCS 组的初步非参数比较
- [x] 初步提示与 PCS 相关较强的连续变量包括 GGT、ALT、ALP、AST、CA19-9、总胆红素、总胆汁酸等
- [x] 初步提示与 PCS 相关的分类变量包括症状持续时间、疼痛频率、腹痛、结石位置等
- [ ] 生成正式 Table 1：PCS 组与非 PCS 组基线特征比较
- [ ] 连续变量根据分布报告为 mean ± SD 或 median (IQR)
- [ ] 分类变量报告 n (%)
- [ ] 根据变量分布选择 t 检验、Mann-Whitney U 检验、卡方检验或 Fisher 精确检验
- [ ] 对多重比较结果说明是否进行 FDR 校正
- [ ] 输出：`output/baseline_table.xlsx`

### 1.4 传统统计建模

- [ ] 完成单因素 Logistic 回归
- [ ] 报告每个候选变量的 OR、95% CI 和 P 值
- [ ] 根据临床意义和单因素结果筛选多因素模型候选变量
- [ ] 检查多重共线性，尤其是 ALT、AST、ALP、GGT、胆红素和胆汁酸之间的相关性
- [ ] 完成多因素 Logistic 回归
- [ ] 报告独立危险因素及其临床解释
- [ ] 输出：`output/logistic_regression_results.xlsx`
- [ ] 讨论 Logistic 回归与机器学习模型的角色差异：解释危险因素 vs 提升预测性能

### 1.5 机器学习模型构建

- [x] 明确建模任务：二分类预测 PCS 发生
- [x] 进行训练集/测试集划分，建议使用分层抽样
- [x] 考虑重复交叉验证或 bootstrap 内部验证
- [x] 建立基线模型：Logistic Regression
- [x] 建立非线性模型：Random Forest
- [x] 建立梯度提升模型：XGBoost 或 LightGBM
- [x] 可选比较模型：SVM、KNN、Naive Bayes
- [x] 针对 PCS 阳性比例约 22.5% 的类别不平衡问题，比较 class_weight、阈值调整或重采样策略
- [ ] 使用训练集进行调参，测试集只用于最终评估
- [x] 记录随机种子、软件版本和参数设置，保证可复现

### 1.6 模型评价

- [x] 输出各模型 AUC
- [x] 输出 Accuracy、Sensitivity、Specificity、Precision、F1-score
- [x] 输出混淆矩阵
- [x] 绘制 ROC 曲线：`output/modeling/roc_curve.png`
- [x] 绘制 PR 曲线，可选：`output/modeling/pr_curve.png`
- [x] 绘制校准曲线：`output/modeling/calibration_curve.png`
- [x] 计算 Brier score
- [x] 完成决策曲线分析（DCA）：`output/modeling/dca_curve.png`
- [x] 输出模型性能汇总表：`output/modeling/model_performance.xlsx`
- [x] 选择最佳模型时同时考虑 AUC、敏感度、校准度和临床净获益

### 1.7 SHAP 可解释性分析

- [x] 对最佳机器学习模型进行 SHAP 分析
- [x] 输出 SHAP summary plot：`output/shap_analysis/shap_summary_plot.png`
- [x] 输出 SHAP feature importance bar plot：`output/shap_analysis/shap_feature_importance.png`
- [x] 对关键变量绘制 SHAP dependence plot
- [x] 重点解释 GGT、ALT、ALP、AST、总胆红素、总胆汁酸、CA19-9、症状持续时间、疼痛频率等变量
- [x] 选择 1-2 个典型高风险病例做 waterfall/force plot：`output/shap_analysis/shap_single_patient_waterfall.png`
- [x] 将 SHAP 结果转化为临床语言，避免只描述算法图形

### 1.8 风险分层与临床转化

- [ ] 根据最佳模型预测概率确定低、中、高风险阈值
- [ ] 比较不同风险层的实际 PCS 发生率
- [ ] 输出风险分层表：`output/risk_stratification.xlsx`
- [ ] 绘制风险分层柱状图或热图
- [ ] 给出初步临床建议：低风险常规随访，中风险加强宣教和症状监测，高风险重点随访和早期干预
- [ ] 在讨论中强调该分层策略仍需外部验证

### 1.9 代码、结果和复现

- [ ] 整理脚本命名，使其对应论文分析流程
- [ ] 新增 `scripts/build_baseline_table.py`
- [ ] 新增 `scripts/run_logistic_regression.py`
- [x] 新增 `scripts/train_ml_models.py`
- [x] 新增 `scripts/explain_best_model_shap.py`
- [x] 新增 `scripts/build_pcs_symptom_task.py`
- [ ] 新增 `scripts/risk_stratification.py`
- [x] 统一所有输出路径到 `output/`
- [x] 将关键表格保存为 xlsx/csv，关键图片保存为 png/pdf
- [ ] 在 README 中补充一键运行说明

### 1.10 PCS 症状类型标签化与补充预测任务

- [x] 从原始 Excel 中抽取 `PCS症状类型` 自由文本字段
- [x] 将原始文本转化为结构化标签：腹痛、便秘、胃肠功能紊乱、消化不良、肠道菌群失调、慢性胃炎、腹胀
- [x] 合并为适合建模的症状发生类别：`no_pcs`、`pain`、`constipation`、`gi_dysfunction_dyspepsia`
- [x] 对 PCS 阳性但症状文本缺失者标记为 `pcs_unknown`
- [x] 输出症状标签数据集：`output/pcs_symptom_task/pcs_symptom_labeled_dataset.csv`
- [x] 输出标签映射表：`output/pcs_symptom_task/pcs_symptom_label_mapping.csv`
- [x] 构建补充预测任务：PCS 症状发生类别多分类预测
- [x] 使用 5 折分层交叉验证比较多种模型
- [x] 输出性能报告：`output/pcs_symptom_task/pcs_symptom_prediction_report.md`
- [x] 输出类别分布、混淆矩阵和 one-vs-rest ROC 曲线
- [ ] 在论文 Methods 中说明该任务为探索性补充分析，且症状标签不作为原 PCS 二分类模型输入特征
- [ ] 在 Discussion 中讨论症状亚型样本量小和类别不平衡带来的不确定性

## 2. 论文各 Section 待完成事项

### 2.1 Title

- [ ] 确定论文题目，建议突出 PCS、机器学习、风险预测和可解释性
- [ ] 备选题目 1：基于机器学习的胆囊切除术后综合征发生风险预测模型构建及可解释性分析
- [ ] 备选题目 2：融合多源临床数据的胆囊切除术后综合征风险分层模型构建与 SHAP 解释研究
- [ ] 备选题目 3：胆囊疾病患者术后 PCS 发生的危险因素分析及智能预测模型研究
- [ ] 根据目标期刊风格确定中文题目、英文题目和短标题

### 2.2 Abstract

- [ ] 背景：说明 PCS 的临床负担和术前/围手术期风险识别需求
- [ ] 目的：构建并验证 PCS 发生风险预测模型，识别关键预测因素
- [ ] 方法：说明研究设计、样本量、变量来源、统计分析、机器学习模型、SHAP 和模型评价指标
- [ ] 结果：填入 PCS 发生率、最佳模型 AUC、敏感度、特异度、校准度、DCA 结果和关键变量
- [ ] 结论：说明模型对个体化随访和风险分层的潜在价值
- [ ] 补充关键词：post-cholecystectomy syndrome、machine learning、risk prediction、SHAP、clinical prediction model

### 2.3 Introduction

- [ ] 介绍胆囊切除术后 PCS 的定义、常见症状和发生率
- [ ] 说明 PCS 对生活质量、随访负担和医疗资源使用的影响
- [ ] 总结现有研究多集中于症状描述、发生率和传统危险因素分析
- [ ] 指出当前缺口：缺乏面向 PCS 的术前或围手术期个体化风险预测工具
- [ ] 引出机器学习适合处理多维临床变量和非线性关系
- [ ] 引出 SHAP 在医学 AI 模型透明化和临床可解释性中的价值
- [ ] 明确本研究目的和创新点

### 2.4 Methods - Study Design and Population

- [ ] 明确研究类型：回顾性队列研究或回顾性病例对照研究
- [ ] 说明数据来源医院、时间范围和病例筛选流程
- [ ] 写清纳入标准
- [ ] 写清排除标准
- [ ] 说明伦理审批编号或伦理豁免情况
- [ ] 说明是否获得知情同意豁免
- [ ] 绘制病例筛选流程图，可作为 Figure 1

### 2.5 Methods - Outcome Definition

- [ ] 明确 PCS 的诊断标准
- [ ] 明确 PCS 是否由门诊记录、随访记录、再入院记录或电话随访确定
- [ ] 明确 PCS 判定的时间窗
- [ ] 说明 PCS 发生时间和症状类型为何不作为预测变量
- [ ] 如有多个症状，说明症状分类规则

### 2.6 Methods - Predictors

- [ ] 按类别描述候选预测变量
- [ ] 人口学资料：年龄、性别、职业、身高、体重、BMI、教育水平
- [ ] 生活方式：吸烟、饮酒
- [ ] 临床症状：症状持续时间、腹痛、疼痛频率、放射痛、进食相关疼痛
- [ ] 既往病史：高血压、高血脂、糖尿病、焦虑/抑郁、既往腹部手术史
- [ ] 影像学特征：胆囊壁增厚、最大结石直径、结石数量、结石位置、胆总管直径、脂肪肝、胆囊萎缩
- [ ] 实验室指标：ALT、AST、ALP、GGT、总胆红素、总胆汁酸、总胆固醇、甘油三酯、甲胎蛋白、CA19-9
- [ ] 说明变量采集时间点：术前、入院时或围手术期

### 2.7 Methods - Missing Data and Preprocessing

- [ ] 报告缺失值比例
- [ ] 说明缺失值处理策略
- [ ] 说明连续变量异常值处理策略
- [ ] 说明分类变量编码方式
- [ ] 说明连续变量标准化方式
- [ ] 说明训练/测试划分和交叉验证策略
- [ ] 说明如何避免数据泄漏

### 2.8 Methods - Statistical Analysis

- [ ] 说明连续变量分布判断方法
- [ ] 说明组间比较检验方法
- [ ] 说明单因素 Logistic 回归方法
- [ ] 说明多因素 Logistic 回归变量筛选原则
- [ ] 说明 OR、95% CI 和 P 值报告规范
- [ ] 说明显著性阈值和多重比较处理
- [ ] 说明使用的软件和版本

### 2.9 Methods - Machine Learning and Model Evaluation

- [x] 描述候选模型及选择原因
- [ ] 描述超参数调优方法
- [x] 描述类别不平衡处理方法
- [x] 描述性能指标：AUC、Accuracy、Sensitivity、Specificity、Precision、F1-score、Brier score
- [x] 描述 ROC、校准曲线和 DCA
- [ ] 描述最终模型选择原则
- [ ] 按 TRIPOD/TRIPOD+AI 规范补齐模型报告信息

### 2.10 Methods - Explainability and Risk Stratification

- [x] 描述 SHAP 的用途和分析对象
- [x] 说明 summary plot、importance plot、dependence plot 和 individual explanation 的解释方式
- [ ] 说明风险分层阈值来源
- [ ] 说明如何比较不同风险层的实际 PCS 发生率
- [ ] 描述 PCS 症状类型自由文本标签化和补充多分类预测任务

### 2.11 Results - Cohort Characteristics

- [ ] 报告总样本量、PCS 阳性数、PCS 阴性数和发生率
- [ ] 报告 Table 1 基线特征
- [ ] 描述 PCS 组和非 PCS 组差异明显的变量
- [ ] 结合现有初步结果重点关注肝胆实验室指标和症状变量
- [ ] 放置 Figure 1：病例筛选流程或结局分布

### 2.12 Results - Univariate and Multivariable Analysis

- [ ] 报告单因素 Logistic 回归结果
- [ ] 报告多因素 Logistic 回归结果
- [ ] 形成 Table 2：危险因素分析
- [ ] 说明哪些变量是独立危险因素
- [ ] 区分统计学显著性和临床意义

### 2.13 Results - Model Performance

- [x] 形成 Table 3：不同模型性能比较
- [x] 报告最佳模型及其 AUC、敏感度、特异度、F1-score、Brier score
- [x] 放置 Figure 2：ROC 曲线
- [x] 放置 Figure 3：校准曲线
- [x] 放置 Figure 4：DCA 曲线
- [x] 说明模型在 PCS 阳性识别中的表现，避免只强调 accuracy

### 2.14 Results - SHAP Interpretation

- [x] 形成 Figure 5：SHAP summary plot
- [x] 形成 Figure 6：SHAP feature importance
- [x] 描述对模型贡献最大的变量
- [x] 解释关键变量对 PCS 风险的方向性影响
- [x] 用一例高风险个体解释模型如何形成预测结果
- [ ] 将 SHAP 结果与传统统计结果进行对照

### 2.15 Results - Risk Stratification

- [ ] 报告低、中、高风险组人数和实际 PCS 发生率
- [ ] 形成 Table 4：风险分层结果
- [ ] 形成 Figure 7：风险分层可视化
- [ ] 说明风险分层对术后随访强度和健康宣教的潜在指导意义

### 2.16 Results - PCS Symptom Category Prediction

- [x] 报告原始 PCS 症状文本分布
- [x] 报告症状标签映射和合并类别
- [x] 构建 PCS 症状发生类别多分类预测任务
- [x] 形成补充模型性能表
- [x] 形成症状类别分布图、混淆矩阵和 one-vs-rest ROC 曲线
- [ ] 将该任务明确表述为探索性补充分析

### 2.17 Discussion

- [ ] 总结主要发现：PCS 发生率、关键危险因素、最佳模型表现、SHAP 解释和风险分层
- [ ] 与既往 PCS 发生率和危险因素研究比较
- [ ] 讨论肝胆实验室指标升高与 PCS 风险的可能机制
- [ ] 讨论症状持续时间、疼痛频率、腹痛等临床表现与 PCS 的关系
- [ ] 讨论机器学习模型相较传统 Logistic 回归的优势和局限
- [ ] 讨论 SHAP 对临床理解和模型信任的价值
- [ ] 讨论风险分层在个体化随访中的潜在应用
- [ ] 讨论 PCS 症状类别预测任务的探索性价值和类别不平衡问题
- [ ] 强调本研究仍属于内部验证，不能直接替代临床判断

### 2.18 Limitations

- [ ] 单中心或单数据来源导致外部推广性有限
- [ ] 样本量较小，PCS 阳性仅 70 例，复杂模型可能存在过拟合风险
- [ ] 回顾性研究存在选择偏倚和信息偏倚
- [ ] PCS 诊断和随访时间窗可能存在异质性
- [ ] 缺少外部验证队列
- [ ] 部分潜在影响因素尚未纳入，例如手术方式、术中情况、术后用药、并发症、心理因素量表等
- [ ] PCS 症状类别预测任务中各亚型样本量较小，模型结果仅适合作为探索性发现
- [ ] 模型尚未进行前瞻性临床验证

### 2.19 Conclusion

- [ ] 用 2-4 句话总结研究发现
- [ ] 强调模型可用于识别 PCS 高风险患者
- [ ] 强调 SHAP 提供关键变量解释
- [ ] 强调未来需多中心外部验证和前瞻性评估

### 2.20 References

- [ ] 系统检索 PCS 定义、发生率、症状谱和危险因素文献
- [ ] 系统检索胆囊切除术后并发症机器学习预测模型文献
- [ ] 引用 SHAP 原始论文和 TreeSHAP 论文
- [ ] 引用 TRIPOD/TRIPOD+AI 报告规范
- [ ] 引用 PROBAST/PROBAST+AI 风险偏倚评价工具
- [ ] 优先补充近 5 年系统综述、指南、队列研究和医学 AI 预测模型研究
- [ ] 按目标期刊格式统一参考文献

### 2.21 Tables and Figures

- [ ] Table 1：PCS 组与非 PCS 组基线特征
- [ ] Table 2：单因素和多因素 Logistic 回归
- [x] Table 3：模型性能比较
- [ ] Table 4：风险分层结果
- [ ] Supplementary Table 1：变量定义和数据字典
- [ ] Supplementary Table 2：模型超参数
- [x] Supplementary Table 3：PCS 症状标签映射与症状类别预测性能
- [ ] Figure 1：研究流程图或病例筛选流程图
- [x] Figure 2：ROC 曲线
- [x] Figure 3：校准曲线
- [x] Figure 4：DCA 曲线
- [x] Figure 5：SHAP summary plot
- [x] Figure 6：SHAP feature importance
- [ ] Figure 7：风险分层图
- [x] Supplementary Figure：PCS 症状类别分布、混淆矩阵和 one-vs-rest ROC 曲线

## 3. 投稿前质量控制清单

- [ ] 所有图表编号与正文引用一致
- [ ] 所有表格小数位统一
- [ ] 所有 P 值格式统一，例如 `<0.001` 或三位小数
- [ ] OR、95% CI、AUC、敏感度和特异度均有置信区间时优先报告
- [ ] 明确统计软件、Python 包和版本
- [ ] 明确伦理审批和数据隐私处理
- [ ] 按 TRIPOD/TRIPOD+AI 检查报告完整性
- [ ] 检查是否存在数据泄漏或把结局后变量纳入预测模型的问题
- [ ] 检查图像分辨率是否满足期刊要求，通常建议 300 dpi 以上
- [ ] 检查英文摘要、关键词和缩写表
- [ ] 准备 cover letter
- [ ] 准备 highlights 或 graphical abstract，如目标期刊要求

## 4. 建议优先级

### P0：必须先完成

- [ ] 确认 PCS 诊断标准和随访时间窗
- [ ] 确认纳入排除标准和伦理信息
- [ ] 完成正式基线表
- [ ] 完成 Logistic 回归
- [ ] 完成机器学习模型评估

### P1：决定论文质量

- [x] 完成校准曲线和 DCA
- [x] 完成 SHAP 解释
- [ ] 完成风险分层
- [ ] 按 TRIPOD/TRIPOD+AI 补齐方法学描述

### P2：投稿包装

- [ ] 统一图表风格
- [ ] 补充近 5 年文献
- [ ] 精修 Discussion 和 Limitations
- [ ] 按目标期刊格式排版
