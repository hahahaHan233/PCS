# PCS 医学期刊论文项目 TODO

最后更新：2026-06-13

项目主题：基于多源术前/围手术期临床数据构建胆囊切除术后综合征（post-cholecystectomy syndrome, PCS）发生风险预测模型，并结合 SHAP 可解释机器学习、校准评价、决策曲线分析和风险分层，服务于 PCS 高风险患者识别和个体化随访。

## 0. 当前总览

### 已完成

- [x] 原始数据已整理为 `dataset.csv`
- [x] 样本量确认：311 例
- [x] PCS 阳性：70 例
- [x] PCS 阴性：241 例
- [x] PCS 发生率：22.5%
- [x] 已排除身份识别变量：姓名、住院号
- [x] 已排除结局后变量：PCS 发生时间、PCS 症状类型
- [x] 已生成数据字典：`output/dataset_schema_mapping.csv`
- [x] 已生成描述统计：`output/dataset_numeric_summary.csv`
- [x] 已生成分类变量统计：`output/dataset_categorical_summary.csv`
- [x] 已生成数据探索图：`output/dataset_insights/`
- [x] 已生成 PCS 关联探索图表：`output/pcs_association_figures/`
- [x] 已完成机器学习模型训练和 5 折分层交叉验证
- [x] 已完成 ROC、PR、校准曲线、Brier score 和 DCA
- [x] 已完成 voting / stacking / weighted voting 集成架构验证
- [x] 已完成 SHAP 可解释性分析
- [x] 已完成 PCS 症状类型文本标签化
- [x] 已完成 PCS 阳性人群内症状亚型预测探索
- [x] 已完成 PCA、t-SNE、K-means 潜在表型探索
- [x] 已建立 LaTeX demo 项目并插入主要 PDF 图件
- [x] 已更新 `README.md` 为当前项目主页

### 仍未完成

- [ ] 明确 PCS 诊断标准、判定来源和随访时间窗
- [ ] 明确纳入标准、排除标准、研究类型和病例筛选流程
- [ ] 明确伦理审批编号或伦理豁免说明
- [x] 完成正式 Table 1：PCS 组与非 PCS 组基线特征
- [x] 完成单因素和多因素 Logistic 回归
- [x] 完成风险分层实验
- [x] 完成内部稳健性验证：bootstrap 或重复交叉验证
- [x] 完成特征组消融实验
- [ ] 完成论文初稿、图表编号和投稿格式整理

## 1. 当前核心结果

- [x] 建模变量数：36 个术前/围手术期变量
- [x] 当前最佳主模型：Random Forest
- [x] Random Forest OOF AUC：0.917
- [x] Random Forest OOF average precision：0.842
- [x] Random Forest OOF Brier score：0.088
- [x] Random Forest OOF confusion matrix：TN=228, FP=13, FN=20, TP=50
- [x] Weighted soft voting OOF AUC：0.915，接近但未超过 Random Forest
- [x] Soft voting OOF Brier score：0.080，概率校准质量较好
- [x] SHAP top 变量：GGT、ALT、AST、ALP、CA19-9、总胆红素、症状持续时间、总胆汁酸、疼痛频率、年龄
- [x] SHAP 领域贡献最高：Laboratory，其次为 Symptoms
- [x] 症状亚型预测效果较弱，当前应作为探索性补充分析
- [x] 潜在表型聚类 silhouette score = 0.219，应作为 hypothesis-generating analysis

## 2. P0：投稿前必须补齐

这些任务直接决定医学审稿人是否认为研究可信。

### 2.1 临床研究定义

- [ ] 明确研究类型：回顾性队列研究或回顾性病例对照研究
- [ ] 明确数据来源医院、时间范围和病例筛选流程
- [ ] 明确纳入标准
- [ ] 明确排除标准
- [ ] 明确 PCS 诊断标准
- [ ] 明确 PCS 判定来源：门诊记录、随访记录、电话随访、再入院记录或其他来源
- [ ] 明确 PCS 判定时间窗
- [ ] 明确 `PCS发生时间` 的单位，当前按天处理
- [ ] 明确 `PCS症状类型` 是否允许多标签
- [ ] 明确伦理审批编号或伦理豁免信息
- [ ] 明确是否获得知情同意豁免
- [ ] 绘制病例筛选流程图，作为 Figure 1

### 2.2 正式 Table 1

- [x] 新增统一脚本：`scripts/run_manuscript_experiments.py`
- [x] 输出：`output/manuscript_experiments/manuscript_experiment_results.xlsx`
- [x] 输出：`output/manuscript_experiments/tables/baseline_table.csv`
- [x] 连续变量根据分布报告 median (IQR)
- [x] 分类变量报告 n (%)
- [x] 根据分布选择 Mann-Whitney U 检验
- [x] 分类变量选择卡方检验或 Fisher 精确检验
- [x] 报告 PCS 组与非 PCS 组差异
- [x] 输出 FDR 校正结果
- [x] 在 LaTeX PDF 中插入 condensed Table 1

### 2.3 Logistic 回归

- [x] 新增统一脚本：`scripts/run_manuscript_experiments.py`
- [x] 输出：`output/manuscript_experiments/tables/univariate_logistic.csv`
- [x] 输出：`output/manuscript_experiments/tables/multivariable_logistic.csv`
- [x] 完成单因素 Logistic 回归
- [x] 报告 OR、95% CI 和 P 值
- [x] 根据临床意义和 SHAP top 变量筛选多因素模型候选变量
- [x] 检查多重共线性，尤其是 ALT、AST、ALP、GGT、总胆红素、总胆汁酸、CA19-9
- [x] 完成多因素 Logistic 回归
- [x] 报告独立危险因素
- [x] 将 Logistic 回归结果与 SHAP top 变量对照
- [x] 形成论文 Table 2 并插入 LaTeX PDF

### 2.4 风险分层

- [x] 新增统一脚本：`scripts/run_manuscript_experiments.py`
- [x] 使用 OOF 预测概率进行风险分层，避免乐观偏倚
- [x] 确定低/中/高风险阈值：<10%、10-30%、>=30%
- [x] 比较不同风险层的实际 PCS 发生率
- [x] 报告各风险层人数、PCS 例数和 PCS 发生率
- [x] 报告 PPV、Sensitivity、Specificity 等阈值指标
- [x] 输出：`output/manuscript_experiments/tables/risk_stratification.csv`
- [x] 输出：`output/manuscript_experiments/tables/clinical_threshold_table.csv`
- [x] 绘制风险分层柱状图
- [x] 形成论文 Table 4
- [x] 形成论文 Figure 7
- [x] 在 LaTeX PDF 中说明风险分层仍需外部验证

### 2.5 报告规范

- [ ] 按 TRIPOD/TRIPOD+AI 检查报告完整性
- [ ] 报告缺失值比例和处理策略
- [ ] 报告所有模型超参数
- [ ] 报告 Python、scikit-learn、xgboost、shap 等软件版本
- [ ] 说明如何避免数据泄漏
- [ ] 说明最终模型选择原则：AUC、校准、DCA、临床可解释性共同考虑
- [ ] 明确当前仅为内部验证，不能直接用于临床部署

## 3. P1：提高论文贡献和接受概率的补强实验

这些任务不一定是投稿最低要求，但能明显提升“医学 + 计算机交叉”论文的质量。

### 3.1 Bootstrap 或重复交叉验证

- [x] 新增统一脚本：`scripts/run_manuscript_experiments.py`
- [x] 对主模型进行 500 次 bootstrap 内部验证
- [x] 报告 AUC 的 95% CI
- [x] 报告 Brier score 的 95% CI
- [x] 报告 calibration intercept
- [x] 报告 calibration slope
- [x] 输出 optimism-corrected performance
- [x] 输出：`output/manuscript_experiments/tables/bootstrap_summary.csv`

### 3.2 特征组消融实验

- [x] 新增统一脚本：`scripts/run_manuscript_experiments.py`
- [x] 构建仅人口学变量模型
- [x] 构建仅症状变量模型
- [x] 构建仅影像变量模型
- [x] 构建仅实验室指标模型
- [x] 构建症状 + 实验室模型
- [x] 构建全部变量模型
- [x] 比较 AUC、AP、Brier、Sensitivity、Specificity、F1
- [x] 证明实验室与症状+实验室特征携带主要预测信号
- [x] 输出：`output/manuscript_experiments/tables/feature_ablation.csv`
- [x] 绘制消融实验性能柱状图并插入 LaTeX PDF

### 3.3 阈值决策表

- [x] 新增统一脚本：`scripts/run_manuscript_experiments.py`
- [x] 在 10%、20%、30% 风险阈值下计算临床指标
- [x] 报告每 100 名患者中真阳性、假阳性和漏诊数
- [ ] 报告 number needed to follow-up 或类似临床解释指标
- [x] 与 DCA 结果结合解释推荐阈值范围
- [x] 输出：`output/manuscript_experiments/tables/clinical_threshold_table.csv`

### 3.4 校准增强

- [x] 新增统一脚本：`scripts/run_manuscript_experiments.py`
- [x] 比较原始 Random Forest 概率
- [x] 比较 Platt scaling
- [x] 比较 isotonic calibration
- [x] 报告 Brier score、calibration intercept、calibration slope
- [x] 绘制校准曲线对比图并插入 LaTeX PDF
- [x] 在 PDF 中说明 Platt scaling 当前 Brier score 最低

### 3.5 SHAP 稳定性

- [x] 新增统一脚本：`scripts/run_manuscript_experiments.py`
- [x] 在交叉验证折内计算 SHAP 排名
- [x] 报告 top 10 特征排名稳定性
- [x] 输出：`output/manuscript_experiments/tables/shap_stability.csv`
- [ ] 绘制 SHAP 排名稳定性图

### 3.6 敏感性分析

- [x] 新增统一脚本：`scripts/run_manuscript_experiments.py`
- [x] 对实验室极端值进行 winsorization 后重新建模
- [x] 比较 median imputation 与 mean imputation
- [x] 比较 class_weight 与无 class_weight 建模
- [ ] 如样本允许，比较 SMOTE 或欠采样策略
- [x] 报告主模型结论是否稳定

### 3.7 亚组性能

- [x] 新增统一脚本：`scripts/run_manuscript_experiments.py`
- [x] 按性别报告模型性能
- [x] 按年龄组报告模型性能
- [x] 按是否腹痛报告模型性能
- [x] 按 GGT 中位数分层报告模型性能
- [x] 报告各亚组样本量，避免过度解释小样本亚组

### 3.8 PCS 发生时间分析

- [ ] 确认 `PCS发生时间` 的单位和可靠性
- [ ] 如果时间可靠，补充 Kaplan-Meier 或 Cox 回归探索
- [x] 探索高风险组是否更早发生 PCS
- [ ] 当前不建议作为主模型任务，除非随访时间窗完整

## 4. P2：论文写作与投稿包装

### 4.1 主文结构

- [ ] Title：确定中英文题目
- [ ] Abstract：写入样本量、PCS 发生率、最佳模型指标、SHAP 发现
- [ ] Introduction：突出 PCS 术前风险识别缺口
- [ ] Methods：补齐研究设计、结局定义、预测变量、预处理、统计分析、机器学习、SHAP、风险分层
- [ ] Results：按基线特征、Logistic、模型性能、SHAP、风险分层、补充探索组织
- [ ] Discussion：解释肝胆实验室指标、症状负担、模型临床价值和局限性
- [ ] Conclusion：强调内部验证模型和未来外部验证需求

### 4.2 图表清单

- [ ] Figure 1：病例筛选流程图或研究流程图
- [x] Figure 2：ROC 曲线
- [x] Figure 3：校准曲线
- [x] Figure 4：DCA 曲线
- [x] Figure 5：SHAP summary plot
- [x] Figure 6：SHAP feature importance
- [ ] Figure 7：风险分层可视化
- [ ] Table 1：基线特征
- [ ] Table 2：单因素和多因素 Logistic 回归
- [x] Table 3：模型性能比较
- [ ] Table 4：风险分层结果
- [x] Supplementary Figure：症状类别分布、混淆矩阵、one-vs-rest ROC
- [x] Supplementary Figure：PCA / t-SNE 降维图
- [x] Supplementary Figure：潜在表型聚类热图
- [ ] Supplementary Table：变量定义、单位和缺失比例
- [ ] Supplementary Table：模型超参数
- [x] Supplementary Table：症状标签映射与症状类别预测性能
- [x] Supplementary Table：cluster 的 PCS 发生率、症状亚型分布和发生时间统计

### 4.3 文献与规范

- [ ] 系统检索 PCS 定义、发生率、症状谱和危险因素文献
- [ ] 系统检索胆囊切除术后并发症机器学习预测文献
- [ ] 引用 SHAP 原始论文
- [ ] 引用 TreeSHAP 论文
- [ ] 引用 TRIPOD 和 TRIPOD+AI
- [ ] 引用 PROBAST 或 PROBAST+AI
- [ ] 根据目标期刊格式统一参考文献
- [ ] 准备 cover letter
- [ ] 准备 highlights 或 graphical abstract，如目标期刊要求

### 4.4 图件与排版

- [x] 主要图件已保存为 PDF
- [x] LaTeX demo 已使用 PDF 图件
- [ ] 检查所有图件分辨率是否满足 300 dpi 或期刊要求
- [ ] 统一图注，注明 OOF predictions、final fitted model、exploratory analysis 等来源
- [ ] 统一表格小数位
- [ ] 统一 P 值格式，例如 `<0.001` 或三位小数
- [ ] 检查图表编号和正文引用一致

## 5. P3：工程复现与项目整理

- [x] `scripts/build_dataset_artifacts.py`
- [x] `scripts/plot_pcs_associations.py`
- [x] `scripts/train_ml_models.py`
- [x] `scripts/explain_best_model_shap.py`
- [x] `scripts/build_pcs_symptom_task.py`
- [x] `scripts/explore_latent_phenotypes.py`
- [x] `scripts/export_figures_to_pdf.py`
- [x] `scripts/run_manuscript_experiments.py`
- [ ] 拆分可选脚本：`scripts/build_baseline_table.py`
- [ ] 拆分可选脚本：`scripts/run_logistic_regression.py`
- [ ] 拆分可选脚本：`scripts/risk_stratification.py`
- [ ] 拆分可选脚本：`scripts/bootstrap_internal_validation.py`
- [ ] 拆分可选脚本：`scripts/run_feature_ablation.py`
- [ ] 拆分可选脚本：`scripts/clinical_threshold_table.py`
- [ ] 拆分可选脚本：`scripts/calibration_comparison.py`
- [ ] 拆分可选脚本：`scripts/shap_stability.py`
- [ ] 拆分可选脚本：`scripts/sensitivity_analyses.py`
- [ ] 拆分可选脚本：`scripts/subgroup_performance.py`
- [ ] 增加一键运行脚本，例如 `scripts/run_all.py` 或 `Makefile`
- [ ] 保存依赖版本：`requirements.txt`
- [ ] 输出最终项目运行说明

## 6. 建议执行顺序

1. 明确 PCS 诊断标准、纳入排除标准、伦理信息和病例筛选流程。
2. 将已完成的 Table 1、Logistic、风险分层、bootstrap、消融、阈值表、校准增强、SHAP 稳定性、敏感性和亚组结果写入 Results。
3. 将 SHAP 与 Logistic 结果对照写入 Discussion。
4. 把症状亚型和潜在表型放入 Supplementary / Exploratory sections。
5. 根据目标期刊精简 LaTeX 展示版，转换为正式投稿结构。
6. 按 TRIPOD+AI 检查全文。
7. 补充外部验证或前瞻性验证计划描述。

## 7. 最终投稿前检查

- [ ] 研究设计、数据来源和伦理信息完整
- [ ] PCS 结局定义清楚
- [ ] 没有将结局后变量纳入预测模型
- [ ] 缺失值和异常值处理说明清楚
- [ ] 基线表和 Logistic 回归完整
- [ ] 模型性能不仅报告 AUC，也报告校准和 DCA
- [ ] 风险分层具有临床解释
- [ ] SHAP 解释不过度表述为因果关系
- [ ] 症状亚型和聚类明确标注为探索性分析
- [ ] 局限性包含小样本、单中心、回顾性、缺少外部验证
- [ ] 未来工作包含多中心外部验证和前瞻性临床评估
