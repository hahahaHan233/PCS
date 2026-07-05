# PCS Risk Prediction

一个面向胆囊切除术后综合征（post-cholecystectomy syndrome, PCS）风险预测的机器学习项目。仓库包含数据整理、模型训练、结果导出和一个可直接通过 GitHub Pages 发布的静态网页小程序。

> 说明：本项目用于科研展示和模型工程复现，不用于替代临床诊断、治疗决策或真实患者管理。

## 在线网页

GitHub Pages 发布地址：

[https://hahahahan233.github.io/PCS/](https://hahahahan233.github.io/PCS/)

网页小程序位于 `docs/`，可直接输入变量并在浏览器本地计算 PCS 风险。该网页不需要后端服务，输入数据不会上传到服务器。

## 项目功能

- 从原始 Excel 数据构建标准化建模数据集
- 训练和比较多种机器学习模型
- 生成模型性能、可解释性和辅助分析输出
- 导出浏览器可运行的静态预测模型
- 提供可部署到 GitHub Pages 的前端页面
- 保留 LaTeX 文稿与图表组织目录，方便后续写作整理

## 项目结构

```text
PCS/
├── data/                         # 原始数据目录
├── dataset.csv                   # 建模用标准化数据集
├── docs/                         # GitHub Pages 静态网页
│   ├── index.html                # 网页入口
│   ├── app.js                    # 前端交互与预测逻辑
│   ├── model.js                  # 浏览器可运行的模型参数
│   └── styles.css                # 页面样式
├── scripts/                      # 数据处理、建模和导出脚本
│   ├── build_dataset_artifacts.py
│   ├── train_ml_models.py
│   ├── explain_best_model_shap.py
│   ├── run_manuscript_experiments.py
│   ├── export_figures_to_pdf.py
│   └── export_web_predictor.py   # 导出 docs/model.js
├── output/                       # 脚本生成的结果文件
├── latex_demo_project/           # LaTeX 文稿与图表资源
├── data_check.py                 # 数据检查脚本
├── correlation_visualization.py  # 相关性可视化脚本
├── TODO.md                       # 后续任务记录
└── README.md
```

## 快速开始

### 1. 安装依赖

建议使用 Python 3.10+。

```powershell
python -m pip install pandas numpy scipy scikit-learn xgboost shap matplotlib seaborn openpyxl
```

### 2. 运行建模流程

```powershell
python scripts/build_dataset_artifacts.py
python scripts/train_ml_models.py
python scripts/explain_best_model_shap.py
```

更多补充分析脚本可按需运行：

```powershell
python scripts/run_manuscript_experiments.py
python scripts/build_pcs_symptom_task.py
python scripts/explore_latent_phenotypes.py
python scripts/export_figures_to_pdf.py
```

### 3. 更新网页预测模型

当 `dataset.csv` 或训练逻辑更新后，重新导出网页模型：

```powershell
python scripts/export_web_predictor.py
```

该命令会更新：

```text
docs/model.js
```

### 4. 本地打开网页

直接双击打开：

```text
docs/index.html
```

或在浏览器中访问本地文件路径。

## GitHub Pages 部署

本仓库的网页部署目录为 `docs/`。

在 GitHub 仓库中进入：

```text
Settings -> Pages
```

推荐配置：

```text
Source: Deploy from a branch
Branch: main
Folder: /docs
```

保存后等待 GitHub 构建完成，即可通过项目 Pages 地址访问。

## 数据与隐私

- 原始数据位于 `data/`，请根据实际隐私要求决定是否公开。
- 静态网页中的 `docs/model.js` 包含模型参数，不包含原始患者记录。
- 如果仓库设为公开，`docs/` 中的模型文件和前端代码也会公开。
- 任何临床部署前都需要进一步的外部验证、模型校准、伦理审批和隐私审查。

## 主要脚本说明

| 脚本 | 用途 |
| --- | --- |
| `scripts/build_dataset_artifacts.py` | 构建标准化数据集和数据字典 |
| `scripts/train_ml_models.py` | 训练机器学习模型并输出验证结果 |
| `scripts/explain_best_model_shap.py` | 生成模型解释相关输出 |
| `scripts/run_manuscript_experiments.py` | 运行补充实验和表格生成流程 |
| `scripts/export_figures_to_pdf.py` | 整理图表到 LaTeX 项目目录 |
| `scripts/export_web_predictor.py` | 导出浏览器可运行的静态预测模型 |

## 许可与使用

本项目当前主要用于个人科研与展示。若用于论文、课题、演示或二次开发，请在使用前确认数据授权、隐私合规和模型适用边界。
