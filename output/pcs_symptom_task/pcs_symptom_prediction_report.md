# PCS Symptom Occurrence Prediction Task

## Label construction

The original free-text field `PCS症状类型` was converted into structured labels and a merged modeling category. The merged outcome is intended as an additional prediction task and is **not** used as an input feature for the original PCS binary prediction model.

Constructed classes:

| class | display_name | n |
| --- | --- | --- |
| no_pcs | No PCS | 241 |
| gi_dysfunction_dyspepsia | GI dysfunction / dyspepsia phenotype | 26 |
| pain | Pain phenotype | 24 |
| constipation | Constipation phenotype | 19 |
| pcs_unknown | pcs_unknown | 1 |

Raw symptom text distribution:

| raw_symptom | n |
| --- | --- |
| None | 242 |
| 腹痛 | 24 |
| 便秘 | 19 |
| 胃肠功能紊乱 | 12 |
| 消化不良 | 8 |
| 肠道菌群失调 | 3 |
| 慢性胃炎 | 2 |
| 腹胀 | 1 |

## Prediction task

- Task: multiclass prediction of PCS symptom occurrence category.
- Outcome: `pcs_symptom_category`.
- Included classes for modeling: constipation, gi_dysfunction_dyspepsia, no_pcs, pain.
- Excluded from modeling: `pcs_unknown`, because only one PCS-positive patient lacked a symptom text label.
- Predictors: the same preoperative/perioperative variables used in the original PCS prediction task.
- Validation: 5-fold stratified cross-validation.

## Model performance

| model | oof_accuracy | oof_balanced_accuracy | oof_macro_f1 | oof_weighted_f1 | oof_macro_ovr_auc | oof_log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| Gradient Boosting | 0.810 | 0.435 | 0.452 | 0.794 | 0.844 | 0.626 |
| XGBoost | 0.810 | 0.422 | 0.439 | 0.791 | 0.848 | 0.570 |
| SVM | 0.819 | 0.391 | 0.413 | 0.777 | 0.804 | 0.577 |
| Random Forest | 0.797 | 0.391 | 0.381 | 0.777 | 0.849 | 0.613 |
| Multinomial Logistic | 0.677 | 0.367 | 0.349 | 0.710 | 0.673 | 1.246 |

The current top-ranked model by macro F1 is **Gradient Boosting** with OOF macro F1 = **0.452**, OOF balanced accuracy = **0.435**, and OOF macro one-vs-rest AUC = **0.844**.

## Figures

![Class distribution](pcs_symptom_class_distribution.png)

![Confusion matrix](pcs_symptom_confusion_matrix.png)

![One-vs-rest ROC](pcs_symptom_ovr_roc.png)

## Manuscript-ready wording

To further characterize PCS heterogeneity, the free-text PCS symptom field was mapped to structured symptom phenotypes. PCS-negative patients were assigned to `no_pcs`; PCS-positive symptoms were grouped into pain, constipation, and gastrointestinal dysfunction/dyspepsia phenotypes. A secondary multiclass prediction task was then constructed using the same preoperative/perioperative predictors. This task should be interpreted as exploratory because symptom subtype sample sizes are small and external validation is unavailable.
