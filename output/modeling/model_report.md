# PCS Machine Learning Model Report

## Dataset and validation

- Dataset: `dataset.csv`
- Sample size: 311; PCS positive: 70; PCS negative: 241; prevalence: 22.5%
- Predictors: 36 preoperative/perioperative clinical variables
- Validation: 5-fold stratified cross-validation, random state = 42
- Preprocessing: median imputation and standardization for numeric variables; most-frequent imputation and one-hot encoding for categorical variables
- Probability estimates and curves are based on out-of-fold predictions from cross-validation.

## Performance summary

| Model | AUC | Average precision | Sensitivity | Specificity | F1 | Brier score | OOF AUC | OOF Brier | OOF confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.922 ± 0.041 ↑ | 0.845 ± 0.060 ↑ | 0.714 ± 0.134 ↑ | 0.946 ± 0.011 ↑ | 0.747 ± 0.078 ↑ | 0.088 ± 0.010 ↓ | 0.917 | 0.088 | TN=228, FP=13, FN=20, TP=50 |
| Gradient Boosting | 0.899 ± 0.040 ↑ | 0.830 ± 0.042 ↑ | 0.729 ± 0.078 ↑ | 0.950 ± 0.043 ↑ | 0.768 ± 0.051 ↑ | 0.081 ± 0.020 ↓ | 0.895 | 0.081 | TN=229, FP=12, FN=19, TP=51 |
| XGBoost | 0.889 ± 0.056 ↑ | 0.799 ± 0.080 ↑ | 0.757 ± 0.108 ↑ | 0.930 ± 0.031 ↑ | 0.756 ± 0.066 ↑ | 0.091 ± 0.016 ↓ | 0.889 | 0.091 | TN=224, FP=17, FN=17, TP=53 |
| SVM | 0.855 ± 0.060 ↑ | 0.795 ± 0.077 ↑ | 0.657 ± 0.093 ↑ | 0.967 ± 0.028 ↑ | 0.741 ± 0.080 ↑ | 0.089 ± 0.020 ↓ | 0.856 | 0.089 | TN=233, FP=8, FN=24, TP=46 |
| Gaussian NB | 0.857 ± 0.065 ↑ | 0.729 ± 0.045 ↑ | 0.900 ± 0.064 ↑ | 0.398 ± 0.179 ↑ | 0.462 ± 0.080 ↑ | 0.459 ± 0.152 ↓ | 0.853 | 0.459 | TN=96, FP=145, FN=7, TP=63 |
| Logistic Regression | 0.848 ± 0.048 ↑ | 0.818 ± 0.049 ↑ | 0.743 ± 0.064 ↑ | 0.900 ± 0.031 ↑ | 0.713 ± 0.069 ↑ | 0.110 ± 0.015 ↓ | 0.849 | 0.110 | TN=217, FP=24, FN=18, TP=52 |

## Current best model

The top-ranked model by out-of-fold AUC is **Random Forest** with OOF AUC = **0.917**, OOF average precision = **0.842**, and OOF Brier score = **0.088**.

Because PCS prevalence is imbalanced, the report emphasizes sensitivity, precision-recall performance, calibration, and decision curve analysis in addition to AUC.

## Curves

![ROC curves](roc_curve.png)

![Precision-recall curves](pr_curve.png)

![Calibration curves](calibration_curve.png)

![Decision curve analysis](dca_curve.png)

![Metric heatmap](model_metric_heatmap.png)

## Output files

- `model_performance.xlsx`: summary and fold-level performance
- `model_performance_summary.csv`: model-level summary
- `model_performance_by_fold.csv`: fold-level metrics
- `cross_validated_oof_predictions.csv`: out-of-fold predicted probabilities
- `modeling_metadata.json`: reproducibility metadata

## Notes for manuscript writing

- These results are internally validated only and should be described as cross-validated performance.
- Final threshold selection should be clinically justified; the current confusion matrices use threshold = 0.5 for comparability.
- External validation is still required before clinical deployment.
