# Manuscript Enhancement Experiments

## Purpose

This report collects the additional experiments requested to strengthen the clinical and computational evidence of the PCS manuscript.

## Key outputs

- Full workbook: `output/manuscript_experiments/manuscript_experiment_results.xlsx`
- Condensed LaTeX tables: `latex_demo_project/tables/`
- New PDF figures copied to: `latex_demo_project/figures/pdf/`

## Main findings

- Table 1 and logistic regression outputs were generated to provide a conventional clinical-statistical basis.
- Risk stratification was based on Random Forest out-of-fold probabilities with low, intermediate, and high risk strata.
- Bootstrap internal validation used 500 resamples.
- Feature ablation compared demographics, symptoms, imaging, laboratory, symptoms+laboratory, and all-variable models.
- Calibration comparison evaluated raw Random Forest, Platt scaling, and isotonic calibration.
- SHAP stability was assessed across cross-validation folds where the SHAP package was available.
- Sensitivity analyses compared winsorized numeric values, mean imputation, and no class weighting.
- Subgroup analyses evaluated model behavior across sex, age, abdominal pain, and GGT-defined strata.
- PCS onset analysis is descriptive only because censoring and follow-up windows are not fully specified.

## Caution

These analyses remain internally validated. External validation and prospective assessment are still required before clinical deployment.
