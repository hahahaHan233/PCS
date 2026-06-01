# Latent Phenotype Exploration Report

## Purpose

This exploratory analysis evaluates whether preoperative/perioperative variables reveal latent clinical phenotypes related to PCS occurrence, PCS symptom subtype, or PCS onset time. The analysis is hypothesis-generating and does not provide causal evidence.

## Methods

- Predictors were processed using the same imputation, scaling, and one-hot encoding strategy used in the supervised models.
- Dimensionality reduction was performed using PCA and t-SNE.
- K-means clustering was evaluated for k = 2 to 5 using silhouette score.
- The best K-means solution was k = 2 with silhouette score = 0.219.
- PCS-positive patients with available symptom labels: 69.

## PCA variance

- PC1 explained 12.0% of transformed feature variance.
- PC2 explained 8.4% of transformed feature variance.

## Cluster summary

| cluster | n | pcs_n | pcs_rate | pcs_onset_median_days | pcs_onset_iqr_days |
| --- | --- | --- | --- | --- | --- |
| 0 | 45 | 43 | 95.6% | 43.0 | 66.0 |
| 1 | 266 | 27 | 10.2% | 86.0 | 82.5 |

PCS rates ranged from 10.2% to 95.6%. Median PCS onset time was 43.0-86.0 days across clusters.

## Symptom subtype distribution among PCS-positive patients

| cluster | pcs_symptom_category | n | percent_within_cluster_pcs |
| --- | --- | --- | --- |
| 0 | constipation | 13 | 30.2 |
| 0 | gi_dysfunction_dyspepsia | 14 | 32.6 |
| 0 | pain | 15 | 34.9 |
| 0 | pcs_unknown | 1 | 2.3 |
| 1 | constipation | 6 | 22.2 |
| 1 | gi_dysfunction_dyspepsia | 12 | 44.4 |
| 1 | pain | 9 | 33.3 |

## Figures

![PCA by PCS status](figS4_pca_by_pcs.pdf)

![t-SNE by PCS status](figS4_tsne_by_pcs.pdf)

![PCS-positive t-SNE by symptom subtype](figS4_tsne_pcs_positive_by_symptom.pdf)

![Cluster heatmap](figS5_cluster_heatmap.pdf)

![Cluster PCS rate](figS6_cluster_pcs_rate.pdf)

![Cluster symptom distribution](figS7_cluster_symptom_distribution.pdf)

## Interpretation

These plots should be used to assess whether PCS status or symptom subtypes show visible separation in the preoperative feature space. If subtype separation remains weak, that supports the interpretation that symptom-subtype prediction is difficult with the currently available variables and sample size.
