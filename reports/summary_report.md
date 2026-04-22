# Glioma IDH Mutation Prediction Summary

## Raw File Inspection

| File | Role | Used? | Reason |
|---|---|---:|---|
| `data_mrna_seq_v2_rsem.txt` | Source 2 cBioPortal RSEM mRNA expression matrix | Yes | Primary gene-expression feature matrix |
| `data_mutations.txt` | Source 2 MAF-like mutation table | Yes, label only | Used only to mark `IDH_status = 1` for IDH1/IDH2 mutation |
| `data_clinical_sample.txt` | Source 2 sample metadata | Yes, matching only | Used for sample/patient matching and cohort summaries |
| `HiSeqV2` | Source 1 Xena-style expression matrix | No | Overlaps all 514 Source 2 expression samples and would mix expression scales/cohorts |
| `GBMLGG_mc3_gene_level.txt` | Source 1 gene-level mutation indicator matrix | No | Mutation-derived data; unsafe as predictive input and redundant for label creation |

## Dataset

- Matched samples: 514
- Matched patients: 514
- Sample type: 514 primary tumors
- Genes after duplicate aggregation and missingness filtering: 20,524
- IDH-mutated: 416
- IDH-wildtype/no IDH1/IDH2 mutation row: 98
- Train/test split: 411 train, 103 held-out test

## Leakage and Overfitting Controls

- Mutation files are used only to create `IDH_status`.
- Predictive features are gene expression only.
- Clinical metadata are not used as predictors in the default model.
- Split is performed before preprocessing.
- Median imputation, log transform, variance filtering, scaling, and SelectKBest feature selection are inside the sklearn `Pipeline`.
- Cross-validation refits feature selection inside each training fold.
- Source 1 mutation matrix is excluded.
- Source 2 has one sample per patient, so the split is patient-level for the matched cohort.

## Model Selection

Candidate models:

- Logistic Regression
- Linear SVM
- RBF SVM
- Random Forest
- ExtraTrees

XGBoost and LightGBM were coded as optional candidates but were not installed in this environment.

The final model is `linear_svm`. Tree models had perfect training scores, so the final selection rule preferred models within 0.005 of the best CV ROC AUC, then ranked by CV balanced accuracy and smaller train/CV gaps.

## Final Linear SVM Performance

Cross-validation on training set:

- ROC AUC: 0.9909 +/- 0.0095
- Average precision: 0.9976 +/- 0.0025
- Balanced accuracy: 0.9729 +/- 0.0175
- F1: 0.9879 +/- 0.0077

Held-out final test set:

- ROC AUC: 0.9380
- Average precision: 0.9672
- Accuracy: 0.9709
- Balanced accuracy: 0.9440
- Precision: 0.9762
- Recall: 0.9880
- F1: 0.9820

Confusion matrix on final test set:

| | Pred WT | Pred IDH-mut |
|---|---:|---:|
| True WT | 18 | 2 |
| True IDH-mut | 1 | 82 |

## Top Genes

Top model-weighted genes include:

`AQP5`, `CALCRL`, `ULBP3`, `TNFAIP6`, `TOM1L1`, `C2orf70`, `TCEA3`, `C2orf27A`, `BMP2`, `FERMT1`, `HMX1`, `COX15`, `NSUN7`, `G0S2`, `ASB13`.

Full ranked genes are in `results/top_genes.csv`.

## Key Outputs

- Saved model pipeline: `models/best_idh_expression_pipeline.joblib`
- Model comparison: `results/model_comparison.json`
- Test predictions: `results/test_predictions.csv`
- Inference smoke-test predictions: `results/inference_predictions.csv`
- Plots: `results/plots/`
- Full JSON training report: `reports/training_report.json`

