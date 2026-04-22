# Glioma IDH Mutation Prediction From Gene Expression

This project builds a leakage-controlled machine learning pipeline to predict glioma IDH1/IDH2 mutation status from gene expression.

## Data Decision

Source 2 is used as the main training dataset:

- `data_mrna_seq_v2_rsem.txt`: gene expression matrix, used as predictive features.
- `data_mutations.txt`: MAF-like mutation table, used only to create `IDH_status`.
- `data_clinical_sample.txt`: sample metadata, used for matching and cohort summaries.

Source 1 was inspected but excluded from model training:

- `HiSeqV2`: expression data with full overlap for the Source 2 cohort plus additional samples. Mixing it with Source 2 would combine expression scales/cohorts and could create redundancy rather than independent validation.
- `GBMLGG_mc3_gene_level.txt`: gene-level mutation matrix. It is mutation-derived and therefore excluded from predictive features to prevent leakage.

## Leakage Controls

- Labels are `1` when `IDH1` or `IDH2` appears mutated in `data_mutations.txt`, otherwise `0`.
- Mutation-derived variables are never included in the feature matrix.
- Clinical fields are not used as predictors in the default model.
- Train/test split happens before imputation, log transform, scaling, feature selection, tuning, and model fitting.
- Feature selection is inside the sklearn `Pipeline`, so it is refit inside each cross-validation fold.
- Source 2 has one matched sample per patient, so the split is patient-level for this cohort.
- A final test set is held out and used once after model selection.

## File Structure

```text
glioma_idh_project/
  data/
    raw/
    processed/
  models/
    best_idh_expression_pipeline.joblib
  reports/
    source_inspection.json
    clean_dataset_summary.json
    training_report.json
  results/
    model_comparison.json
    test_predictions.csv
    top_genes.csv
    plots/
  src/
    config.py
    data_utils.py
    inspect_sources.py
    prepare_dataset.py
    train.py
    infer.py
  web/
    predictor.py
    styles.css
  app.py
  requirements.txt
```

## Run

```powershell
python src/inspect_sources.py
python src/prepare_dataset.py
python src/train.py
```

For this Codex workspace, the working Python executable is:

```powershell
C:\Users\ahmed\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

Inference on a cBioPortal-style expression matrix:

```powershell
python src/infer.py --input data/raw/data_mrna_seq_v2_rsem.txt --output results/inference_predictions.csv
```

## Web App

The project includes a Gradio web interface on top of the saved model. It does not retrain the model or replace the existing CLI pipeline.

Run locally:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Then open the local URL printed by Gradio, usually `http://127.0.0.1:7860`.

Test with the included sample file:

```text
data/examples/sample_expression_input.csv
```

Accepted upload formats:

- cBioPortal gene-by-sample TSV/TXT with `Hugo_Symbol` and `Entrez_Gene_Id`
- sample-by-gene CSV with a `SAMPLE_ID` column

Web predictions are written to:

```text
results/web_predictions/
```

The web app provides:

- Human-readable predicted labels: `IDH-mutant` and `IDH-wildtype`
- IDH-mutant probability with visual probability bars
- Confidence levels: high, moderate, and borderline/uncertain
- Borderline-case warnings
- Single-sample detail view and batch dashboard view
- Class-count and probability-distribution charts
- Input validation for unsupported files, empty/malformed inputs, duplicate IDs, missing `SAMPLE_ID`, missing model genes, and non-numeric required gene columns
- Enriched prediction CSV export with interpretation and borderline flags
- HTML batch summary report export
- Global top-gene interpretability from `results/top_genes.csv`
- Input format, how-it-works, and model-info panels

The advanced threshold slider changes only the web display/export decision threshold. It does not modify or retrain the saved model.

For later deployment, keep `app.py`, `web/`, `src/`, `models/best_idh_expression_pipeline.joblib`, and `requirements.txt` together in the deployed project. Gradio can be hosted on services such as Hugging Face Spaces, a VM, or a containerized app platform.

## Outputs

- `models/best_idh_expression_pipeline.joblib`: saved preprocessing, feature selection, and trained classifier.
- `reports/training_report.json`: model comparison, train/CV/test checks, and final test metrics.
- `results/top_genes.csv`: top selected genes by model coefficient or feature importance.
- `results/plots/*.png`: ROC, precision-recall, confusion matrix, and top-gene plots.

SHAP is included as an optional dependency. The default run writes model-native interpretability because SHAP was not required to train the model and may need extra installation time.
