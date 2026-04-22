from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"
TOP_GENES_PATH = RESULTS_DIR / "top_genes.csv"
TRAINING_REPORT_PATH = REPORTS_DIR / "training_report.json"
CLEAN_DATASET_PATH = REPORTS_DIR / "clean_dataset_summary.json"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_info() -> dict:
    training = read_json(TRAINING_REPORT_PATH)
    clean = read_json(CLEAN_DATASET_PATH)
    metrics = training.get("final_test_metrics", {})
    return {
        "model_type": "Linear SVM",
        "task": "IDH mutation prediction in glioma from gene expression",
        "training_samples": training.get("n_train"),
        "test_samples": training.get("n_test"),
        "cohort_samples": clean.get("samples"),
        "genes": clean.get("genes_after_duplicate_aggregation_and_missing_filter"),
        "roc_auc": metrics.get("roc_auc"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "f1": metrics.get("f1"),
    }


def top_genes(limit: int = 25) -> pd.DataFrame:
    if not TOP_GENES_PATH.exists():
        return pd.DataFrame(columns=["gene", "importance", "signed_value"])
    return pd.read_csv(TOP_GENES_PATH).head(limit)

