from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODELS_DIR, RESULTS_DIR  # noqa: E402
from infer import load_model, prepare_features, required_feature_names  # noqa: E402

from .charts import batch_charts_html
from .components import (
    error_html,
    ready_html,
    results_table_html,
    single_case_html,
    success_html,
    summary_cards_html,
)
from .settings import (
    DEFAULT_DECISION_THRESHOLD,
    MIN_FEATURE_OVERLAP,
    confidence_level,
    interpretation_text,
    is_borderline,
    predicted_status,
)
from .validation import InputValidationError, validate_and_load_expression


WEB_OUTPUT_DIR = RESULTS_DIR / "web_predictions"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_idh_expression_pipeline.joblib"


def uploaded_path(uploaded_file: Any) -> Path:
    if uploaded_file is None:
        raise InputValidationError("Please upload a gene expression file before running prediction.")
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)
    if hasattr(uploaded_file, "name"):
        return Path(uploaded_file.name)
    raise InputValidationError("Could not read the uploaded file path from Gradio.")


def enrich_predictions(raw: pd.DataFrame, threshold: float = DEFAULT_DECISION_THRESHOLD) -> pd.DataFrame:
    enriched = raw.copy()
    statuses = enriched["IDH_mutation_probability"].map(lambda value: predicted_status(value, threshold))
    enriched["predicted"] = [status for status, _ in statuses]
    enriched["predicted_label"] = [label for _, label in statuses]
    enriched["confidence_level"] = enriched["IDH_mutation_probability"].map(lambda value: confidence_level(value, threshold))
    enriched["borderline_flag"] = enriched["IDH_mutation_probability"].map(lambda value: is_borderline(value, threshold))
    enriched["interpretation"] = [
        interpretation_text(label, bool(borderline))
        for label, borderline in zip(enriched["predicted_label"], enriched["borderline_flag"])
    ]
    return enriched[
        [
            "SAMPLE_ID",
            "predicted",
            "predicted_label",
            "IDH_mutation_probability",
            "confidence_level",
            "interpretation",
            "borderline_flag",
        ]
    ]


def summarize(predictions: pd.DataFrame, feature_metadata: dict, threshold: float, validation_metadata: dict) -> dict:
    confidence_order = {"High confidence": 3, "Moderate confidence": 2, "Borderline / uncertain": 1}
    avg_confidence_score = predictions["confidence_level"].map(confidence_order).mean()
    if avg_confidence_score >= 2.5:
        average_confidence = "High confidence"
    elif avg_confidence_score >= 1.5:
        average_confidence = "Moderate confidence"
    else:
        average_confidence = "Borderline / uncertain"

    metadata = dict(feature_metadata)
    metadata.update(validation_metadata)
    metadata.update(
        {
            "n_samples": int(len(predictions)),
            "n_predicted_mutant": int((predictions["predicted_label"] == "IDH-mutant").sum()),
            "n_predicted_wildtype": int((predictions["predicted_label"] == "IDH-wildtype").sum()),
            "n_borderline": int(predictions["borderline_flag"].sum()),
            "mean_probability": float(predictions["IDH_mutation_probability"].mean()),
            "average_confidence": average_confidence,
            "decision_threshold": float(threshold),
        }
    )
    return metadata


def predict_from_upload(uploaded_file: Any, threshold: float = DEFAULT_DECISION_THRESHOLD) -> tuple[pd.DataFrame, dict]:
    model = load_model(DEFAULT_MODEL_PATH)
    required = required_feature_names(model)
    expression, validation_metadata = validate_and_load_expression(uploaded_path(uploaded_file), required)
    X, feature_metadata = prepare_features(
        expression,
        model,
        allow_missing=False,
        min_feature_overlap=MIN_FEATURE_OVERLAP,
    )
    scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    raw = pd.DataFrame({"SAMPLE_ID": X.index.astype(str), "IDH_mutation_probability": scores})
    predictions = enrich_predictions(raw, threshold=threshold)
    metadata = summarize(predictions, feature_metadata, threshold, validation_metadata)
    return predictions, metadata


def save_web_predictions(predictions: pd.DataFrame) -> Path:
    WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = WEB_OUTPUT_DIR / f"idh_predictions_{timestamp}.csv"
    predictions.to_csv(output_path, index=False)
    return output_path


def save_html_report(predictions: pd.DataFrame, metadata: dict) -> Path:
    WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = WEB_OUTPUT_DIR / f"idh_prediction_report_{timestamp}.html"
    html = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Glioma IDH Prediction Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
          h1 {{ margin-bottom: 4px; }}
          .mutant {{ color: #166534; font-weight: 700; }}
          .wildtype {{ color: #92400e; font-weight: 700; }}
          table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
          th, td {{ border: 1px solid #d8dee9; padding: 9px; text-align: left; font-size: 13px; }}
          th {{ background: #f4f7fb; }}
          .note {{ color: #64748b; }}
        </style>
      </head>
      <body>
        <h1>Glioma IDH Prediction Report</h1>
        <p class="note">Research/demo output only. Not intended for clinical diagnosis.</p>
        <ul>
          <li>Total samples: {metadata['n_samples']}</li>
          <li>IDH-mutant: {metadata['n_predicted_mutant']}</li>
          <li>IDH-wildtype: {metadata['n_predicted_wildtype']}</li>
          <li>Borderline cases: {metadata['n_borderline']}</li>
          <li>Mean IDH-mutant probability: {metadata['mean_probability'] * 100:.1f}%</li>
          <li>Decision threshold: {metadata['decision_threshold']:.2f}</li>
        </ul>
        {predictions.to_html(index=False, escape=True)}
      </body>
    </html>
    """
    output_path.write_text(html, encoding="utf-8")
    return output_path


def filter_predictions(predictions: pd.DataFrame | None, label: str, confidence: str, sample_query: str) -> str:
    if predictions is None or predictions.empty:
        return results_table_html(pd.DataFrame())
    filtered = predictions.copy()
    if label and label != "All predictions":
        filtered = filtered[filtered["predicted_label"] == label]
    if confidence and confidence != "All confidence levels":
        filtered = filtered[filtered["confidence_level"] == confidence]
    if sample_query:
        query = str(sample_query).strip().lower()
        if query:
            filtered = filtered[filtered["SAMPLE_ID"].astype(str).str.lower().str.contains(query, regex=False)]
    filtered = filtered.sort_values("IDH_mutation_probability", ascending=False)
    return results_table_html(filtered)


def app_empty_outputs(message: str):
    empty = pd.DataFrame()
    return (
        error_html(message),
        "",
        batch_charts_html(empty),
        results_table_html(empty),
        None,
        None,
        empty,
        "No prediction file generated.",
    )


def run_prediction(uploaded_file: Any) -> tuple[pd.DataFrame, str, str | None]:
    """Backward-compatible wrapper used by older app versions and smoke tests."""
    try:
        predictions, metadata = predict_from_upload(uploaded_file, DEFAULT_DECISION_THRESHOLD)
        output_path = save_web_predictions(predictions)
        display = predictions[["SAMPLE_ID", "predicted_label", "IDH_mutation_probability", "confidence_level"]].copy()
        display["IDH_mutation_probability"] = display["IDH_mutation_probability"].round(4)
        return display, success_html(metadata), str(output_path)
    except Exception as exc:
        return pd.DataFrame(columns=["SAMPLE_ID", "predicted_label", "IDH_mutation_probability", "confidence_level"]), error_html(str(exc)), None


def run_prediction_for_app(uploaded_file: Any, threshold: float = DEFAULT_DECISION_THRESHOLD):
    try:
        predictions, metadata = predict_from_upload(uploaded_file, threshold)
        csv_path = save_web_predictions(predictions)
        report_path = save_html_report(predictions, metadata)
        status = success_html(metadata) + summary_cards_html(metadata)
        details = single_case_html(predictions)
        charts = batch_charts_html(predictions)
        table = results_table_html(predictions.sort_values("IDH_mutation_probability", ascending=False))
        export_note = (
            f"Generated {metadata['n_samples']} prediction rows. "
            f"CSV and HTML report are available in results/web_predictions/."
        )
        return status, details, charts, table, str(csv_path), str(report_path), predictions, export_note
    except InputValidationError as exc:
        return app_empty_outputs(str(exc))
    except FileNotFoundError as exc:
        return app_empty_outputs(str(exc))
    except pd.errors.EmptyDataError:
        return app_empty_outputs("The uploaded file is empty or could not be parsed.")
    except pd.errors.ParserError:
        return app_empty_outputs("The uploaded file appears malformed. Please check delimiter and column structure.")
    except ValueError as exc:
        return app_empty_outputs(str(exc))
    except Exception as exc:
        return app_empty_outputs(f"Unexpected error: {type(exc).__name__}: {exc}")

