from __future__ import annotations

import argparse
from pathlib import Path

import bootstrap  # noqa: F401
import joblib
import pandas as pd

from config import MODELS_DIR, RESULTS_DIR
from data_utils import finite_numeric_frame, read_expression_matrix


SUPPORTED_INPUT_SUFFIXES = {".csv", ".tsv", ".txt"}


def load_model(model_path: Path | str | None = None):
    model_path = Path(model_path) if model_path else MODELS_DIR / "best_idh_expression_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path}")
    return joblib.load(model_path)


def load_input(path: Path | str, require_sample_id: bool = False) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file was not found: {path}")
    if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError("Unsupported file type. Please upload a .tsv, .txt, or .csv expression file.")

    header = pd.read_csv(path, sep="\t", nrows=0)
    if {"Hugo_Symbol", "Entrez_Gene_Id"}.issubset(set(header.columns)):
        return read_expression_matrix(path)

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("The uploaded file is empty.")
    if "SAMPLE_ID" in frame.columns:
        return frame.set_index("SAMPLE_ID")
    if require_sample_id:
        raise ValueError("Sample-by-gene CSV files must include a SAMPLE_ID column.")
    return frame


def required_feature_names(model) -> list[str]:
    try:
        return list(model.named_steps["feature_names"].feature_names_in_)
    except Exception as exc:
        raise ValueError("The model pipeline does not expose the expected feature names.") from exc


def prepare_features(
    expression: pd.DataFrame,
    model,
    *,
    allow_missing: bool = True,
    min_feature_overlap: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    X = finite_numeric_frame(expression)
    required = required_feature_names(model)
    present = [gene for gene in required if gene in X.columns]
    missing = [gene for gene in required if gene not in X.columns]
    overlap = len(present) / len(required) if required else 0.0

    if not allow_missing and missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"The uploaded file is missing {len(missing):,} required gene columns "
            f"({overlap:.1%} overlap). Examples: {preview}"
        )
    if overlap < min_feature_overlap:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"The uploaded file contains too few genes used by the model "
            f"({len(present):,}/{len(required):,}; {overlap:.1%} overlap). "
            f"Missing examples: {preview}"
        )

    for gene in missing:
        X[gene] = pd.NA
    X = X[required]
    metadata = {
        "n_samples": int(X.shape[0]),
        "n_required_features": int(len(required)),
        "n_present_features": int(len(present)),
        "n_missing_features": int(len(missing)),
        "feature_overlap": float(overlap),
    }
    return X, metadata


def predict_expression_file(
    input_path: Path | str,
    model_path: Path | str | None = None,
    *,
    require_sample_id: bool = False,
    allow_missing: bool = True,
    min_feature_overlap: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    model = load_model(model_path)
    expression = load_input(input_path, require_sample_id=require_sample_id)
    X, metadata = prepare_features(
        expression,
        model,
        allow_missing=allow_missing,
        min_feature_overlap=min_feature_overlap,
    )
    scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    preds = model.predict(X)
    output = pd.DataFrame(
        {
            "SAMPLE_ID": X.index.astype(str),
            "IDH_mutation_probability": scores,
            "predicted_IDH_status": preds,
            "predicted": ["IDH-mutant" if int(pred) == 1 else "IDH-wildtype" for pred in preds],
        }
    )
    metadata["n_predicted_mutant"] = int((output["predicted_IDH_status"] == 1).sum())
    metadata["n_predicted_wildtype"] = int((output["predicted_IDH_status"] == 0).sum())
    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict glioma IDH mutation status from expression data.")
    parser.add_argument("--input", required=True, help="Expression file: cBioPortal gene-by-sample TSV or sample-by-gene CSV.")
    parser.add_argument("--output", default=str(RESULTS_DIR / "inference_predictions.csv"), help="Output CSV path.")
    parser.add_argument("--model", default=str(MODELS_DIR / "best_idh_expression_pipeline.joblib"), help="Saved sklearn pipeline.")
    args = parser.parse_args()

    output, _ = predict_expression_file(args.input, args.model)
    output.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
