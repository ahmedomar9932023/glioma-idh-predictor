from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_utils import finite_numeric_frame, read_expression_matrix
from infer import SUPPORTED_INPUT_SUFFIXES


class InputValidationError(ValueError):
    pass


def _read_header(path: Path, sep: str) -> list[str]:
    try:
        return list(pd.read_csv(path, sep=sep, nrows=0).columns)
    except pd.errors.EmptyDataError as exc:
        raise InputValidationError("The uploaded file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise InputValidationError("The uploaded file appears malformed. Please check the delimiter and columns.") from exc


def validate_and_load_expression(path: Path | str, required_features: list[str]) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        raise InputValidationError(f"Input file was not found: {path}")
    if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        raise InputValidationError("Unsupported file type. Upload a .csv, .tsv, or .txt expression file.")

    tab_header = _read_header(path, sep="\t")
    if {"Hugo_Symbol", "Entrez_Gene_Id"}.issubset(set(tab_header)):
        expression = read_expression_matrix(path)
        if expression.empty:
            raise InputValidationError("No sample columns were found in the gene-by-sample expression matrix.")
        duplicate_samples = expression.index[expression.index.duplicated()].unique().tolist()
        if duplicate_samples:
            raise InputValidationError(
                "Duplicate sample IDs were detected after parsing the gene-by-sample matrix. "
                f"Examples: {', '.join(map(str, duplicate_samples[:5]))}"
            )
        metadata = {
            "input_orientation": "cBioPortal gene-by-sample TSV/TXT",
            "extra_columns_ignored": 0,
            "non_numeric_expected_columns": [],
        }
        return finite_numeric_frame(expression), metadata

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise InputValidationError("The uploaded file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise InputValidationError("The uploaded file appears malformed. Please upload a valid CSV or cBioPortal TSV/TXT.") from exc

    if frame.empty:
        raise InputValidationError("The uploaded file is empty.")
    if "SAMPLE_ID" not in frame.columns:
        raise InputValidationError(
            "Sample-by-gene CSV files must include a SAMPLE_ID column. "
            "For gene-by-sample uploads, use a tab-delimited cBioPortal-style file with Hugo_Symbol and Entrez_Gene_Id."
        )
    if frame["SAMPLE_ID"].isna().any() or frame["SAMPLE_ID"].astype(str).str.strip().eq("").any():
        raise InputValidationError("SAMPLE_ID contains blank or missing values.")

    duplicated = frame.loc[frame["SAMPLE_ID"].duplicated(), "SAMPLE_ID"].astype(str).unique().tolist()
    if duplicated:
        raise InputValidationError(f"Duplicate SAMPLE_ID values were found. Examples: {', '.join(duplicated[:5])}")

    expression = frame.set_index("SAMPLE_ID")
    required_present = [col for col in required_features if col in expression.columns]
    extra_columns = [col for col in expression.columns if col not in required_features]

    non_numeric_expected = []
    for col in required_present:
        converted = pd.to_numeric(expression[col], errors="coerce")
        if converted.isna().sum() > expression[col].isna().sum():
            non_numeric_expected.append(col)
    if non_numeric_expected:
        raise InputValidationError(
            "Some required gene-expression columns contain non-numeric values. "
            f"Examples: {', '.join(non_numeric_expected[:8])}"
        )

    metadata = {
        "input_orientation": "sample-by-gene CSV",
        "extra_columns_ignored": len(extra_columns),
        "non_numeric_expected_columns": non_numeric_expected,
    }
    return finite_numeric_frame(expression), metadata

