from __future__ import annotations

import json

import pandas as pd

from config import IDH_GENES, PROCESSED_DIR, REPORTS_DIR, SOURCE2_CLINICAL, SOURCE2_EXPRESSION, SOURCE2_MUTATIONS
from data_utils import finite_numeric_frame, label_idh_status, read_cbio_clinical, read_expression_matrix


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    clinical = read_cbio_clinical(SOURCE2_CLINICAL)
    expression = read_expression_matrix(SOURCE2_EXPRESSION)
    expression = finite_numeric_frame(expression)

    matched_samples = [s for s in clinical["SAMPLE_ID"].tolist() if s in expression.index]
    expression = expression.loc[matched_samples].copy()
    labels = label_idh_status(SOURCE2_MUTATIONS, matched_samples, IDH_GENES)
    labels = labels.merge(clinical, on=["SAMPLE_ID", "PATIENT_ID"], how="left", validate="one_to_one")

    missing_fraction = expression.isna().mean(axis=0)
    keep_genes = missing_fraction[missing_fraction <= 0.05].index
    expression = expression[keep_genes]
    expression = expression.fillna(expression.median(axis=0))

    expression.insert(0, "SAMPLE_ID", expression.index)
    expression.to_csv(PROCESSED_DIR / "source2_expression_matched.csv.gz", index=False, compression="gzip")
    labels.to_csv(PROCESSED_DIR / "source2_labels_metadata.csv", index=False)

    summary = {
        "samples": int(len(labels)),
        "patients": int(labels["PATIENT_ID"].nunique()),
        "genes_after_duplicate_aggregation_and_missing_filter": int(expression.shape[1] - 1),
        "idh_mutated": int(labels["IDH_status"].sum()),
        "idh_wildtype": int((labels["IDH_status"] == 0).sum()),
        "class_balance": labels["IDH_status"].value_counts().sort_index().to_dict(),
        "sample_type_counts": labels["SAMPLE_TYPE"].value_counts(dropna=False).to_dict() if "SAMPLE_TYPE" in labels else {},
        "tumor_type_counts": labels["TUMOR_TYPE"].value_counts(dropna=False).to_dict() if "TUMOR_TYPE" in labels else {},
        "grade_counts": labels["GRADE"].value_counts(dropna=False).to_dict() if "GRADE" in labels else {},
        "note": "Mutation file was used only to create IDH_status. No mutation-derived columns are included in expression features.",
    }
    (REPORTS_DIR / "clean_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

