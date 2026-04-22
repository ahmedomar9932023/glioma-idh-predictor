from __future__ import annotations

import json

import pandas as pd

from config import (
    IDH_GENES,
    PROCESSED_DIR,
    REPORTS_DIR,
    SOURCE1_EXPRESSION,
    SOURCE1_MUTATION_MATRIX,
    SOURCE2_CLINICAL,
    SOURCE2_EXPRESSION,
    SOURCE2_MUTATIONS,
)
from data_utils import read_cbio_clinical, tcga_patient_id


def header_samples(path, skip_first_cols: int) -> list[str]:
    return list(pd.read_csv(path, sep="\t", nrows=0).columns[skip_first_cols:])


def line_count(path) -> int:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def id_summary(ids: list[str]) -> dict:
    patients = {tcga_patient_id(x) for x in ids}
    duplicates = len(ids) - len(set(ids))
    return {
        "sample_count": len(ids),
        "patient_count": len(patients),
        "duplicate_sample_ids": duplicates,
        "first_five": ids[:5],
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    clinical = read_cbio_clinical(SOURCE2_CLINICAL)
    mrna_samples = header_samples(SOURCE2_EXPRESSION, 2)
    hiseq_samples = header_samples(SOURCE1_EXPRESSION, 1)
    mc3_samples = header_samples(SOURCE1_MUTATION_MATRIX, 1)
    clinical_samples = clinical["SAMPLE_ID"].tolist()

    mut = pd.read_csv(
        SOURCE2_MUTATIONS,
        sep="\t",
        usecols=["Hugo_Symbol", "Tumor_Sample_Barcode"],
        dtype=str,
        low_memory=False,
    )
    mut_samples = sorted(mut["Tumor_Sample_Barcode"].dropna().unique())
    idh_samples = sorted(mut.loc[mut["Hugo_Symbol"].isin(IDH_GENES), "Tumor_Sample_Barcode"].dropna().unique())

    sets = {
        "source2_mrna": set(mrna_samples),
        "source2_clinical": set(clinical_samples),
        "source2_mutations": set(mut_samples),
        "source1_hiseq": set(hiseq_samples),
        "source1_mc3": set(mc3_samples),
    }

    comparisons = {}
    for a, b in [
        ("source2_mrna", "source2_clinical"),
        ("source2_mrna", "source2_mutations"),
        ("source2_mrna", "source1_hiseq"),
        ("source2_mrna", "source1_mc3"),
        ("source1_hiseq", "source1_mc3"),
    ]:
        comparisons[f"{a}__{b}"] = {
            "sample_overlap": len(sets[a] & sets[b]),
            "patient_overlap": len({tcga_patient_id(x) for x in sets[a]} & {tcga_patient_id(x) for x in sets[b]}),
        }

    report = {
        "file_roles": {
            "data_mrna_seq_v2_rsem.txt": "Source 2 cBioPortal RSEM mRNA expression matrix; used as primary predictive feature source.",
            "data_mutations.txt": "Source 2 MAF-like mutation table; used only to create IDH_status label from IDH1/IDH2.",
            "data_clinical_sample.txt": "Source 2 sample metadata; used for sample matching, patient IDs, and summaries, not model features.",
            "HiSeqV2": "Source 1 Xena-style gene expression matrix; inspected but excluded because it overlaps the Source 2 cohort and changes expression scale/cohort.",
            "GBMLGG_mc3_gene_level.txt": "Source 1 gene-level mutation indicator matrix; inspected but excluded from predictive features to avoid mutation-derived leakage.",
        },
        "shapes": {
            "clinical_rows_columns": list(clinical.shape),
            "source2_expression_genes_samples": [line_count(SOURCE2_EXPRESSION) - 1, len(mrna_samples)],
            "source2_mutation_rows_samples": [len(mut), len(mut_samples)],
            "source1_hiseq_genes_samples": [line_count(SOURCE1_EXPRESSION) - 1, len(hiseq_samples)],
            "source1_mc3_genes_samples": [line_count(SOURCE1_MUTATION_MATRIX) - 1, len(mc3_samples)],
        },
        "id_summaries": {
            "source2_mrna": id_summary(mrna_samples),
            "source2_clinical": id_summary(clinical_samples),
            "source2_mutations": id_summary(mut_samples),
            "source1_hiseq": id_summary(hiseq_samples),
            "source1_mc3": id_summary(mc3_samples),
        },
        "overlap": comparisons,
        "label_summary_if_source2_expression_matched": {
            "matched_expression_clinical_samples": len(sets["source2_mrna"] & sets["source2_clinical"]),
            "idh_mutated_samples": len(set(idh_samples) & sets["source2_mrna"] & sets["source2_clinical"]),
            "idh_wildtype_or_no_idh_mutation_rows": len(sets["source2_mrna"] & sets["source2_clinical"]) - len(set(idh_samples) & sets["source2_mrna"] & sets["source2_clinical"]),
        },
        "decision": "Use Source 2 as the main clean dataset. Exclude Source 1 from training because HiSeqV2 is overlapping/redundant expression data and GBMLGG_mc3_gene_level is mutation-derived data unsuitable as input features.",
    }

    (REPORTS_DIR / "source_inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

