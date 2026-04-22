from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def tcga_patient_id(sample_id: str) -> str:
    return "-".join(str(sample_id).split("-")[:3])


def read_cbio_clinical(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", dtype=str)


def make_unique_gene_names(symbols: pd.Series, entrez: pd.Series | None = None) -> pd.Series:
    names = symbols.fillna("").astype(str).str.strip()
    if entrez is not None:
        fallback = "ENTREZ_" + entrez.fillna("").astype(str).str.strip()
        names = names.mask(names.eq(""), fallback)
    names = names.mask(names.eq("ENTREZ_"), pd.NA).fillna("UNKNOWN_GENE")
    return names


def read_expression_matrix(path: Path, gene_col: str = "Hugo_Symbol", entrez_col: str = "Entrez_Gene_Id") -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", low_memory=False)
    if gene_col not in raw.columns:
        gene_col = raw.columns[0]
    entrez = raw[entrez_col] if entrez_col in raw.columns else None
    genes = make_unique_gene_names(raw[gene_col], entrez)
    values = raw.drop(columns=[c for c in [gene_col, entrez_col] if c in raw.columns])
    values = values.apply(pd.to_numeric, errors="coerce")
    values.insert(0, "gene", genes)
    grouped = values.groupby("gene", sort=False).mean(numeric_only=True)
    return grouped.T


def read_xena_expression(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", low_memory=False)
    gene_col = raw.columns[0]
    raw[gene_col] = make_unique_gene_names(raw[gene_col])
    values = raw.set_index(gene_col)
    values = values.apply(pd.to_numeric, errors="coerce")
    values = values.groupby(level=0, sort=False).mean()
    return values.T


def label_idh_status(mutations_path: Path, sample_ids: list[str], idh_genes: set[str]) -> pd.DataFrame:
    usecols = ["Hugo_Symbol", "Tumor_Sample_Barcode"]
    mut = pd.read_csv(mutations_path, sep="\t", usecols=usecols, dtype=str, low_memory=False)
    idh_mutated = set(mut.loc[mut["Hugo_Symbol"].isin(idh_genes), "Tumor_Sample_Barcode"].dropna())
    labels = pd.DataFrame({"SAMPLE_ID": sample_ids})
    labels["PATIENT_ID"] = labels["SAMPLE_ID"].map(tcga_patient_id)
    labels["IDH_status"] = labels["SAMPLE_ID"].isin(idh_mutated).astype(int)
    labels["label_source"] = "data_mutations.txt: IDH1/IDH2 presence only"
    return labels


def finite_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)

