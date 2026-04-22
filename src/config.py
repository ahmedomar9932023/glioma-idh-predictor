from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = PROJECT_ROOT / "reports"

SOURCE2_CLINICAL = RAW_DIR / "data_clinical_sample.txt"
SOURCE2_EXPRESSION = RAW_DIR / "data_mrna_seq_v2_rsem.txt"
SOURCE2_MUTATIONS = RAW_DIR / "data_mutations.txt"
SOURCE1_EXPRESSION = RAW_DIR / "HiSeqV2"
SOURCE1_MUTATION_MATRIX = RAW_DIR / "GBMLGG_mc3_gene_level.txt"

RANDOM_STATE = 42
IDH_GENES = {"IDH1", "IDH2"}

