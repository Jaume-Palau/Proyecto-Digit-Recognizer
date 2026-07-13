from pathlib import Path


# Raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parents[1]

# Datos
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

COMPETITION_DATA_DIR = (
    RAW_DATA_DIR / "conquerx-b05-lec01-digit-recognizer"
)

TRAIN_CSV = COMPETITION_DATA_DIR / "train.csv"
TEST_CSV = COMPETITION_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = COMPETITION_DATA_DIR / "sample_submission.csv"

# Salidas
OUTPUTS_DIR = ROOT_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
MODELS_DIR = OUTPUTS_DIR / "models"
PICKLE_DIR = OUTPUTS_DIR / "pickle"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

# Código y experimentación
SRC_DIR = ROOT_DIR / "src"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SCRIPTS_DIR = ROOT_DIR / "scripts"