from pathlib import Path  # -- OS-independent path handling

# -- Anchor all paths to the project root (where this file lives),
#    regardless of which directory the scripts are launched from.
ROOT_DIR     = Path(__file__).parent.resolve()
DATA_DIR     = ROOT_DIR / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SCRIPTS_DIR  = ROOT_DIR / "scripts"
MODELS_DIR   = ROOT_DIR / "models"

# -- Data files
WEATHER_CSV      = RAW_DIR / "weather_CZ.csv"
ENERGY_CSV       = RAW_DIR / "energy_CZ.csv"
DATA_CSV     = PROCESSED_DIR / "data.csv"
TRAINING_NPZ = PROCESSED_DIR / "data_to_training.npz"

# -- Time series parameters
SEQ_LENGTH   = 42       # input window (days of history the model sees; equivalent to 6 weeks)
AHEAD        = 14       # forecast horizon (days ahead to predict)
BATCH_SIZE   = 32
SEED         = 42

# -- Training parameters
MAX_EPOCHS   = 500
PATIENCE     = int(MAX_EPOCHS*0.15)       # early stopping patience

# -- Data split: last 2 months for validation (includes the last 14 days for comparison)
VALID_MONTHS = 2

# -- Macro-region populations (ČSÚ, 31 Dec 2024)
CB_POPULATION = 2_864_095   # Praha + Středočeský
NB_POPULATION = 2_637_437   # Ústecký + Liberecký + Karlovarský + Královéhradecký + Pardubický
SB_POPULATION = 1_785_514   # Jihočeský + Plzeňský + Vysočina
SM_POPULATION = 1_808_341   # Jihomoravský + Zlínský
NM_POPULATION = 1_814_113   # Moravskoslezský + Olomoucký

# -- Energy crisis period
CRISIS_START = "2022-07-01"
CRISIS_END   = "2023-03-31"
