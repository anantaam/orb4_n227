"""Central path configuration for the ORB strategy package."""
from pathlib import Path

# Package root = same directory as this file (flat layout)
PKG_ROOT    = Path(__file__).parent

SCRIPTS_DIR = PKG_ROOT / "scripts"
CONFIG_DIR  = PKG_ROOT / "config"
CACHE_DIR   = PKG_ROOT / "cache"
RESULTS_DIR = PKG_ROOT / "results"

FNOLIST_CSV    = CONFIG_DIR / "index_constituents" / "nsefnolist.csv"
CONFIG_YAML    = CONFIG_DIR / "config.yaml"
PICKLE_PATH    = CACHE_DIR  / "daily_metrics.pkl"
BAR_CACHE_PATH = CACHE_DIR  / "bar_cache.pkl"

# Raw 1-min bar data (CSV fallback — superseded by DB loader)
DATA_DIR = Path(r"D:\orb2\1mindata")

# PostgreSQL — nse database (trust auth, no password)
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "nse"
DB_USER = "postgres"
PSQL    = Path(r"D:\pgsql\pgsql\bin\psql.exe")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
