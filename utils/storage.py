from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "mega_sena.csv"

def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_results() -> pd.DataFrame:
    """
    Carrega o CSV histórico.
    """
    ensure_data_dir()
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo {CSV_PATH} não encontrado.")
    df = pd.read_csv(CSV_PATH)
    return df
