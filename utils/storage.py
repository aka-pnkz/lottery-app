from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "mega_sena.csv"

def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_results() -> pd.DataFrame:
    """
    Carrega o CSV histórico. Se estiver vazio, devolve DataFrame vazio
    com colunas padrão.
    """
    ensure_data_dir()
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo {CSV_PATH} não encontrado no repositório.")

    try:
        df = pd.read_csv(CSV_PATH)
    except EmptyDataError:
        df = pd.DataFrame(
            columns=["concurso", "data", "dezena1", "dezena2",
                     "dezena3", "dezena4", "dezena5", "dezena6"]
        )
    return df
