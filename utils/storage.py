from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "mega_sena.csv"

def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_results() -> pd.DataFrame:
    """
    Carrega o CSV histórico da Mega-Sena.
    Trata arquivos vazios e encoding Latin-1 (padrão em arquivos da Caixa).
    """
    ensure_data_dir()
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo {CSV_PATH} não encontrado no repositório.")

    try:
        # arquivos baixados da Caixa costumam vir em Latin-1 / ISO-8859-1
        df = pd.read_csv(CSV_PATH, encoding="latin1")
    except EmptyDataError:
        df = pd.DataFrame(
            columns=["Concurso", "Data Sorteio",
                     "Dezena1", "Dezena2", "Dezena3",
                     "Dezena4", "Dezena5", "Dezena6"]
        )
    return df
