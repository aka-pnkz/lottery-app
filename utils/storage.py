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

    - Usa encoding Latin-1 (arquivos oficiais da Caixa).
    - Ignora linhas problemáticas (on_bad_lines='skip').
    - Se o arquivo estiver vazio, devolve DataFrame vazio com colunas padrão.
    """
    ensure_data_dir()
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo {CSV_PATH} não encontrado no repositório.")

    try:from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "mega_sena.csv"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_results() -> pd.DataFrame:
    """
    Carrega o CSV histórico da Mega-Sena.

    - Arquivo separado por ponto e vírgula (;).
    - Usa encoding Latin-1 (arquivos oficiais da Caixa).
    - Ignora linhas problemáticas.
    """
    ensure_data_dir()
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo {CSV_PATH} não encontrado no repositório.")

    try:
        df = pd.read_csv(
            CSV_PATH,
            encoding="latin1",
            sep=";",           # <-- agora usa ponto e vírgula
            on_bad_lines="skip",
        )
    except EmptyDataError:
        df = pd.DataFrame(
            columns=[
                "Concurso",
                "Data Sorteio",
                "Dezena1",
                "Dezena2",
                "Dezena3",
                "Dezena4",
                "Dezena5",
                "Dezena6",
            ]
        )

    return df
