import pandas as pd

def analyze_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame de concursos da Mega-Sena com colunas de dezenas
    (ex.: dezena1..dezena6 ou mais) e devolve a frequência de cada número (1–60).
    """
    # Ajuste aqui os nomes das colunas de dezenas conforme seu CSV
    dezena_cols = [c for c in df.columns if c.lower().startswith("dezena")]
    numeros = range(1, 61)

    freq = {n: 0 for n in numeros}
    for col in dezena_cols:
        valores = df[col].dropna().astype(int)
        for n, count in valores.value_counts().items():
            if n in freq:
                freq[n] += count

    freq_df = pd.DataFrame(
        [{"numero": n, "frequencia": f} for n, f in freq.items()]
    ).sort_values("numero")

    total_dezenas = freq_df["frequencia"].sum()
    if total_dezenas > 0:
        freq_df["frequencia_relativa"] = freq_df["frequencia"] / total_dezenas
    else:
        freq_df["frequencia_relativa"] = 0.0

    return freq_df


def analyze_delay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula atraso (quantos concursos desde a última vez que o número saiu).
    """
    dezena_cols = [c for c in df.columns if c.lower().startswith("dezena")]
    numeros = range(1, 61)

    atrasos = []
    total_concursos = len(df)

    for n in numeros:
        # Índice dos concursos em que o número apareceu
        mask = df[dezena_cols].eq(n).any(axis=1)
        if mask.any():
            ultimo_idx = df[mask].index.max()
            atraso = total_concursos - (df.index.get_loc(ultimo_idx) + 1)
        else:
            atraso = total_concursos  # nunca saiu
        atrasos.append({"numero": n, "atraso": atraso})

    atrasos_df = pd.DataFrame(atrasos).sort_values("numero")
    return atrasos_df
