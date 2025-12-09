import pandas as pd

def _dezena_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.lower().startswith("dezena")]

def analyze_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Frequência absoluta e relativa de cada dezena (1–60).
    """
    dezena_cols = _dezena_cols(df)
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
    Atraso (quantos concursos desde a última vez que o número saiu).
    """
    dezena_cols = _dezena_cols(df)
    numeros = range(1, 61)

    if df.empty or not dezena_cols:
        return pd.DataFrame([{"numero": n, "atraso": None} for n in numeros])

    atrasos = []
    total_concursos = len(df)

    for n in numeros:
        mask = df[dezena_cols].eq(n).any(axis=1)
        if mask.any():
            ultimo_idx = df[mask].index.max()
            atraso = total_concursos - (df.index.get_loc(ultimo_idx) + 1)
        else:
            atraso = total_concursos
        atrasos.append({"numero": n, "atraso": atraso})

    atrasos_df = pd.DataFrame(atrasos).sort_values("numero")
    return atrasos_df

def analyze_par_impar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribuição de pares x ímpares em cada concurso.
    """
    dezena_cols = _dezena_cols(df)
    if df.empty or not dezena_cols:
        return pd.DataFrame()

    stats = []
    for _, row in df[dezena_cols].dropna(how="all").iterrows():
        dezenas = [int(d) for d in row.dropna()]
        pares = sum(1 for d in dezenas if d % 2 == 0)
        impares = len(dezenas) - pares
        stats.append({"pares": pares, "impares": impares})

    if not stats:
        return pd.DataFrame()

    stats_df = pd.DataFrame(stats)
    distrib = stats_df.value_counts(["pares", "impares"]).reset_index(name="frequencia")
    total = distrib["frequencia"].sum()
    distrib["percentual"] = distrib["frequencia"] / total
    return distrib.sort_values("frequencia", ascending=False)
