import random
import pandas as pd

# universo padrão da Mega-Sena
DEZENAS = list(range(1, 61))


def _gerar_jogo_aleatorio(qtd_dezenas: int) -> list[int]:
    return sorted(random.sample(DEZENAS, qtd_dezenas))


def _conta_par_impar(dezenas: list[int]) -> tuple[int, int]:
    pares = sum(1 for d in dezenas if d % 2 == 0)
    impares = len(dezenas) - pares
    return pares, impares


def _tem_sequencia_longa(dezenas: list[int], tamanho: int = 3) -> bool:
    # verifica se existe pelo menos "tamanho" números consecutivos
    consecutivos = 1
    for a, b in zip(dezenas, dezenas[1:]):
        if b == a + 1:
            consecutivos += 1
            if consecutivos >= tamanho:
                return True
        else:
            consecutivos = 1
    return False


def _jogo_balanceado_par_impar(qtd_dezenas: int) -> list[int]:
    # tenta até achar jogo com 3–3 ou 4–2 entre pares/ímpares (para 6 dezenas)
    tentativas = 0
    while True:
        tentativas += 1
        jogo = _gerar_jogo_aleatorio(qtd_dezenas)
        pares, impares = _conta_par_impar(jogo)
        if qtd_dezenas == 6 and (pares, impares) in [(3, 3), (2, 4), (4, 2)]:
            return jogo
        if tentativas > 500:
            return jogo  # fallback se não achar rápido


def _jogo_faixas(qtd_dezenas: int) -> list[int]:
    faixas = [
        range(1, 21),    # baixa
        range(21, 41),   # média
        range(41, 61),   # alta
    ]
    dezenas_escolhidas: set[int] = set()

    # garante pelo menos 1 por faixa
    for faixa in faixas:
        dezenas_escolhidas.add(random.choice(list(faixa)))

    # completa restante aleatório no universo todo, sem repetir
    while len(dezenas_escolhidas) < qtd_dezenas:
        dezenas_escolhidas.add(random.choice(DEZENAS))

    return sorted(dezenas_escolhidas)


def _jogo_sem_sequencias(qtd_dezenas: int) -> list[int]:
    tentativas = 0
    while True:
        tentativas += 1
        jogo = _gerar_jogo_aleatorio(qtd_dezenas)
        if not _tem_sequencia_longa(jogo, tamanho=3):
            return jogo
        if tentativas > 500:
            return jogo  # fallback


def _jogo_hot_cold(qtd_dezenas: int, freq_df: pd.DataFrame,
                   modo: str = "hot_cold_misto") -> list[int]:
    """
    freq_df: DataFrame com colunas ['numero', 'frequencia'].
    modo:
      - 'hot'  -> prioriza mais frequentes
      - 'cold' -> prioriza menos frequentes
      - 'hot_cold_misto' -> mistura 3 quentes, 2 frias, 1 neutra (para 6 dezenas)
    """
    # ordena por frequência
    df = freq_df.sort_values("frequencia", ascending=False).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return _gerar_jogo_aleatorio(qtd_dezenas)

    tercio = max(n // 3, 1)
    quentes = df.iloc[:tercio]["numero"].tolist()
    frias = df.iloc[-tercio:]["numero"].tolist()
    neutras = df.iloc[tercio:-tercio]["numero"].tolist() if n > 2 * tercio else []

    dezenas_escolhidas: set[int] = set()

    if modo == "hot":
        pool = quentes or DEZENAS
        while len(dezenas_escolhidas) < qtd_dezenas:
            dezenas_escolhidas.add(int(random.choice(pool)))

    elif modo == "cold":
        pool = frias or DEZENAS
        while len(dezenas_escolhidas) < qtd_dezenas:
            dezenas_escolhidas.add(int(random.choice(pool)))

    else:  # hot_cold_misto
        # exemplo para 6 dezenas: 3 quentes, 2 frias, 1 neutra
        alvo_quentes = min(3, qtd_dezenas)
        alvo_frias = min(2, max(qtd_dezenas - alvo_quentes, 0))
        alvo_neutras = max(qtd_dezenas - alvo_quentes - alvo_frias, 0)

        for _ in range(alvo_quentes):
            if quentes:
                dezenas_escolhidas.add(int(random.choice(quentes)))
        for _ in range(alvo_frias):
            if frias:
                dezenas_escolhidas.add(int(random.choice(frias)))
        for _ in range(alvo_neutras):
            if neutras:
                dezenas_escolhidas.add(int(random.choice(neutras)))

        while len(dezenas_escolhidas) < qtd_dezenas:
            dezenas_escolhidas.add(random.choice(DEZENAS))

    return sorted(dezenas_escolhidas)


def gerar_jogos(qtd_jogos: int,
                dezenas_por_jogo: int,
                estrategia: str = "aleatorio_puro",
                freq_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Gera jogos segundo diferentes estratégias.
    estrategia:
      - 'aleatorio_puro'
      - 'balanceado_par_impar'
      - 'faixas'
      - 'sem_sequencias'
      - 'hot'
      - 'cold'
      - 'hot_cold_misto'
    Para estratégias hot/cold, fornecer freq_df com colunas ['numero','frequencia'].
    """
    jogos = []

    for i in range(1, qtd_jogos + 1):
        if estrategia == "aleatorio_puro":
            dezenas = _gerar_jogo_aleatorio(dezenas_por_jogo)

        elif estrategia == "balanceado_par_impar":
            dezenas = _jogo_balanceado_par_impar(dezenas_por_jogo)

        elif estrategia == "faixas":
            dezenas = _jogo_faixas(dezenas_por_jogo)

        elif estrategia == "sem_sequencias":
            dezenas = _jogo_sem_sequencias(dezenas_por_jogo)

        elif estrategia in {"hot", "cold", "hot_cold_misto"} and freq_df is not None:
            dezenas = _jogo_hot_cold(dezenas_por_jogo, freq_df, modo=estrategia)

        else:
            dezenas = _gerar_jogo_aleatorio(dezenas_por_jogo)

        jogos.append([i] + dezenas)

    col_dezenas = [f"dezena{k}" for k in range(1, dezenas_por_jogo + 1)]
    colunas = ["jogo"] + col_dezenas

    return pd.DataFrame(jogos, columns=colunas)
