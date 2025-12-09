import math

TOTAL_DEZENAS = 60
DEZENAS_SORTEADAS = 6

def comb(n: int, k: int) -> int:
    return math.comb(n, k)

def total_combinacoes_simples() -> int:
    """
    Total de combinações possíveis de 6 dezenas dentre 60.
    """
    return comb(TOTAL_DEZENAS, DEZENAS_SORTEADAS)

def prob_sena(qtd_dezenas_escolhidas: int) -> float:
    """
    Probabilidade teórica de acertar a Sena com uma aposta de N dezenas.
    Fórmula baseada nas combinações possíveis, compatível com tabelas de odds oficiais. [web:32][web:88][web:112]
    """
    if not (6 <= qtd_dezenas_escolhidas <= 20):
        raise ValueError("Quantidade de dezenas deve estar entre 6 e 20.")

    combinacoes_aposta = comb(qtd_dezenas_escolhidas, DEZENAS_SORTEADAS)
    total = total_combinacoes_simples()
    return combinacoes_aposta / total
