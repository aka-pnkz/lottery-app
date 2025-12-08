import math

TOTAL_DEZENAS = 60
DEZENAS_SORTEADAS = 6

def comb(n: int, k: int) -> int:
    return math.comb(n, k)

def total_combinacoes_simples() -> int:
    """
    Quantidade total de combinações possíveis em um concurso (6 dezenas dentre 60).
    """
    return comb(TOTAL_DEZENAS, DEZENAS_SORTEADAS)

def prob_sena(qtd_dezenas_escolhidas: int) -> float:
    """
    Probabilidade teórica de acertar a Sena com uma aposta de N dezenas.
    """
    if not (6 <= qtd_dezenas_escolhidas <= 20):
        raise ValueError("Quantidade de dezenas deve estar entre 6 e 20.")

    # Número de combinações de 6 dentro das N dezenas escolhidas
    combinacoes_aposta = comb(qtd_dezenas_escolhidas, DEZENAS_SORTEADAS)
    total = total_combinacoes_simples()
    return combinacoes_aposta / total

def prob_quina(qtd_dezenas_escolhidas: int) -> float:
    """
    Probabilidade aproximada de pelo menos uma quina.
    (Aqui você pode sofisticar depois; por enquanto deixamos como placeholder.)
    """
    # Placeholder simples: você pode substituir por fórmula exata depois
    return 0.0

def prob_quadra(qtd_dezenas_escolhidas: int) -> float:
    """
    Probabilidade aproximada de pelo menos uma quadra.
    (Também pode ser refinado depois.)
    """
    return 0.0
