from dataclasses import dataclass

@dataclass
class ApostaPreco:
    dezenas: int
    preco: float

# Valores de exemplo; confira com a tabela oficial mais recente.
TABELA_PRECOS = {
    6: 6.00,
    7: 42.00,
    8: 168.00,
    9: 504.00,
    10: 1260.00,
    11: 2772.00,
    12: 5544.00,
    13: 10296.00,
    14: 18018.00,
    15: 30030.00,
    16: 48048.00,
    17: 72072.00,
    18: 108108.00,
    19: 162162.00,
    20: 216216.00,
}


def preco_por_jogo(dezenas_por_jogo: int) -> float:
    if dezenas_por_jogo not in TABELA_PRECOS:
        raise ValueError("Quantidade de dezenas não suportada na tabela de preços.")
    return float(TABELA_PRECOS[dezenas_por_jogo])

def custo_total(qtd_jogos: int, dezenas_por_jogo: int) -> float:
    return qtd_jogos * preco_por_jogo(dezenas_por_jogo)
