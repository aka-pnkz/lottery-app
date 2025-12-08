from dataclasses import dataclass

@dataclass
class ApostaPreco:
    dezenas: int
    preco: float

# Tabela de exemplo: ajuste se necessário conforme a tabela oficial mais recente
TABELA_PRECOS = {
    6: 5.00,
    7: 35.00,
    8: 140.00,
    9: 420.00,
    10: 1050.00,
    11: 2310.00,
    12: 4620.00,
    13: 8580.00,
    14: 15015.00,
    15: 22522.50,
    # se quiser, continue até 20 com valores atualizados
}

def preco_por_jogo(dezenas_por_jogo: int) -> float:
    if dezenas_por_jogo not in TABELA_PRECOS:
        raise ValueError("Quantidade de dezenas não suportada na tabela de preços.")
    return float(TABELA_PRECOS[dezenas_por_jogo])

def custo_total(qtd_jogos: int, dezenas_por_jogo: int) -> float:
    return qtd_jogos * preco_por_jogo(dezenas_por_jogo)
