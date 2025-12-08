import random
import pandas as pd

TOTAL_DEZENAS = 60

def gerar_jogos(
    qtd_jogos: int,
    dezenas_por_jogo: int,
    estrategia: str = "aleatorio_puro",
    numeros_bloqueados: list[int] | None = None,
) -> pd.DataFrame:
    """
    Gera jogos válidos para a Mega-Sena, respeitando 6 <= dezenas_por_jogo <= 20.
    estrategia: por enquanto só 'aleatorio_puro', mas você pode expandir.
    """
    if not (6 <= dezenas_por_jogo <= 20):
        raise ValueError("dezenas_por_jogo deve estar entre 6 e 20.")

    if qtd_jogos <= 0:
        raise ValueError("qtd_jogos deve ser > 0.")

    numeros_bloqueados = set(numeros_bloqueados or [])
    universo = [n for n in range(1, TOTAL_DEZENAS + 1) if n not in numeros_bloqueados]

    if len(universo) < dezenas_por_jogo:
        raise ValueError("Universo de dezenas disponível é menor que dezenas_por_jogo.")

    jogos = []
    for i in range(qtd_jogos):
        if estrategia == "aleatorio_puro":
            jogo = sorted(random.sample(universo, dezenas_por_jogo))
        else:
            # placeholder para outras estratégias
            jogo = sorted(random.sample(universo, dezenas_por_jogo))

        jogos.append(jogo)

    # Converte lista de listas em DataFrame com colunas dezena1..dezenaN
    max_len = max(len(j) for j in jogos)
    data = []
    for idx, jogo in enumerate(jogos, start=1):
        row = {"jogo": idx}
        for i, d in enumerate(jogo, start=1):
            row[f"dezena{i}"] = d
        # Preenche até max_len (útil se quiser misturar jogos com N diferentes depois)
        for i in range(len(jogo) + 1, max_len + 1):
            row[f"dezena{i}"] = None
        data.append(row)

    df = pd.DataFrame(data)
    return df
