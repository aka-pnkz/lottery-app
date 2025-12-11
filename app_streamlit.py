import itertools
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ==========================
# CONFIG GERAL
# ==========================
st.set_page_config(
    page_title="Mega Sena Helper",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_PATH = "historico_mega_sena.csv"

PRECO_SIMPLES_6_DEFAULT = 7.50
TOTAL_COMBS_MEGA = math.comb(60, 6)


# ==========================
# CARGA E ESTATÍSTICAS BÁSICAS
# ==========================
@st.cache_data
def carregar_concursos(caminho_csv: str = CSV_PATH) -> pd.DataFrame:
    """
    Espera CSV com colunas:
    concurso;data;d1;d2;d3;d4;d5;d6
    """
    df = pd.read_csv(caminho_csv, sep=";")
    df["concurso"] = df["concurso"].astype(int)
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    dezenas_cols = ["d1", "d2", "d3", "d4", "d5", "d6"]
    for c in dezenas_cols:
        df[c] = df[c].astype(int)
    return df


def calcular_frequencias(df: pd.DataFrame) -> pd.DataFrame:
    dezenas_cols = ["d1", "d2", "d3", "d4", "d5", "d6"]
    todas = df[dezenas_cols].values.ravel()
    freq = pd.Series(todas).value_counts().sort_index()
    freq_df = freq.reset_index()
    freq_df.columns = ["dezena", "frequencia"]
    freq_df["dezena"] = freq_df["dezena"].astype(int)
    return freq_df


def pares_impares(jogo: list[int]) -> tuple[int, int]:
    pares = sum(1 for d in jogo if d % 2 == 0)
    impares = len(jogo) - pares
    return pares, impares


def baixos_altos(jogo: list[int]) -> tuple[int, int]:
    baixos = sum(1 for d in jogo if 1 <= d <= 30)
    altos = len(jogo) - baixos
    return baixos, altos


def tem_sequencia_longa(jogo: list[int], limite: int = 3) -> bool:
    jogo_ord = sorted(jogo)
    atual = 1
    for i in range(1, len(jogo_ord)):
        if jogo_ord[i] == jogo_ord[i - 1] + 1:
            atual += 1
            if atual >= limite:
                return True
        else:
            atual = 1
    return False


# ==========================
# ESTRATÉGIAS DE GERAÇÃO
# ==========================
def gerar_aleatorio_puro(qtd_jogos: int, tam_jogo: int) -> list[list[int]]:
    universe = np.arange(1, 61)
    jogos = []
    for _ in range(qtd_jogos):
        dezenas = np.random.choice(universe, size=tam_jogo, replace=False)
        jogos.append(sorted(dezenas.tolist()))
    return jogos


def gerar_balanceado_par_impar(qtd_jogos: int, tam_jogo: int) -> list[list[int]]:
    universe = np.arange(1, 61)
    jogos = []

    for _ in range(qtd_jogos):
        tentativas = 0
        while True:
            tentativas += 1
            dezenas = np.random.choice(universe, size=tam_jogo, replace=False)
            pares, impares = pares_impares(dezenas)
            if tam_jogo == 6:
                if (pares, impares) in [(3, 3), (4, 2), (2, 4)]:
                    jogos.append(sorted(dezenas.tolist()))
                    break
            else:
                if pares not in (0, tam_jogo) and impares not in (0, tam_jogo):
                    jogos.append(sorted(dezenas.tolist()))
                    break

            if tentativas > 50:
                jogos.append(sorted(dezenas.tolist()))
                break

    return jogos


def gerar_setorial(qtd_jogos: int, tam_jogo: int) -> list[list[int]]:
    s1 = np.arange(1, 21)
    s2 = np.arange(21, 41)
    s3 = np.arange(41, 61)

    jogos = []
    for _ in range(qtd_jogos):
        if tam_jogo <= 6:
            q1, q2, q3 = 2, 2, tam_jogo - 4
        else:
            base = tam_jogo // 3
            resto = tam_jogo % 3
            q1 = base + (1 if resto > 0 else 0)
            q2 = base + (1 if resto > 1 else 0)
            q3 = base

        dezenas = np.concatenate(
            [
                np.random.choice(s1, size=min(q1, len(s1)), replace=False),
                np.random.choice(s2, size=min(q2, len(s2)), replace=False),
                np.random.choice(s3, size=min(q3, len(s3)), replace=False),
            ]
        )

        if len(dezenas) > tam_jogo:
            dezenas = np.random.choice(dezenas, size=tam_jogo, replace=False)

        jogos.append(sorted(dezenas.tolist()))

    return jogos


def gerar_quentes_frias_mix(
    qtd_jogos: int,
    tam_jogo: int,
    freq_df: pd.DataFrame,
    proporcao: tuple[int, int, int] = (3, 2, 1),
) -> list[list[int]]:
    total_quentes, total_frias, total_neutras = proporcao

    freq_ord = freq_df.sort_values("frequencia", ascending=False)
    quentes = freq_ord["dezena"].values[:20]
    frias = freq_ord.sort_values("frequencia", ascending=True)["dezena"].values[:20]
    neutras = np.setdiff1d(np.arange(1, 61), np.union1d(quentes, frias))

    jogos = []
    for _ in range(qtd_jogos):
        q_quentes = min(total_quentes, tam_jogo)
        q_frias = min(total_frias, max(0, tam_jogo - q_quentes))
        q_neutras = max(0, tam_jogo - q_quentes - q_frias)

        dezenas = []

        if len(quentes) > 0 and q_quentes > 0:
            dezenas.extend(
                np.random.choice(
                    quentes, size=min(q_quentes, len(quentes)), replace=False
                )
            )
        if len(frias) > 0 and q_frias > 0:
            dezenas.extend(
                np.random.choice(
                    frias, size=min(q_frias, len(frias)), replace=False
                )
            )
        if len(neutras) > 0 and q_neutras > 0:
            dezenas.extend(
                np.random.choice(
                    neutras, size=min(q_neutras, len(neutras)), replace=False
                )
            )

        if len(dezenas) < tam_jogo:
            universe = np.setdiff1d(np.arange(1, 61), dezenas)
            extra = np.random.choice(
                universe, size=tam_jogo - len(dezenas), replace=False
            )
            dezenas = np.concatenate([dezenas, extra])

        jogos.append(sorted(list(map(int, dezenas))))

    return jogos


def gerar_sem_sequencias(
    qtd_jogos: int,
    tam_jogo: int,
    limite_sequencia: int = 3,
) -> list[list[int]]:
    universe = np.arange(1, 61)
    jogos = []

    for _ in range(qtd_jogos):
        tentativas = 0
        while True:
            tentativas += 1
            dezenas = np.random.choice(universe, size=tam_jogo, replace=False)
            if not tem_sequencia_longa(dezenas, limite=limite_sequencia):
                jogos.append(sorted(dezenas.tolist()))
                break
            if tentativas > 100:
                jogos.append(sorted(dezenas.tolist()))
                break

    return jogos


def gerar_wheeling_simples(
    base_dezenas: list[int],
    max_jogos: int,
) -> list[list[int]]:
    """
    Desdobramento simples: gera combinações de 6 dezenas dentro de uma base fixa. [web:180][web:181]
    """
    base_ordenada = sorted(set(base_dezenas))
    if len(base_ordenada) < 6:
        return []

    todas_combinacoes = itertools.combinations(base_ordenada, 6)
    jogos = []
    for comb in todas_combinacoes:
        jogos.append(list(comb))
        if len(jogos) >= max_jogos:
            break
    return jogos


def formatar_jogo(jogo: list[int]) -> str:
    return " - ".join(f"{d:02d}" for d in jogo)


# ==========================
# CUSTO, COBERTURA E PROBABILIDADE
# ==========================
def preco_aposta_megasena(n_dezenas: int, preco_6: float) -> float:
    """
    Valor aproximado de UMA aposta com n dezenas: C(n, 6) * preço_base. [web:96][web:186]
    """
    if n_dezenas < 6:
        raise ValueError("Mega-Sena exige pelo menos 6 dezenas.")
    comb = math.comb(n_dezenas, 6)
    return comb * preco_6


def calcular_custo_total(jogos: list[list[int]], preco_6: float) -> float:
    total = 0.0
    for jogo in jogos:
        n = len(jogo)
        total += preco_aposta_megasena(n, preco_6)
    return total


def cobertura_jogo(jogo: list[int]) -> tuple[int, float]:
    """
    Retorna (comb_6, fator) onde comb_6 = C(n,6) e fator = comb_6 / C(60,6). [web:98][web:186]
    """
    n = len(jogo)
    if n < 6:
        return 0, 0.0
    comb_6 = math.comb(n, 6)
    fator = comb_6 / TOTAL_COMBS_MEGA
    return comb_6, fator


def prob_sena_pacote(jogos: list[list[int]]) -> float:
    """
    Probabilidade aproximada de acertar a sena com pelo menos 1 jogo. [web:98][web:106]
    """
    probs = []
    for jogo in jogos:
        n = len(jogo)
        if n < 6:
            continue
        cobert = math.comb(n, 6)
        p = cobert / TOTAL_COMBS_MEGA
        probs.append(p)

    prob_nao_acontece = 1.0
    for p in probs:
        prob_nao_acontece *= (1.0 - p)

    return 1.0 - prob_nao_acontece


# ==========================
# FILTROS AVANÇADOS
# ==========================
def filtrar_jogos(
    jogos: list[list[int]],
    dezenas_fixas: list[int] | None = None,
    dezenas_proibidas: list[int] | None = None,
    soma_min: int | None = None,
    soma_max: int | None = None,
) -> list[list[int]]:
    dezenas_fixas_set = set(dezenas_fixas or [])
    dezenas_proibidas_set = set(dezenas_proibidas or [])

    filtrados: list[list[int]] = []
    for jogo in jogos:
        s = set(jogo)

        if dezenas_fixas_set and not dezenas_fixas_set.issubset(s):
            continue

        if dezenas_proibidas_set and (dezenas_proibidas_set & s):
            continue

        soma = sum(jogo)
        if soma_min is not None and soma < soma_min:
            continue
        if soma_max is not None and soma > soma_max:
            continue

        filtrados.append(jogo)

    return filtrados


# ==========================
# SIMULAÇÃO DE RESULTADOS
# ==========================
def simular_premios(
    jogos: list[list[int]],
    dezenas_sorteadas: list[int],
) -> pd.DataFrame:
    dezenas_s = set(dezenas_sorteadas)
    linhas = []
    for i, jogo in enumerate(jogos, start=1):
        acertos = len(set(jogo) & dezenas_s)
        linhas.append(
            {
                "jogo_id": i,
                "jogo": formatar_jogo(jogo),
                "acertos": acertos,
            }
        )
    df = pd.DataFrame(linhas)
    return df


def simular_multi_concursos(
    jogos: list[list[int]],
    concursos: pd.DataFrame,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simula os jogos contra vários concursos do histórico e devolve:
    - df_dist: distribuição global de acertos
    - df_por_concurso: resumo por concurso (máx acertos, quantas quadras/quinas/senas). [web:228][web:236]
    """
    dezenas_cols = [f"d{i}" for i in range(1, 7)]
    tot_acertos_global: dict[int, int] = {}
    resumo_concursos: list[dict] = []

    total_jogos = len(jogos)
    total_concursos = len(concursos)

    progress_bar = None
    if show_progress:
        progress_bar = st.progress(0, text="Simulando concursos...")

    for idx, (_, row) in enumerate(concursos.iterrows(), start=1):
        sorteio = set(int(row[c]) for c in dezenas_cols)
        contagem_concurso: dict[int, int] = {}

        for jogo in jogos:
            acertos = len(set(jogo) & sorteio)
            tot_acertos_global[acertos] = tot_acertos_global.get(acertos, 0) + 1
            contagem_concurso[acertos] = contagem_concurso.get(acertos, 0) + 1

        max_acertos = max(contagem_concurso.keys()) if contagem_concurso else 0
        resumo_concursos.append(
            {
                "concurso": int(row["concurso"]),
                "data": row.get("data"),
                "max_acertos_no_concurso": max_acertos,
                "qtd_quadras": contagem_concurso.get(4, 0),
                "qtd_quinas": contagem_concurso.get(5, 0),
                "qtd_senas": contagem_concurso.get(6, 0),
            }
        )

        if show_progress and progress_bar is not None and total_concursos > 0:
            progress = int(idx / total_concursos * 100)
            progress_bar.progress(
                progress,
                text=f"Simulando concursos... {progress}%",
            )

    if show_progress and progress_bar is not None:
        progress_bar.empty()

    total_comb = total_jogos * total_concursos
    linhas = []
    for acertos, qtd in sorted(tot_acertos_global.items()):
        linhas.append(
            {
                "acertos": acertos,
                "qtd_jogos_concurso": qtd,
                "total_jogos_x_concursos": total_comb,
                "proporcao": qtd / total_comb if total_comb > 0 else 0,
            }
        )
    df_dist = pd.DataFrame(linhas)

    df_por_concurso = pd.DataFrame(resumo_concursos)
    if not df_por_concurso.empty:
        df_por_concurso["teve_premio_4_ou_mais"] = (
            df_por_concurso["qtd_quadras"]
            + df_por_concurso["qtd_quinas"]
            + df_por_concurso["qtd_senas"]
        ) > 0

    return df_dist, df_por_concurso


# ==========================
# ANÁLISE HISTÓRICA
# ==========================
def calcular_atraso(freq_df: pd.DataFrame, df_concursos: pd.DataFrame) -> pd.DataFrame:
    dezenas_cols = ["d1", "d2", "d3", "d4", "d5", "d6"]
    ultimo_concurso = {}

    for _, row in df_concursos[["concurso"] + dezenas_cols].iterrows():
        conc = int(row["concurso"])
        for d in row[dezenas_cols]:
            d = int(d)
            ultimo_concurso[d] = conc

    max_concurso = int(df_concursos["concurso"].max())

    atraso_list = []
    for dezena in range(1, 61):
        freq_row = freq_df.loc[freq_df["dezena"] == dezena]
        freq = int(freq_row["frequencia"].iloc[0]) if not freq_row.empty else 0
        ult = ultimo_concurso.get(dezena, None)
        atraso = None if ult is None else max_concurso - ult
        atraso_list.append(
            {
                "dezena": dezena,
                "frequencia": freq,
                "ultimo_concurso": ult,
                "atraso_atual": atraso,
            }
        )

    atraso_df = pd.DataFrame(atraso_list)
    return atraso_df


def calcular_padroes_par_impar_baixa_alta(
    df_concursos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dezenas_cols = ["d1", "d2", "d3", "d4", "d5", "d6"]
    registros = []

    for _, row in df_concursos[["concurso"] + dezenas_cols].iterrows():
        dezenas = [int(row[c]) for c in dezenas_cols]
        pares = sum(1 for d in dezenas if d % 2 == 0)
        impares = len(dezenas) - pares
        baixos = sum(1 for d in dezenas if 1 <= d <= 30)
        altos = len(dezenas) - baixos
        registros.append(
            {
                "concurso": int(row["concurso"]),
                "pares": pares,
                "impares": impares,
                "baixos": baixos,
                "altos": altos,
            }
        )

    df_padroes = pd.DataFrame(registros)

    dist_par_impar = (
        df_padroes.groupby(["pares", "impares"])
        .size()
        .reset_index(name="qtd")
        .sort_values("qtd", ascending=False)
        .reset_index(drop=True)
    )

    dist_baixa_alta = (
        df_padroes.groupby(["baixos", "altos"])
        .size()
        .reset_index(name="qtd")
        .sort_values("qtd", ascending=False)
        .reset_index(drop=True)
    )

    return df_padroes, dist_par_impar, dist_baixa_alta


def calcular_somas(df_concursos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dezenas_cols = ["d1", "d2", "d3", "d4", "d5", "d6"]
    df = df_concursos.copy()
    df["soma"] = df[dezenas_cols].sum(axis=1)

    bins = [0, 120, 150, 180, 210, 240, 300]
    labels = ["0-120", "121-150", "151-180", "181-210", "211-240", "241-300"]
    df["faixa_soma"] = pd.cut(df["soma"], bins=bins, labels=labels, right=True)

    dist_faixas = (
        df["faixa_soma"]
        .value_counts(dropna=False)
        .sort_index()
        .reset_index()
        .rename(columns={"index": "faixa_soma", "faixa_soma": "qtd"})
    )

    return df[["concurso", "soma", "faixa_soma"]], dist_faixas


def calcular_pares_trios(
    df_concursos: pd.DataFrame, top_n: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dezenas_cols = ["d1", "d2", "d3", "d4", "d5", "d6"]
    pares_contagem = {}
    trios_contagem = {}

    for _, row in df_concursos[dezenas_cols].iterrows():
        dezenas = sorted(int(row[c]) for c in dezenas_cols)

        for a, b in itertools.combinations(dezenas, 2):
            key = f"{a:02d}-{b:02d}"
            pares_contagem[key] = pares_contagem.get(key, 0) + 1

        for a, b, c in itertools.combinations(dezenas, 3):
            key = f"{a:02d}-{b:02d}-{c:02d}"
            trios_contagem[key] = trios_contagem.get(key, 0) + 1

    df_pares = (
        pd.DataFrame([{"par": k, "qtd": v} for k, v in pares_contagem.items()])
        .sort_values("qtd", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    df_trios = (
        pd.DataFrame([{"trio": k, "qtd": v} for k, v in trios_contagem.items()])
        .sort_values("qtd", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return df_pares, df_trios


def calcular_ciclos(df_concursos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Calcula ciclos de dezenas: cada ciclo começa em um concurso e termina
    quando todas as 60 dezenas já saíram ao menos uma vez. [web:313][web:314][web:315]
    """
    dezenas_cols = ["d1", "d2", "d3", "d4", "d5", "d6"]
    df_ord = df_concursos.sort_values("concurso").reset_index(drop=True)

    ciclos = []
    dezenas_vistas: set[int] = set()
    inicio_ciclo = None

    for _, row in df_ord.iterrows():
        conc = int(row["concurso"])
        if inicio_ciclo is None:
            inicio_ciclo = conc

        for c in dezenas_cols:
            dezenas_vistas.add(int(row[c]))

        if len(dezenas_vistas) == 60:
            ciclos.append(
                {
                    "ciclo_id": len(ciclos) + 1,
                    "inicio_concurso": inicio_ciclo,
                    "fim_concurso": conc,
                    "qtd_concursos": conc - inicio_ciclo + 1,
                }
            )
            dezenas_vistas = set()
            inicio_ciclo = None

    df_ciclos = pd.DataFrame(ciclos)

    ciclo_atual = {
        "inicio_concurso": inicio_ciclo,
        "qtd_concursos": None,
        "dezenas_faltando": [],
    }
    if inicio_ciclo is not None:
        max_conc = int(df_ord["concurso"].max())
        ciclo_atual["qtd_concursos"] = max_conc - inicio_ciclo + 1
        dezenas_faltando = sorted(set(range(1, 61)) - dezenas_vistas)
        ciclo_atual["dezenas_faltando"] = dezenas_faltando

    return df_ciclos, ciclo_atual


def pagina_analises(df_concursos: pd.DataFrame, freq_df: pd.DataFrame) -> None:
    st.title("Análises estatísticas da Mega-Sena")
    st.caption(
        "Use esta página para entender o comportamento histórico das dezenas. "
        "As análises são descritivas e não garantem aumento de chance em sorteios futuros."
    )

    (
        tab_freq,
        tab_padroes,
        tab_somas,
        tab_pares_trios,
        tab_ciclos,
        tab_ultimos,
    ) = st.tabs(
        [
            "Frequência & Atraso",
            "Par/Ímpar & Baixa/Alta",
            "Soma das dezenas",
            "Pares & Trios",
            "Ciclos & Quentes/Frios",
            "Últimos resultados",
        ]
    )

    # FREQUÊNCIA & ATRASO + FREQUÊNCIA RECENTE
    with tab_freq:
        st.subheader("Frequência e atraso por dezena")
        atraso_df = calcular_atraso(freq_df, df_concursos)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Ordenado por frequência total")
            st.dataframe(
                freq_df.sort_values("frequencia", ascending=False).reset_index(
                    drop=True
                ),
                hide_index=True,
                use_container_width=True,
            )
        with col2:
            st.markdown("#### Ordenado por atraso atual")
            st.dataframe(
                atraso_df.sort_values(
                    ["atraso_atual", "frequencia"], ascending=[False, False]
                ).reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("#### Frequência recente vs total")
        n_rec = st.slider(
            "Concursos recentes para analisar",
            min_value=20,
            max_value=300,
            value=50,
            step=10,
        )
        df_recent = df_concursos.sort_values("concurso", ascending=False).head(n_rec)
        freq_recent = calcular_frequencias(df_recent)
        freq_recent = freq_recent.rename(columns={"frequencia": "freq_recente"})

        freq_merge = freq_df.merge(freq_recent, on="dezena", how="left")
        freq_merge["freq_recente"] = freq_merge["freq_recente"].fillna(0).astype(int)

        colf1, colf2 = st.columns(2)
        with colf1:
            st.markdown("Top dezenas recentes (freq. recente)")
            st.dataframe(
                freq_merge.sort_values(
                    "freq_recente", ascending=False
                ).head(20),
                hide_index=True,
                use_container_width=True,
            )
        with colf2:
            st.markdown("Top dezenas históricas (freq. total)")
            st.dataframe(
                freq_merge.sort_values("frequencia", ascending=False).head(20),
                hide_index=True,
                use_container_width=True,
            )

    # PAR / ÍMPAR & BAIXA / ALTA
    with tab_padroes:
        st.subheader("Distribuição de pares/ímpares e baixa/alta")
        df_padroes, dist_par_impar, dist_baixa_alta = (
            calcular_padroes_par_impar_baixa_alta(df_concursos)
        )

        st.markdown("#### Padrões mais frequentes")
        st.dataframe(dist_par_impar, hide_index=True, use_container_width=True)
        st.dataframe(dist_baixa_alta, hide_index=True, use_container_width=True)

        with st.expander("Ver tabela detalhada por concurso"):
            st.dataframe(
                df_padroes.sort_values("concurso"),
                hide_index=True,
                use_container_width=True,
            )

    # SOMA
    with tab_somas:
        st.subheader("Soma das dezenas por concurso")
        df_somas, dist_faixas = calcular_somas(df_concursos)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Tabela por concurso")
            st.dataframe(
                df_somas.sort_values("concurso").reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
            )
        with col2:
            st.markdown("#### Distribuição por faixa de soma")
            st.dataframe(
                dist_faixas,
                hide_index=True,
                use_container_width=True,
            )

    # PARES & TRIOS
    with tab_pares_trios:
        st.subheader("Pares e trios mais frequentes")
        top_n = st.slider(
            "Quantidade de pares/trios para listar",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
        )
        df_pares, df_trios = calcular_pares_trios(df_concursos, top_n=top_n)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Pares mais frequentes")
            st.dataframe(df_pares, hide_index=True, use_container_width=True)
        with col2:
            st.markdown("#### Trios mais frequentes")
            st.dataframe(df_trios, hide_index=True, use_container_width=True)

    # CICLOS & QUENTES/FRIOS
    with tab_ciclos:
        st.subheader("Ciclos das dezenas")
        df_ciclos, ciclo_atual = calcular_ciclos(df_concursos)

        colc1, colc2 = st.columns(2)
        with colc1:
            st.markdown("#### Ciclos fechados")
            if df_ciclos.empty:
                st.info("Nenhum ciclo completo encontrado no histórico atual.")
            else:
                st.dataframe(df_ciclos, hide_index=True, use_container_width=True)

        with colc2:
            st.markdown("#### Ciclo em andamento")
            if ciclo_atual["inicio_concurso"] is None:
                st.write("Nenhum ciclo em andamento (ciclo atual acabou de fechar).")
            else:
                st.write(
                    f"Início do ciclo: concurso {ciclo_atual['inicio_concurso']}"
                )
                st.write(
                    f"Concursos no ciclo atual: {ciclo_atual['qtd_concursos']}"
                )
                dezenas_faltando = ciclo_atual.get("dezenas_faltando", [])
                if dezenas_faltando:
                    st.write(
                        f"Dezenas que ainda não saíram neste ciclo ({len(dezenas_faltando)}): "
                        + ", ".join(f"{d:02d}" for d in dezenas_faltando)
                    )
                else:
                    st.write("Todas as dezenas já saíram, ciclo deveria fechar em breve.")

        st.markdown("---")
        st.subheader("Números quentes, frios e atrasados")

        atraso_df = calcular_atraso(freq_df, df_concursos)
        top_quentes = freq_df.sort_values("frequencia", ascending=False).head(20)
        top_frios = freq_df.sort_values("frequencia", ascending=True).head(20)
        top_atrasados = atraso_df.sort_values(
            ["atraso_atual", "frequencia"], ascending=[False, False]
        ).head(20)

        colq1, colq2, colq3 = st.columns(3)
        with colq1:
            st.markdown("**Mais sorteadas (quentes)**")
            st.dataframe(top_quentes, hide_index=True, use_container_width=True)
        with colq2:
            st.markdown("**Menos sorteadas (frias)**")
            st.dataframe(top_frios, hide_index=True, use_container_width=True)
        with colq3:
            st.markdown("**Mais atrasadas**")
            st.dataframe(top_atrasados, hide_index=True, use_container_width=True)

    # ÚLTIMOS RESULTADOS
    with tab_ultimos:
        st.subheader("Últimos resultados da Mega-Sena (histórico local)")
        qtd_ultimos = st.slider(
            "Quantidade de concursos para exibir",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
        )
        ultimos = df_concursos.sort_values("concurso", ascending=False).head(
            qtd_ultimos
        )
        st.dataframe(
            ultimos.sort_values("concurso", ascending=False),
            hide_index=True,
            use_container_width=True,
        )


# ==========================
# CARGA DOS DADOS
# ==========================
try:
    df_concursos = carregar_concursos(CSV_PATH)
    freq_df = calcular_frequencias(df_concursos)
except FileNotFoundError:
    st.error(
        "Arquivo de concursos não encontrado. "
        "Certifique-se de que historico_mega_sena.csv está na raiz do projeto."
    )
    st.stop()
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar/usar os dados: {e}")
    st.stop()


# ==========================
# SESSION STATE – JOGOS
# ==========================
if "jogos" not in st.session_state:
    st.session_state["jogos"] = []
if "jogos_info" not in st.session_state:
    st.session_state["jogos_info"] = []


# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.title("Mega Sena Helper 🎰")

    pagina = st.radio("Navegação", ["Gerar jogos", "Análises estatísticas"])

    gerar = False
    gerar_misto = False
    modo_geracao = "Uma estratégia"

    preco_base_6 = st.number_input(
        "Valor aposta simples (6 dezenas)",
        min_value=1.0,
        max_value=50.0,
        value=PRECO_SIMPLES_6_DEFAULT,
        step=0.5,
    )

    orcamento_max = st.number_input(
        "Orçamento máximo (opcional)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=0.0,
        step=10.0,
    )

    st.markdown("### Filtros avançados (opcional)")
    with st.expander("Restrições sobre os jogos gerados", expanded=False):
        dezenas_fixas_txt = st.text_input(
            "Dezenas fixas (sempre incluir)", placeholder="Ex: 10, 53"
        )
        dezenas_proibidas_txt = st.text_input(
            "Dezenas proibidas (nunca incluir)", placeholder="Ex: 1, 2, 3"
        )
        soma_min = st.number_input(
            "Soma mínima (opcional)", min_value=0, max_value=600, value=0
        )
        soma_max = st.number_input(
            "Soma máxima (opcional)", min_value=0, max_value=600, value=0
        )

    def parse_lista(texto: str) -> list[int]:
        if not texto:
            return []
        return [
            int(x.strip())
            for x in texto.split(",")
            if x.strip().isdigit() and 1 <= int(x.strip()) <= 60
        ]

    dezenas_fixas = parse_lista(dezenas_fixas_txt)
    dezenas_proibidas = parse_lista(dezenas_proibidas_txt)
    soma_min_val = soma_min if soma_min > 0 else None
    soma_max_val = soma_max if soma_max > 0 else None

    if pagina == "Gerar jogos":
        st.markdown("### Modo de geração")

        modo_geracao = st.radio(
            "Como deseja gerar os jogos?",
            ["Uma estratégia", "Misto de estratégias"],
        )

        if modo_geracao == "Uma estratégia":
            st.markdown("### Estratégia de geração")

            estrategia = st.selectbox(
                "Estratégia",
                [
                    "Aleatório puro",
                    "Balanceado par/ímpar",
                    "Setorial (faixas)",
                    "Quentes/Frias/Mix",
                    "Sem sequências longas",
                    "Wheeling simples (base fixa)",
                ],
            )

            st.markdown("### Parâmetros básicos")

            qtd_jogos = st.number_input(
                "Quantidade de jogos",
                min_value=1,
                max_value=500,
                value=10,
                step=1,
                format="%d",
            )
            tam_jogo = st.slider(
                "Dezenas por jogo",
                6,
                15,
                6,
            )

            if estrategia == "Quentes/Frias/Mix":
                st.markdown("#### Mix de dezenas")
                col_q1, col_q2, col_q3 = st.columns(3)
                with col_q1:
                    q_quentes = st.number_input("Quentes", 0, 10, 3)
                with col_q2:
                    q_frias = st.number_input("Frias", 0, 10, 2)
                with col_q3:
                    q_neutras = st.number_input("Neutras", 0, 10, 1)

            if estrategia == "Sem sequências longas":
                st.markdown("#### Controle de sequência")
                limite_seq = st.slider(
                    "Máx. sequência permitida",
                    2,
                    6,
                    3,
                )

            if estrategia == "Wheeling simples (base fixa)":
                st.markdown("#### Base fixa")
                st.text_input(
                    "Base (separada por vírgulas)",
                    key="base_wheeling_value",
                    placeholder="Ex: 1, 5, 12, 23, 34, 45, 56",
                )

            st.markdown("---")
            gerar = st.button(
                "Gerar jogos",
                type="primary",
                use_container_width=True,
            )

        else:
            st.markdown("### Parâmetros básicos do misto")

            tam_jogo_mix = st.slider(
                "Dezenas por jogo (exceto wheeling)",
                6,
                15,
                6,
                key="tam_jogo_mix",
            )

            st.markdown("### Quantos jogos por estratégia?")

            jogos_misto: dict[str, int] = {}

            jogos_misto["Aleatório puro"] = st.number_input(
                "Aleatório puro",
                min_value=0,
                max_value=500,
                value=1,
                step=1,
                key="mix_qtd_aleatorio",
            )

            jogos_misto["Balanceado par/ímpar"] = st.number_input(
                "Balanceado par/ímpar",
                min_value=0,
                max_value=500,
                value=1,
                step=1,
                key="mix_qtd_balanceado",
            )

            jogos_misto["Setorial (faixas)"] = st.number_input(
                "Setorial (faixas)",
                min_value=0,
                max_value=500,
                value=1,
                step=1,
                key="mix_qtd_setorial",
            )

            with st.expander("Quentes/Frias/Mix (opcional)", expanded=False):
                jogos_misto["Quentes/Frias/Mix"] = st.number_input(
                    "Jogos Quentes/Frias/Mix",
                    min_value=0,
                    max_value=500,
                    value=1,
                    step=1,
                    key="mix_qtd_qfm",
                )
                col_q1m, col_q2m, col_q3m = st.columns(3)
                with col_q1m:
                    mix_q_quentes = st.number_input(
                        "Quentes", 0, 10, 3, key="mix_q_quentes"
                    )
                with col_q2m:
                    mix_q_frias = st.number_input(
                        "Frias", 0, 10, 2, key="mix_q_frias"
                    )
                with col_q3m:
                    mix_q_neutras = st.number_input(
                        "Neutras", 0, 10, 1, key="mix_q_neutras"
                    )

            with st.expander("Sem sequências longas (opcional)", expanded=False):
                jogos_misto["Sem sequências longas"] = st.number_input(
                    "Jogos Sem sequências longas",
                    min_value=0,
                    max_value=500,
                    value=1,
                    step=1,
                    key="mix_qtd_sem_seq",
                )
                mix_limite_seq = st.slider(
                    "Máx. sequência permitida (misto)",
                    2,
                    6,
                    3,
                    key="mix_limite_seq",
                )

            with st.expander("Wheeling simples (opcional)", expanded=False):
                jogos_misto["Wheeling simples (base fixa)"] = st.number_input(
                    "Jogos Wheeling simples",
                    min_value=0,
                    max_value=500,
                    value=0,
                    step=1,
                    key="mix_qtd_wheeling",
                )
                st.text_input(
                    "Base fixa (separada por vírgulas)",
                    key="mix_base_wheeling_value",
                    placeholder="Ex: 1, 5, 12, 23, 34, 45, 56",
                )

            st.markdown("---")
            gerar_misto = st.button(
                "Gerar jogos mistos",
                type="primary",
                use_container_width=True,
            )


# ==========================
# EXPLICAÇÕES
# ==========================
explicacoes = {
    "Aleatório puro": "Sorteia dezenas totalmente aleatórias entre 1 e 60.",
    "Balanceado par/ímpar": "Tenta manter distribuições como 3–3 ou 4–2 de pares/ímpares.",
    "Setorial (faixas)": "Distribui dezenas entre as faixas 1–20, 21–40 e 41–60.",
    "Quentes/Frias/Mix": "Combina dezenas mais sorteadas, mais atrasadas e neutras.",
    "Sem sequências longas": "Evita jogos com muitas dezenas consecutivas.",
    "Wheeling simples (base fixa)": "Gera combinações de 6 dezenas dentro de uma base fixa.",
}


# ==========================
# CORPO – PÁGINAS
# ==========================
if pagina == "Gerar jogos":
    st.title("Gerador de jogos da Mega-Sena")
    st.caption(
        "Escolha o modo e as estratégias na barra lateral, ajuste filtros/opções de custo "
        "e clique em **Gerar** para ver jogos, custo e análises."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Resumo do histórico")
        st.metric("Total de concursos", len(df_concursos))
        st.caption(
            f"De {df_concursos['data'].min().date()} "
            f"até {df_concursos['data'].max().date()}"
        )
    with col2:
        st.markdown("### Parâmetros gerais")
        st.write(f"- Modo: **{modo_geracao}**")
        if dezenas_fixas:
            st.write(f"- Dezenas fixas: {sorted(dezenas_fixas)}")
        if dezenas_proibidas:
            st.write(f"- Dezenas proibidas: {sorted(dezenas_proibidas)}")
        if orcamento_max > 0:
            st.write(
                f"- Orçamento máx.: R$ {orcamento_max:,.2f}"
                .replace(",", "X").replace(".", ",").replace("X", ".")
            )

    st.divider()

    if modo_geracao == "Uma estratégia" and "estrategia" in locals():
        with st.expander("Como funciona esta estratégia?", expanded=False):
            st.write(explicacoes.get(estrategia, ""))
    elif modo_geracao == "Misto de estratégias":
        with st.expander("O que cada estratégia faz?", expanded=False):
            for nome, desc in explicacoes.items():
                st.markdown(f"**{nome}**")
                st.write(desc)

    tab_jogos, tab_tabela, tab_analise = st.tabs(
        ["Jogos gerados", "Tabela / Resumo / Exportar", "Análise & Simulação"]
    )

    # Inicialmente, usa o que está em sessão
    jogos: list[list[int]] = st.session_state["jogos"]
    jogos_info: list[dict] = st.session_state["jogos_info"]

    # ---------- GERAÇÃO SIMPLES ----------
    if modo_geracao == "Uma estratégia" and gerar and "estrategia" in locals():
        if estrategia == "Aleatório puro":
            jogos = gerar_aleatorio_puro(int(qtd_jogos), tam_jogo)
        elif estrategia == "Balanceado par/ímpar":
            jogos = gerar_balanceado_par_impar(int(qtd_jogos), tam_jogo)
        elif estrategia == "Setorial (faixas)":
            jogos = gerar_setorial(int(qtd_jogos), tam_jogo)
        elif estrategia == "Quentes/Frias/Mix":
            jogos = gerar_quentes_frias_mix(
                qtd_jogos=int(qtd_jogos),
                tam_jogo=tam_jogo,
                freq_df=freq_df,
                proporcao=(q_quentes, q_frias, q_neutras),
            )
        elif estrategia == "Sem sequências longas":
            jogos = gerar_sem_sequencias(
                qtd_jogos=int(qtd_jogos),
                tam_jogo=tam_jogo,
                limite_sequencia=limite_seq,
            )
        elif estrategia == "Wheeling simples (base fixa)":
            try:
                base_texto = st.session_state.get("base_wheeling_value", "")
                base_dezenas = [
                    int(x.strip())
                    for x in str(base_texto).split(",")
                    if x.strip().isdigit()
                ]
                jogos = gerar_wheeling_simples(
                    base_dezenas=base_dezenas,
                    max_jogos=int(qtd_jogos),
                )
            except Exception:
                jogos = []

        jogos = filtrar_jogos(
            jogos,
            dezenas_fixas=dezenas_fixas,
            dezenas_proibidas=dezenas_proibidas,
            soma_min=soma_min_val,
            soma_max=soma_max_val,
        )
        jogos_info = [{"estrategia": estrategia, "jogo": j} for j in jogos]

    # ---------- GERAÇÃO MISTA ----------
    if modo_geracao == "Misto de estratégias" and gerar_misto:
        jogos = []
        jogos_info = []

        qtd_ap = jogos_misto.get("Aleatório puro", 0)
        if qtd_ap > 0:
            js = gerar_aleatorio_puro(int(qtd_ap), tam_jogo_mix)
            jogos.extend(js)
            jogos_info.extend({"estrategia": "Aleatório puro", "jogo": j} for j in js)

        qtd_bal = jogos_misto.get("Balanceado par/ímpar", 0)
        if qtd_bal > 0:
            js = gerar_balanceado_par_impar(int(qtd_bal), tam_jogo_mix)
            jogos.extend(js)
            jogos_info.extend(
                {"estrategia": "Balanceado par/ímpar", "jogo": j} for j in js
            )

        qtd_set = jogos_misto.get("Setorial (faixas)", 0)
        if qtd_set > 0:
            js = gerar_setorial(int(qtd_set), tam_jogo_mix)
            jogos.extend(js)
            jogos_info.extend(
                {"estrategia": "Setorial (faixas)", "jogo": j} for j in js
            )

        qtd_qfm = jogos_misto.get("Quentes/Frias/Mix", 0)
        if qtd_qfm > 0:
            js = gerar_quentes_frias_mix(
                qtd_jogos=int(qtd_qfm),
                tam_jogo=tam_jogo_mix,
                freq_df=freq_df,
                proporcao=(mix_q_quentes, mix_q_frias, mix_q_neutras),
            )
            jogos.extend(js)
            jogos_info.extend(
                {"estrategia": "Quentes/Frias/Mix", "jogo": j} for j in js
            )

        qtd_ss = jogos_misto.get("Sem sequências longas", 0)
        if qtd_ss > 0:
            js = gerar_sem_sequencias(
                qtd_jogos=int(qtd_ss),
                tam_jogo=tam_jogo_mix,
                limite_sequencia=mix_limite_seq,
            )
            jogos.extend(js)
            jogos_info.extend(
                {"estrategia": "Sem sequências longas", "jogo": j} for j in js
            )

        qtd_wh = jogos_misto.get("Wheeling simples (base fixa)", 0)
        if qtd_wh > 0:
            try:
                base_texto = st.session_state.get("mix_base_wheeling_value", "")
                base_dezenas = [
                    int(x.strip())
                    for x in str(base_texto).split(",")
                    if x.strip().isdigit()
                ]
                js = gerar_wheeling_simples(
                    base_dezenas=base_dezenas,
                    max_jogos=int(qtd_wh),
                )
                jogos.extend(js)
                jogos_info.extend(
                    {"estrategia": "Wheeling simples (base fixa)", "jogo": j}
                    for j in js
                )
            except Exception:
                pass

        jogos_filtrados = filtrar_jogos(
            jogos,
            dezenas_fixas=dezenas_fixas,
            dezenas_proibidas=dezenas_proibidas,
            soma_min=soma_min_val,
            soma_max=soma_max_val,
        )
        novo_info = [info for info in jogos_info if info["jogo"] in jogos_filtrados]
        jogos = [info["jogo"] for info in novo_info]
        jogos_info = novo_info

    # ---------- ORÇAMENTO (só quando gerar/gerar_misto) ----------
    aviso_orcamento = ""
    if (gerar or gerar_misto) and jogos and orcamento_max > 0:
        jogos_dentro = []
        jogos_info_dentro = []
        custo_acum = 0.0
        for info in jogos_info:
            jogo = info["jogo"]
            custo_jogo = preco_aposta_megasena(len(jogo), preco_base_6)
            if custo_acum + custo_jogo > orcamento_max:
                break
            custo_acum += custo_jogo
            jogos_dentro.append(jogo)
            jogos_info_dentro.append(info)

        if len(jogos_dentro) < len(jogos):
            aviso_orcamento = (
                f"Foram mantidos {len(jogos_dentro)} jogos dentro do orçamento. "
                f"{len(jogos) - len(jogos_dentro)} jogos foram descartados."
            )
        jogos = jogos_dentro
        jogos_info = jogos_info_dentro

    # ---------- ATUALIZA SESSION_STATE SE HOUVE NOVA GERAÇÃO ----------
    if (gerar or gerar_misto) and jogos:
        st.session_state["jogos"] = jogos
        st.session_state["jogos_info"] = jogos_info
    else:
        jogos = st.session_state["jogos"]
        jogos_info = st.session_state["jogos_info"]

    # ---------- EXIBIÇÃO ----------
    if not jogos and (gerar or gerar_misto):
        tab_jogos.warning(
            "Nenhum jogo foi gerado após aplicar filtros e orçamento. "
            "Revise os parâmetros na barra lateral."
        )
        tab_tabela.info("Nenhum dado para exibir ainda.")
        tab_analise.info("Nenhuma análise disponível sem jogos gerados.")
    elif not jogos:
        tab_jogos.write(
            "Ajuste os parâmetros na barra lateral e clique em **Gerar** para ver seus jogos aqui."
        )
        tab_tabela.write("A tabela aparecerá aqui após a geração.")
        tab_analise.write("A análise aparecerá aqui após a geração.")
    else:
        custo_total = calcular_custo_total(jogos, preco_base_6)
        prob_total = prob_sena_pacote(jogos)

        # TAB JOGOS
        with tab_jogos:
            colr1, colr2, colr3 = st.columns(3)
            with colr1:
                st.metric("Quantidade de jogos", len(jogos))
            with colr2:
                st.metric(
                    "Custo total estimado",
                    f"R$ {custo_total:,.2f}"
                    .replace(",", "X").replace(".", ",").replace("X", "."),
                )
            with colr3:
                if prob_total > 0:
                    inv = 1.0 / prob_total
                    st.metric(
                        "Chance aprox. de 1 Sena",
                        f"1 em {inv:,.0f}".replace(",", "."),
                    )
                else:
                    st.metric("Chance aprox. de 1 Sena", "N/A")

            if aviso_orcamento:
                st.info(aviso_orcamento)

            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                for i, info in enumerate(jogos_info, start=1):
                    st.code(
                        f"{i:02d} - {info['estrategia']}: {formatar_jogo(info['jogo'])}"
                    )

        # TAB TABELA
        with tab_tabela:
            dados = []
            for info in jogos_info:
                jogo = info["jogo"]
                row = {"estrategia": info["estrategia"]}
                for idx, d in enumerate(jogo, start=1):
                    row[f"d{idx}"] = d
                soma_jogo = sum(jogo)
                pares, impares = pares_impares(jogo)
                bax, alt = baixos_altos(jogo)
                comb_6, fator = cobertura_jogo(jogo)

                row["soma"] = soma_jogo
                row["pares"] = pares
                row["impares"] = impares
                row["baixos"] = bax
                row["altos"] = alt
                row["comb_6"] = comb_6
                row["fator_simples"] = fator

                # Faixa de soma e padrão extremo
                if soma_jogo <= 120:
                    faixa_soma = "muito baixa"
                elif soma_jogo <= 180:
                    faixa_soma = "baixa"
                elif soma_jogo <= 240:
                    faixa_soma = "dentro do comum"
                else:
                    faixa_soma = "alta/muito alta"

                padrao_extremo = (
                    pares == 0
                    or impares == 0
                    or bax == 0
                    or alt == 0
                    or faixa_soma in ("muito baixa", "alta/muito alta")
                )

                row["faixa_soma"] = faixa_soma
                row["padrao_extremo"] = padrao_extremo

                dados.append(row)

            jogos_df = pd.DataFrame(dados)
            st.dataframe(jogos_df, hide_index=True, use_container_width=True)

            csv_data = jogos_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar CSV",
                data=csv_data,
                file_name=f"jogos_{datetime.now().date()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # TAB ANÁLISE & SIMULAÇÃO
        with tab_analise:
            st.markdown("#### Cobertura das dezenas nos jogos gerados")

            todas_dezenas = [d for j in jogos for d in j]
            freq_jogos = pd.Series(todas_dezenas).value_counts().sort_index()
            df_freq_jogos = freq_jogos.reset_index()
            df_freq_jogos.columns = ["dezena", "frequencia"]

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown("**Frequência das dezenas**")
                if not df_freq_jogos.empty:
                    st.bar_chart(
                        df_freq_jogos.set_index("dezena")["frequencia"],
                        use_container_width=True,
                    )
                else:
                    st.info("Nenhuma dezena para exibir.")

            with col_a2:
                st.markdown("**Top dezenas mais usadas**")
                if not df_freq_jogos.empty:
                    top_n = min(10, len(df_freq_jogos))
                    st.dataframe(
                        df_freq_jogos.sort_values(
                            "frequencia", ascending=False
                        ).head(top_n),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info("Nenhum dado de frequência disponível.")

            # Padrões gerados
            padroes_linhas = []
            for jogo in jogos:
                p, imp = pares_impares(jogo)
                bax, alt = baixos_altos(jogo)
                padroes_linhas.append(
                    {"pares": p, "impares": imp, "baixos": bax, "altos": alt}
                )
            df_padroes_jogos = pd.DataFrame(padroes_linhas)
            dist_pi = (
                df_padroes_jogos.groupby(["pares", "impares"])
                .size()
                .reset_index(name="qtd")
                .sort_values("qtd", ascending=False)
            )
            dist_ba = (
                df_padroes_jogos.groupby(["baixos", "altos"])
                .size()
                .reset_index(name="qtd")
                .sort_values("qtd", ascending=False)
            )

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**Padrões pares/ímpares nos jogos gerados**")
                st.dataframe(dist_pi, hide_index=True, use_container_width=True)
            with col_p2:
                st.markdown("**Padrões baixos/altos nos jogos gerados**")
                st.dataframe(dist_ba, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Comparador de estratégias (pacote atual)")

            if jogos_info:
                comp_rows = []
                for est in sorted({info["estrategia"] for info in jogos_info}):
                    jogos_est = [info["jogo"] for info in jogos_info if info["estrategia"] == est]
                    if not jogos_est:
                        continue
                    custo_est = calcular_custo_total(jogos_est, preco_base_6)
                    prob_est = prob_sena_pacote(jogos_est)
                    media_dezenas = np.mean([len(j) for j in jogos_est])
                    comp_rows.append(
                        {
                            "estrategia": est,
                            "qtd_jogos": len(jogos_est),
                            "media_dezenas_por_jogo": media_dezenas,
                            "custo_total_est": custo_est,
                            "prob_sena_aprox": prob_est,
                        }
                    )
                df_comp = pd.DataFrame(comp_rows)
                st.dataframe(df_comp, hide_index=True, use_container_width=True)
            else:
                st.info("Nenhuma estratégia para comparar no pacote atual.")

            st.markdown("---")
            st.markdown("#### Simulação de acertos")

            modo_sim = st.radio(
                "Escolher resultado para simulação",
                ["Usar concurso histórico", "Digitar resultado manual"],
                horizontal=True,
            )

            dezenas_sorteadas: list[int] = []

            if modo_sim == "Usar concurso histórico":
                ultimos_ids = df_concursos.sort_values(
                    "concurso", ascending=False
                )["concurso"].head(100)
                concurso_escolhido = st.selectbox(
                    "Concurso para simular (único)",
                    options=ultimos_ids,
                    format_func=lambda c: f"Concurso {c}",
                )
                linha = df_concursos.loc[
                    df_concursos["concurso"] == concurso_escolhido
                ].iloc[0]
                dezenas_sorteadas = [int(linha[f"d{i}"]) for i in range(1, 7)]
                st.write(
                    f"Dezenas sorteadas: {formatar_jogo(dezenas_sorteadas)}"
                )
            else:
                resultado_manual = st.text_input(
                    "Informe 6 dezenas separadas por vírgula",
                    placeholder="Ex: 5, 12, 23, 34, 45, 60",
                )
                dezenas_sorteadas = [
                    int(x.strip())
                    for x in resultado_manual.split(",")
                    if x.strip().isdigit()
                ]
                if len(dezenas_sorteadas) != 6:
                    st.warning("Informe exatamente 6 dezenas para a simulação.")

            if len(dezenas_sorteadas) == 6:
                df_sim = simular_premios(jogos, dezenas_sorteadas)
                st.markdown("**Resumo de acertos no sorteio escolhido**")
                dist_acertos = (
                    df_sim["acertos"].value_counts().sort_index().reset_index()
                )
                dist_acertos.columns = ["acertos", "qtd_jogos"]
                st.dataframe(dist_acertos, hide_index=True, use_container_width=True)

                st.markdown("**Detalhamento por jogo**")
                st.dataframe(df_sim, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Simulação contra vários concursos do histórico")

            sim_multi = st.checkbox(
                "Simular os jogos contra vários concursos recentes"
            )
            if sim_multi:
                qtd_hist = st.slider(
                    "Quantidade de concursos recentes",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                )
                concursos_multi = df_concursos.sort_values(
                    "concurso", ascending=False
                ).head(qtd_hist)

                with st.spinner("Rodando simulação em vários concursos, aguarde..."):
                    df_multi, df_multi_conc = simular_multi_concursos(
                        jogos,
                        concursos_multi,
                        show_progress=True,
                    )

                st.markdown("**Distribuição global de acertos (todos os concursos simulados)**")
                st.dataframe(
                    df_multi.sort_values("acertos"),
                    hide_index=True,
                    use_container_width=True,
                )

                st.markdown("**Resumo por concurso (máx acertos e prêmios)**")
                st.dataframe(
                    df_multi_conc.sort_values("concurso"),
                    hide_index=True,
                    use_container_width=True,
                )

else:
    pagina_analises(df_concursos, freq_df)
