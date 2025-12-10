import itertools
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
)

CSV_PATH = "historico_mega_sena.csv"


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

        dezenas = np.concatenate([
            np.random.choice(s1, size=min(q1, len(s1)), replace=False),
            np.random.choice(s2, size=min(q2, len(s2)), replace=False),
            np.random.choice(s3, size=min(q3, len(s3)), replace=False),
        ])

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
                np.random.choice(quentes, size=min(q_quentes, len(quentes)), replace=False)
            )
        if len(frias) > 0 and q_frias > 0:
            dezenas.extend(
                np.random.choice(frias, size=min(q_frias, len(frias)), replace=False)
            )
        if len(neutras) > 0 and q_neutras > 0:
            dezenas.extend(
                np.random.choice(neutras, size=min(q_neutras, len(neutras)), replace=False)
            )

        if len(dezenas) < tam_jogo:
            universe = np.setdiff1d(np.arange(1, 61), dezenas)
            extra = np.random.choice(universe, size=tam_jogo - len(dezenas), replace=False)
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
# FUNÇÕES DE ANÁLISE
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
        df["faixa_soma"].value_counts(dropna=False)
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
        pd.DataFrame(
            [{"par": k, "qtd": v} for k, v in pares_contagem.items()]
        )
        .sort_values("qtd", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    df_trios = (
        pd.DataFrame(
            [{"trio": k, "qtd": v} for k, v in trios_contagem.items()]
        )
        .sort_values("qtd", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return df_pares, df_trios


def pagina_analises(df_concursos: pd.DataFrame, freq_df: pd.DataFrame) -> None:
    st.header("Análises estatísticas da Mega-Sena")

    st.markdown(
        "As análises abaixo são descritivas: ajudam a entender o histórico "
        "(frequência, atraso, padrões de par/ímpar e baixa/alta, soma, pares e trios), "
        "mas não aumentam a probabilidade de acerto em sorteios futuros."
    )

    tab_freq, tab_padroes, tab_somas, tab_pares_trios = st.tabs(
        [
            "Frequência & Atraso",
            "Par/Ímpar & Baixa/Alta",
            "Soma das dezenas",
            "Pares & Trios",
        ]
    )

    with tab_freq:
        st.subheader("Frequência e atraso por dezena")

        atraso_df = calcular_atraso(freq_df, df_concursos)

        st.markdown(
            "- **frequência**: quantas vezes a dezena saiu no histórico.\n"
            "- **último concurso**: último sorteio em que apareceu.\n"
            "- **atraso atual**: concursos desde a última vez que saiu."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Frequência das dezenas")
            st.dataframe(
                freq_df.sort_values("frequencia", ascending=False).reset_index(drop=True),
                width="stretch",
                height=400,
                hide_index=True,
            )

        with col2:
            st.markdown("#### Ordenado por atraso")
            st.dataframe(
                atraso_df.sort_values(
                    ["atraso_atual", "frequencia"], ascending=[False, False]
                ).reset_index(drop=True),
                width="stretch",
                height=400,
                hide_index=True,
            )

    with tab_padroes:
        st.subheader("Distribuição de pares/ímpares e baixa/alta")

        df_padroes, dist_par_impar, dist_baixa_alta = calcular_padroes_par_impar_baixa_alta(
            df_concursos
        )

        st.markdown(
            "Cada concurso é decomposto em quantos pares/ímpares e quantos números "
            "baixos (1–30) e altos (31–60) saíram."
        )

        st.dataframe(dist_par_impar, width="stretch", hide_index=True)
        st.dataframe(dist_baixa_alta, width="stretch", hide_index=True)

        with st.expander("Ver tabela detalhada por concurso"):
            st.dataframe(
                df_padroes.sort_values("concurso"),
                width="stretch",
                hide_index=True,
            )

    with tab_somas:
        st.subheader("Soma das dezenas por concurso")

        df_somas, dist_faixas = calcular_somas(df_concursos)

        st.markdown(
            "Aqui você vê a soma de cada sorteio e em quais faixas de soma "
            "os resultados mais se concentram."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Tabela por concurso")
            st.dataframe(
                df_somas.sort_values("concurso").reset_index(drop=True),
                width="stretch",
                height=400,
                hide_index=True,
            )

        with col2:
            st.markdown("#### Distribuição por faixa de soma")
            st.dataframe(
                dist_faixas,
                width="stretch",
                height=400,
                hide_index=True,
            )

    with tab_pares_trios:
        st.subheader("Pares e trios mais frequentes")

        st.markdown(
            "Mostra as combinações de 2 e 3 dezenas que mais se repetiram no mesmo concurso "
            "em todo o histórico."
        )

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
            st.dataframe(df_pares, width="stretch", height=400, hide_index=True)

        with col2:
            st.markdown("#### Trios mais frequentes")
            st.dataframe(df_trios, width="stretch", height=400, hide_index=True)


# ==========================
# UI – SIDEBAR
# ==========================
with st.sidebar:
    st.title("Mega Sena Helper 🎰")

    pagina = st.radio(
        "Página",
        ["Gerar jogos", "Análises estatísticas"],
    )

    if pagina == "Gerar jogos":
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

        qtd_jogos = st.number_input(
            "Quantidade de jogos",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
        )

        tam_jogo = st.slider(
            "Dezenas por jogo",
            min_value=6,
            max_value=15,
            value=6,
        )

        if estrategia == "Sem sequências longas":
            limite_seq = st.number_input(
                "Máx. tamanho de sequência permitida",
                min_value=2,
                max_value=6,
                value=3,
            )
        else:
            limite_seq = 3

        if estrategia == "Quentes/Frias/Mix":
            q_quentes = st.number_input("Qtd. quentes", 0, 6, 3)
            q_frias = st.number_input("Qtd. frias", 0, 6, 2)
            q_neutras = st.number_input("Qtd. neutras", 0, 6, 1)
        else:
            q_quentes, q_frias, q_neutras = 3, 2, 1

        # -----------------------
        # CONTROLES DA BASE (WHEELING)
        # -----------------------
        if estrategia == "Wheeling simples (base fixa)":
            modo_base = st.selectbox(
                "Modo da base de dezenas",
                [
                    "Manual",
                    "Mais sorteados (quentes)",
                    "Mais atrasados (frios)",
                    "Mix quentes+frios",
                ],
                key="modo_base_wheeling",
            )

            qtd_base = st.number_input(
                "Quantidade de dezenas na base",
                min_value=6,
                max_value=25,
                value=12,
                step=1,
                help=(
                    "Define quantos números a base terá para o wheeling. "
                    "Bases entre 10 e 20 dezenas são mais comuns."
                ),
                key="qtd_base_wheeling",
            )

            if "base_wheeling_value" not in st.session_state:
                st.session_state.base_wheeling_value = "1,3,5,7,9,11,13,15,17,19"

            if st.button("Gerar base sugerida"):
                st.session_state["__gerar_base_modo"] = st.session_state.modo_base_wheeling
                st.session_state["__gerar_base_qtd"] = int(st.session_state.qtd_base_wheeling)

            base_str = st.text_input(
                "Base de dezenas",
                key="base_wheeling",
                value=st.session_state.base_wheeling_value,
                disabled=(modo_base != "Manual"),
                help=(
                    "No modo Manual, informe de 6 a ~20 dezenas separadas por vírgula, "
                    "por exemplo: 1,3,5,7,9,11,13,15,17,19. "
                    "Nos outros modos, a base é preenchida automaticamente pelas análises."
                ),
            )

            if modo_base == "Manual":
                st.session_state.base_wheeling_value = base_str
        else:
            base_str = st.session_state.get(
                "base_wheeling_value",
                "1,3,5,7,9,11,13,15,17,19",
            )

        st.markdown("---")
        gerar = st.button("Gerar jogos agora", type="primary")

    else:
        estrategia = None
        qtd_jogos = tam_jogo = limite_seq = 0
        q_quentes = q_frias = q_neutras = 0
        base_str = ""
        gerar = False


# ==========================
# CORPO PRINCIPAL
# ==========================
st.title("Gerador e Analisador da Mega-Sena")

st.markdown(
    "Ferramenta para estudo estatístico e organização de jogos da Mega-Sena. "
    "Todas as estratégias e análises são descritivas: não aumentam a chance "
    "matemática de acerto, apenas ajudam a evitar padrões extremos e entender o histórico."
)

try:
    df_concursos = carregar_concursos(CSV_PATH)
    freq_df = calcular_frequencias(df_concursos)

    # Geração automática da base do wheeling (se solicitado)
    if (
        pagina == "Gerar jogos"
        and estrategia == "Wheeling simples (base fixa)"
        and "__gerar_base_modo" in st.session_state
    ):
        atraso_df_local = calcular_atraso(freq_df, df_concursos)
        modo = st.session_state["__gerar_base_modo"]
        qtd = int(st.session_state.get("__gerar_base_qtd", 12))

        base_dezenas_auto: list[int] = []

        if modo == "Mais sorteados (quentes)":
            base_dezenas_auto = (
                freq_df.sort_values("frequencia", ascending=False)["dezena"]
                .head(qtd)
                .tolist()
            )
        elif modo == "Mais atrasados (frios)":
            base_dezenas_auto = (
                atraso_df_local.sort_values("atraso_atual", ascending=False)["dezena"]
                .head(qtd)
                .tolist()
            )
        elif modo == "Mix quentes+frios":
            q_quentes_base = max(3, qtd // 2)
            q_frias_base = max(0, qtd - q_quentes_base)

            quentes_base = (
                freq_df.sort_values("frequencia", ascending=False)["dezena"]
                .head(q_quentes_base)
                .tolist()
            )
            frias_base = (
                atraso_df_local.sort_values("atraso_atual", ascending=False)["dezena"]
                .head(q_frias_base)
                .tolist()
            )
            base_dezenas_auto = sorted(set(quentes_base + frias_base))[:qtd]

        if base_dezenas_auto:
            st.session_state.base_wheeling_value = ",".join(str(d) for d in base_dezenas_auto)

    if pagina == "Gerar jogos":
        explicacoes = {
            "Aleatório puro": (
                "Sorteia dezenas totalmente aleatórias entre 1 e 60, "
                "alinhado ao fato de que todas as combinações têm a mesma chance."
            ),
            "Balanceado par/ímpar": (
                "Tenta manter distribuições como 3–3 ou 4–2 de pares/ímpares, "
                "evitando padrões extremos (tudo par ou tudo ímpar)."
            ),
            "Setorial (faixas)": (
                "Distribui dezenas entre as faixas 1–20, 21–40 e 41–60, "
                "evitando concentrar todos os números em um único trecho."
            ),
            "Quentes/Frias/Mix": (
                "Combina dezenas mais sorteadas (quentes), mais atrasadas (frias) e neutras, "
                "usando o histórico apenas como referência."
            ),
            "Sem sequências longas": (
                "Rejeita jogos com muitas dezenas consecutivas (por exemplo 10–11–12–13), "
                "padrão que muitos apostadores preferem evitar."
            ),
            "Wheeling simples (base fixa)": (
                "Gera combinações de 6 dezenas dentro de uma base fixa, "
                "formando um mini-fechamento com melhor cobertura daquele conjunto."
            ),
        }

        st.info(explicacoes.get(estrategia, ""))

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Histórico carregado")
            st.write(
                f"Total de concursos: **{len(df_concursos)}** "
                f"(de {df_concursos['data'].min().date()} até {df_concursos['data'].max().date()})"
            )

        with col2:
            st.subheader("Parâmetros selecionados")
            st.write(f"- Estratégia: **{estrategia}**")
            st.write(f"- Jogos: **{qtd_jogos}**")
            st.write(f"- Dezenas por jogo: **{tam_jogo}**")

            if estrategia == "Quentes/Frias/Mix":
                st.write(
                    f"- Mix: **{q_quentes} quentes, {q_frias} frias, {q_neutras} neutras**"
                )
            if estrategia == "Sem sequências longas":
                st.write(f"- Máx. sequência: **{limite_seq}** dezenas seguidas")
            if estrategia == "Wheeling simples (base fixa)":
                st.write(f"- Base atual: `{st.session_state.get('base_wheeling_value', '')}`")

            if gerar:
                if estrategia == "Aleatório puro":
                    jogos = gerar_aleatorio_puro(qtd_jogos, tam_jogo)
                elif estrategia == "Balanceado par/ímpar":
                    jogos = gerar_balanceado_par_impar(qtd_jogos, tam_jogo)
                elif estrategia == "Setorial (faixas)":
                    jogos = gerar_setorial(qtd_jogos, tam_jogo)
                elif estrategia == "Quentes/Frias/Mix":
                    jogos = gerar_quentes_frias_mix(
                        qtd_jogos=qtd_jogos,
                        tam_jogo=tam_jogo,
                        freq_df=freq_df,
                        proporcao=(q_quentes, q_frias, q_neutras),
                    )
                elif estrategia == "Sem sequências longas":
                    jogos = gerar_sem_sequencias(
                        qtd_jogos=qtd_jogos,
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
                            max_jogos=qtd_jogos,
                        )
                    except Exception:
                        jogos = []
                else:
                    jogos = []

                st.markdown("### Jogos gerados")
                if not jogos:
                    st.warning("Nenhum jogo gerado. Verifique os parâmetros.")
                else:
                    for i, jogo in enumerate(jogos, start=1):
                        st.code(f"Jogo {i:02d}: {formatar_jogo(jogo)}")

                    jogos_df = pd.DataFrame(
                        jogos,
                        columns=[f"d{i}" for i in range(1, len(jogos[0]) + 1)],
                    )
                    csv_data = jogos_df.to_csv(index=False, sep=";").encode("utf-8")
                    st.download_button(
                        label="Baixar jogos em CSV",
                        data=csv_data,
                        file_name=f"jogos_megasena_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

    else:
        pagina_analises(df_concursos, freq_df)

except FileNotFoundError:
    st.error(
        "Arquivo de concursos não encontrado. "
        "Certifique-se de que historico_mega_sena.csv está na raiz do projeto."
    )
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar/usar os dados: {e}")
