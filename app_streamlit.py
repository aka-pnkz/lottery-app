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
    initial_sidebar_state="expanded",
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


def pagina_analises(df_concursos: pd.DataFrame, freq_df: pd.DataFrame) -> None:
    st.title("Análises estatísticas da Mega-Sena")
    st.caption(
        "Use esta página para entender o comportamento histórico das dezenas. "
        "As análises são descritivas e não garantem aumento de chance em sorteios futuros."
    )

    tab_freq, tab_padroes, tab_somas, tab_pares_trios = st.tabs(
        [
            "Frequência & Atraso",
            "Par/Ímpar & Baixa/Alta",
            "Soma das dezenas",
            "Pares & Trios",
        ]
    )

    # ---- FREQUÊNCIA & ATRASO ----
    with tab_freq:
        st.subheader("Frequência e atraso por dezena")
        st.markdown(
            "Veja quantas vezes cada dezena foi sorteada e há quantos concursos ela não aparece."
        )

        with st.expander("Como interpretar esta aba?", expanded=False):
            st.write("- **Frequência**: quantidade de vezes que a dezena apareceu no histórico.")
            st.write("- **Último concurso**: sorteio mais recente em que a dezena saiu.")
            st.write(
                "- **Atraso atual**: quantos concursos se passaram desde a última vez "
                "que a dezena apareceu."
            )
            st.caption(
                "Você pode usar esses dados apenas como referência histórica. "
                "Sorteios futuros continuam aleatórios."
            )

        atraso_df = calcular_atraso(freq_df, df_concursos)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Ordenado por frequência")
            st.dataframe(
                freq_df.sort_values("frequencia", ascending=False).reset_index(
                    drop=True
                ),
                hide_index=True,
                use_container_width=True,
            )

        with col2:
            st.markdown("#### Ordenado por atraso")
            st.dataframe(
                atraso_df.sort_values(
                    ["atraso_atual", "frequencia"], ascending=[False, False]
                ).reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
            )

    # ---- PAR / ÍMPAR & BAIXA / ALTA ----
    with tab_padroes:
        st.subheader("Distribuição de pares/ímpares e baixa/alta")
        st.markdown(
            "Cada sorteio é dividido em quantos números pares/ímpares e baixos/altos saíram."
        )

        with st.expander("Como interpretar esta aba?", expanded=False):
            st.write(
                "- **Pares/Ímpares**: mostra quantos concursos tiveram, por exemplo, 3 pares e 3 ímpares."
            )
            st.write(
                "- **Baixos/Altos**: considera baixos de 1 a 30 e altos de 31 a 60."
            )
            st.write(
                "- Padrões mais frequentes podem ser usados como referência "
                "para evitar combinações muito extremas (ex.: tudo par)."
            )

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

    # ---- SOMA DAS DEZENAS ----
    with tab_somas:
        st.subheader("Soma das dezenas por concurso")
        st.markdown(
            "Aqui você vê a soma das seis dezenas em cada sorteio e em quais faixas "
            "essas somas mais aparecem."
        )

        with st.expander("Como interpretar esta aba?", expanded=False):
            st.write(
                "- A **soma** é a adição das 6 dezenas de cada concurso (ex.: 10+20+30+40+50+60)."
            )
            st.write(
                "- As **faixas de soma** ajudam a ver em que intervalos os resultados "
                "mais costumam cair."
            )
            st.write(
                "- Alguns apostadores gostam de manter a soma dos jogos em faixas parecidas "
                "com as mais comuns do histórico."
            )

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

    # ---- PARES & TRIOS ----
    with tab_pares_trios:
        st.subheader("Pares e trios mais frequentes")
        st.markdown(
            "Mostra as combinações de 2 e 3 dezenas que mais se repetiram no mesmo "
            "concurso ao longo do histórico."
        )

        with st.expander("Como interpretar esta aba?", expanded=False):
            st.write(
                "- **Pares**: combinações de 2 dezenas que apareceram juntas muitas vezes."
            )
            st.write(
                "- **Trios**: combinações de 3 dezenas que costumam sair juntas."
            )
            st.write(
                "- Você pode usar esses pares/trios como inspiração para montar jogos "
                "manuais ou complementar suas apostas."
            )

        top_n = st.slider(
            "Quantidade de pares/trios para listar",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Controle quantas combinações mais frequentes deseja visualizar.",
        )

        df_pares, df_trios = calcular_pares_trios(df_concursos, top_n=top_n)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Pares mais frequentes")
            st.dataframe(
                df_pares,
                hide_index=True,
                use_container_width=True,
            )

        with col2:
            st.markdown("#### Trios mais frequentes")
            st.dataframe(
                df_trios,
                hide_index=True,
                use_container_width=True,
            )


# ==========================
# CARGA DOS DADOS (SEGURO)
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
# UI – SIDEBAR (CONTROLES)
# ==========================
with st.sidebar:
    st.title("Mega Sena Helper 🎰")

    pagina = st.radio(
        "Navegação",
        ["Gerar jogos", "Análises estatísticas"],
        help="Escolha se deseja criar novos jogos ou analisar o histórico.",
    )

    gerar = False  # default

    if pagina == "Gerar jogos":
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
            help="Tipo de lógica usada para montar os jogos. "
                 "Use a explicação na tela principal para entender cada uma.",
        )

        st.markdown("### Parâmetros básicos")

        qtd_jogos = st.slider(
            "Quantidade de jogos",
            1,
            30,
            10,
            help="Número de jogos diferentes que serão gerados.",
        )
        tam_jogo = st.slider(
            "Dezenas por jogo",
            6,
            15,
            6,
            help="Quantidade de dezenas em cada jogo (aposta).",
        )

        if estrategia == "Quentes/Frias/Mix":
            st.markdown("#### Mix de dezenas")
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                q_quentes = st.number_input(
                    "Quentes",
                    0,
                    10,
                    3,
                    help="Dezenas mais sorteadas no histórico.",
                )
            with col_q2:
                q_frias = st.number_input(
                    "Frias",
                    0,
                    10,
                    2,
                    help="Dezenas menos sorteadas / mais atrasadas.",
                )
            with col_q3:
                q_neutras = st.number_input(
                    "Neutras",
                    0,
                    10,
                    1,
                    help="Dezenas que não estão entre as mais nem entre as menos frequentes.",
                )

            with st.popover("O que são quentes, frias e neutras?"):
                st.write("• **Quentes**: saíram mais vezes no histórico.")
                st.write("• **Frias**: saíram poucas vezes ou estão há muitos concursos sem aparecer.")
                st.write("• **Neutras**: não se destacam nem como muito, nem como pouco sorteadas.")
                st.caption("Isso é apenas referência histórica, não aumenta a chance real de acerto.")

        if estrategia == "Sem sequências longas":
            st.markdown("#### Controle de sequência")
            limite_seq = st.slider(
                "Máx. sequência permitida",
                2,
                6,
                3,
                help="Maior quantidade de dezenas consecutivas que pode aparecer no mesmo jogo.",
            )

        if estrategia == "Wheeling simples (base fixa)":
            st.markdown("#### Base fixa")
            st.text_input(
                "Base (separada por vírgulas)",
                key="base_wheeling_value",
                placeholder="Ex: 1, 5, 12, 23, 34, 45, 56",
                help="Informe uma lista de dezenas. O sistema monta combinações de 6 números dentro dessa base.",
            )

        st.markdown("---")
        gerar = st.button(
            "Gerar jogos",
            type="primary",
            use_container_width=True,
            help="Clique após ajustar os parâmetros para criar novos jogos.",
        )


# ==========================
# EXPLICAÇÕES DAS ESTRATÉGIAS
# ==========================
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


# ==========================
# CORPO – PÁGINAS
# ==========================
if pagina == "Gerar jogos":
    # Cabeçalho
    st.title("Gerador de jogos da Mega-Sena")
    st.caption(
        "1) Escolha a estratégia e os parâmetros na barra lateral. "
        "2) Clique em **Gerar jogos**. "
        "3) Use a lista ou a tabela para copiar ou baixar os jogos."
    )

    # Bloco de resumo em duas colunas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Resumo do histórico")
        st.metric(
            "Total de concursos",
            value=len(df_concursos),
            help="Quantidade total de sorteios disponíveis no histórico.",
        )
        st.caption(
            f"De {df_concursos['data'].min().date()} "
            f"até {df_concursos['data'].max().date()}"
        )

    with col2:
        st.markdown("### Parâmetros atuais")
        if "estrategia" in locals():
            st.write(f"- Estratégia: **{estrategia}**")
            if "qtd_jogos" in locals():
                st.write(f"- Jogos: **{qtd_jogos}** | Dezenas por jogo: **{tam_jogo}**")
            if "q_quentes" in locals() and estrategia == "Quentes/Frias/Mix":
                st.write(f"- Mix: {q_quentes} quentes / {q_frias} frias / {q_neutras} neutras")
            if "limite_seq" in locals() and estrategia == "Sem sequências longas":
                st.write(f"- Máx. sequência: {limite_seq} dezenas seguidas")
            if estrategia == "Wheeling simples (base fixa)":
                st.write(
                    f"- Base: `{st.session_state.get('base_wheeling_value', '')}`"
                )
        else:
            st.write("Defina a estratégia na barra lateral para ver o resumo.")

    st.divider()

    # Explicação da estratégia em expander
    if "estrategia" in locals():
        with st.expander("Como funciona esta estratégia?", expanded=False):
            st.write(explicacoes.get(estrategia, ""))
            st.caption(
                "Observação: todas as combinações têm a mesma chance matemática. "
                "As estratégias servem apenas para organizar a forma de jogar."
            )

    # Tabs para jogos e tabela/export
    tab_jogos, tab_tabela = st.tabs(["Jogos gerados", "Tabela / Exportar"])

    jogos: list[list[int]] = []

    if gerar and "estrategia" in locals():
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

    # Conteúdo das tabs
    if not jogos and gerar:
        tab_jogos.warning(
            "Nenhum jogo foi gerado. Revise os parâmetros na barra lateral e tente novamente."
        )
        tab_tabela.info("Nenhum dado para exibir ainda.")
    elif not jogos:
        tab_jogos.write(
            "Ajuste os parâmetros na barra lateral e clique em **Gerar jogos** para ver seus jogos aqui."
        )
        tab_tabela.write("A tabela com os jogos aparecerá aqui após a geração.")
    else:
        with tab_jogos:
            st.markdown("#### Lista de jogos")
            st.caption("Use esta lista para visualizar rapidamente e copiar jogos individuais.")
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                for i, jogo in enumerate(jogos, start=1):
                    st.code(f"Jogo {i:02d}: {formatar_jogo(jogo)}")

        with tab_tabela:
            st.markdown("#### Tabela completa e exportação")
            st.caption("Use a tabela para copiar em bloco ou exportar para CSV.")
            jogos_df = pd.DataFrame(
                jogos,
                columns=[f"d{i}" for i in range(1, len(jogos[0]) + 1)],
            )
            st.dataframe(jogos_df, hide_index=True, use_container_width=True)

            csv_data = jogos_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar CSV",
                data=csv_data,
                file_name=f"jogos_{datetime.now().date()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    pagina_analises(df_concursos, freq_df)
