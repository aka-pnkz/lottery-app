import math
import os
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==========================
# CONFIG GERAL
# ==========================

st.set_page_config(
    page_title="Lottery Helper",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0f172a0d;
        }
        .main-title {
            font-size: 2.0rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .main-subtitle {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 0.75rem;
        }
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            padding: 0.75rem 0.9rem;
            border-radius: 0.75rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 3px rgba(15,23,42,0.08);
        }
        div[data-testid="metric-container"] > label {
            font-size: 0.8rem;
            color: #6b7280;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            background-color: #e5e7eb33;
        }
        .stTabs [aria-selected="true"] {
            background-color: #111827;
            color: #f9fafb;
        }
        .stDataFrame thead tr th {
            font-size: 0.80rem;
            padding-top: 0.4rem;
            padding-bottom: 0.4rem;
        }
        .stDataFrame tbody tr td {
            font-size: 0.80rem;
            padding-top: 0.25rem;
            padding-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_css()

# preços oficiais atuais da aposta mínima (2025) [file:1]
PRECO_BASE_MEGA = 6.00
PRECO_BASE_LOTO = 3.50

# endpoints oficiais da Caixa para download histórico (XLSX) [web:2]
URL_LOTOFACIL_DOWNLOAD = (
    "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/download"
    "?modalidade=Lotof%C3%A1cil"
)
URL_MEGA_DOWNLOAD = (
    "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/download"
    "?modalidade=Mega-Sena"
)

# ==========================
# ETL / HISTÓRICO
# ==========================


def carregar_concursos(caminho_csv: str, n_dezenas: int) -> pd.DataFrame:
    cols = ["concurso", "data"] + [f"d{i}" for i in range(1, n_dezenas + 1)]
    df = pd.read_csv(caminho_csv, sep=";")

    df = df[cols]

    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["concurso", "data"])

    hoje = pd.Timestamp.today().normalize()
    df = df[(df["data"] >= "1996-01-01") & (df["data"] <= hoje)]

    df["concurso"] = df["concurso"].astype(int)

    dezenas_cols = [f"d{i}" for i in range(1, n_dezenas + 1)]
    for c in dezenas_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=dezenas_cols)
    for c in dezenas_cols:
        df[c] = df[c].astype(int)

    df = df.sort_values("concurso")
    return df  # [file:1]


def calcular_frequencias(df: pd.DataFrame, n_dezenas: int) -> pd.DataFrame:
    dezenas_cols = [f"d{i}" for i in range(1, n_dezenas + 1)]
    todas = df[dezenas_cols].values.ravel()
    freq = pd.Series(todas).value_counts().sort_index()
    freq_df = freq.reset_index()
    freq_df.columns = ["dezena", "frequencia"]
    freq_df["dezena"] = freq_df["dezena"].astype(int)
    return freq_df  # [file:1]


def baixar_xlsx_lotofacil() -> BytesIO:
    resp = requests.get(URL_LOTOFACIL_DOWNLOAD, timeout=60)
    resp.raise_for_status()
    return BytesIO(resp.content)  # [web:2]


def baixar_xlsx_megasena() -> BytesIO:
    resp = requests.get(URL_MEGA_DOWNLOAD, timeout=60)
    resp.raise_for_status()
    return BytesIO(resp.content)  # [web:2]


def _limpar_concurso_data(df: pd.DataFrame) -> pd.DataFrame:
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["concurso", "data"])
    hoje = pd.Timestamp.today().normalize()
    df = df[(df["data"] >= "1996-01-01") & (df["data"] <= hoje)]
    df["concurso"] = df["concurso"].astype(int)
    return df  # [file:1]


def atualizar_base_lotofacil(buf_xlsx: BytesIO) -> None:
    df_raw = pd.read_excel(buf_xlsx)

    cols = [
        "Concurso", "Data Sorteio",
        "Bola1", "Bola2", "Bola3", "Bola4", "Bola5",
        "Bola6", "Bola7", "Bola8", "Bola9", "Bola10",
        "Bola11", "Bola12", "Bola13", "Bola14", "Bola15",
    ]
    for c in cols:
        if c not in df_raw.columns:
            raise RuntimeError(f"Coluna ausente na Lotofácil: {c}")

    df = df_raw[cols].copy()

    rename = {"Concurso": "concurso", "Data Sorteio": "data"}
    for i in range(1, 16):
        rename[f"Bola{i}"] = f"d{i}"
    df.rename(columns=rename, inplace=True)

    df = _limpar_concurso_data(df)

    dezenas = [f"d{i}" for i in range(1, 16)]
    for c in dezenas:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=dezenas)
    for c in dezenas:
        df[c] = df[c].astype(int)

    df[dezenas] = np.sort(df[dezenas].values, axis=1)
    df = df.sort_values("concurso")

    df.to_csv("historicolotofacil.csv", sep=";", index=False)  # [file:1]


def atualizar_base_megasena(buf_xlsx: BytesIO) -> None:
    df_raw = pd.read_excel(buf_xlsx)

    cols = [
        "Concurso", "Data do Sorteio",
        "Bola1", "Bola2", "Bola3", "Bola4", "Bola5", "Bola6",
    ]
    for c in cols:
        if c not in df_raw.columns:
            raise RuntimeError(f"Coluna ausente na Mega-Sena: {c}")

    df = df_raw[cols].copy()

    rename = {"Concurso": "concurso", "Data do Sorteio": "data"}
    for i in range(1, 7):
        rename[f"Bola{i}"] = f"d{i}"
    df.rename(columns=rename, inplace=True)

    df = _limpar_concurso_data(df)

    dezenas = [f"d{i}" for i in range(1, 7)]
    for c in dezenas:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=dezenas)
    for c in dezenas:
        df[c] = df[c].astype(int)

    df[dezenas] = np.sort(df[dezenas].values, axis=1)
    df = df.sort_values("concurso")

    df.to_csv("historicomegasena.csv", sep=";", index=False)  # [file:1]


def _remover_arquivo_se_existir(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)  # [web:42]


def atualizar_base_lotofacil_automatico() -> None:
    _remover_arquivo_se_existir("historicolotofacil.csv")
    buf = baixar_xlsx_lotofacil()
    atualizar_base_lotofacil(buf)


def atualizar_base_megasena_automatico() -> None:
    _remover_arquivo_se_existir("historicomegasena.csv")
    buf = baixar_xlsx_megasena()
    atualizar_base_megasena(buf)

# ==========================
# FUNÇÕES DE JOGOS / ANÁLISE
# ==========================


def pares_impares(jogo: list[int]) -> tuple[int, int]:
    pares = sum(1 for d in jogo if d % 2 == 0)
    impares = len(jogo) - pares
    return pares, impares  # [file:1]


def baixos_altos(jogo: list[int], limite_baixo: int) -> tuple[int, int]:
    baixos = sum(1 for d in jogo if 1 <= d <= limite_baixo)
    altos = len(jogo) - baixos
    return baixos, altos  # [file:1]


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
    return False  # [file:1]


PRIMOS_ATE_60 = {
    2, 3, 5, 7, 11, 13, 17, 19,
    23, 29, 31, 37, 41, 43, 47, 53, 59,
}  # [file:1]


def contar_primos(jogo: list[int]) -> int:
    return sum(1 for d in jogo if d in PRIMOS_ATE_60)  # [file:1]


def formatar_jogo(jogo: list[int]) -> str:
    return " - ".join(f"{d:02d}" for d in jogo)  # [file:1]


def gerar_aleatorio_puro(qtd_jogos: int, tam_jogo: int, n_universo: int) -> list[list[int]]:
    universe = np.arange(1, n_universo + 1)
    jogos = []
    for _ in range(qtd_jogos):
        dezenas = np.random.choice(universe, size=tam_jogo, replace=False)
        jogos.append(sorted(dezenas.tolist()))
    return jogos  # [file:1]


def gerar_balanceado_par_impar(qtd_jogos: int, tam_jogo: int, n_universo: int) -> list[list[int]]:
    universe = np.arange(1, n_universo + 1)
    jogos = []
    for _ in range(qtd_jogos):
        tentativas = 0
        while True:
            tentativas += 1
            dezenas = np.random.choice(universe, size=tam_jogo, replace=False)
            pares, impares = pares_impares(dezenas.tolist())
            if pares not in (0, tam_jogo) and impares not in (0, tam_jogo):
                jogos.append(sorted(dezenas.tolist()))
                break
            if tentativas > 50:
                jogos.append(sorted(dezenas.tolist()))
                break
    return jogos  # [file:1]


def gerar_quentes_frias_mix(
    qtd_jogos: int,
    tam_jogo: int,
    freq_df: pd.DataFrame,
    n_universo: int,
    proporcao: tuple[int, int, int] = (5, 5, 5),
) -> list[list[int]]:
    total_quentes, total_frias, total_neutras = proporcao
    freq_ord = freq_df.sort_values("frequencia", ascending=False)
    quentes = freq_ord["dezena"].values[:10]
    frias = freq_ord.sort_values("frequencia", ascending=True)["dezena"].values[:10]
    neutras = np.setdiff1d(np.arange(1, n_universo + 1), np.union1d(quentes, frias))

    jogos = []
    for _ in range(qtd_jogos):
        q_quentes = min(total_quentes, tam_jogo)
        q_frias = min(total_frias, max(0, tam_jogo - q_quentes))
        q_neutras = max(0, tam_jogo - q_quentes - q_frias)

        dezenas: list[int] = []

        if len(quentes) > 0 and q_quentes > 0:
            dezenas.extend(np.random.choice(quentes, size=min(q_quentes, len(quentes)), replace=False))
        if len(frias) > 0 and q_frias > 0:
            dezenas.extend(np.random.choice(frias, size=min(q_frias, len(frias)), replace=False))
        if len(neutras) > 0 and q_neutras > 0:
            dezenas.extend(np.random.choice(neutras, size=min(q_neutras, len(neutras)), replace=False))

        if len(dezenas) < tam_jogo:
            universe = np.setdiff1d(np.arange(1, n_universo + 1), dezenas)
            extra = np.random.choice(universe, size=tam_jogo - len(dezenas), replace=False)
            dezenas = np.concatenate([dezenas, extra])

        jogos.append(sorted(list(map(int, dezenas))))
    return jogos  # [file:1]


def gerar_sem_sequencias(
    qtd_jogos: int, tam_jogo: int, n_universo: int, limite_sequencia: int = 3
) -> list[list[int]]:
    universe = np.arange(1, n_universo + 1)
    jogos = []
    for _ in range(qtd_jogos):
        tentativas = 0
        while True:
            tentativas += 1
            dezenas = np.random.choice(universe, size=tam_jogo, replace=False)
            if not tem_sequencia_longa(dezenas.tolist(), limite=limite_sequencia):
                jogos.append(sorted(dezenas.tolist()))
                break
            if tentativas > 100:
                jogos.append(sorted(dezenas.tolist()))
                break
    return jogos  # [file:1]


def preco_aposta_loteria(n_dezenas: int, n_min_base: int, preco_base: float) -> float:
    if n_dezenas < n_min_base:
        raise ValueError("Número de dezenas menor que o mínimo permitido.")
    comb = math.comb(n_dezenas, n_min_base)
    return comb * preco_base  # [file:1]


def calcular_custo_total(jogos: list[list[int]], n_min_base: int, preco_base: float) -> float:
    total = 0.0
    for jogo in jogos:
        n = len(jogo)
        if n < n_min_base:
            continue
        total += preco_aposta_loteria(n, n_min_base, preco_base)
    return total  # [file:1]


def prob_premio_maximo_pacote(
    jogos: list[list[int]], n_min_base: int, comb_target: int
) -> float:
    probs = []
    for jogo in jogos:
        n = len(jogo)
        if n < n_min_base:
            continue
        comb_jogo = math.comb(n, n_min_base)
        p = comb_jogo / comb_target
        probs.append(p)
    prob_nao = 1.0
    for p in probs:
        prob_nao *= (1.0 - p)
    return 1.0 - prob_nao  # [file:1]


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
    return filtrados  # [file:1]


def simular_premios(jogos: list[list[int]], dezenas_sorteadas: list[int]) -> pd.DataFrame:
    dezenas_s = set(dezenas_sorteadas)
    linhas = []
    for i, jogo in enumerate(jogos, start=1):
        acertos = len(set(jogo) & dezenas_s)
        linhas.append({"jogo_id": i, "jogo": formatar_jogo(jogo), "acertos": acertos})
    return pd.DataFrame(linhas)  # [file:1]


def calcular_atraso(freq_df: pd.DataFrame, df_concursos: pd.DataFrame, n_dezenas: int) -> pd.DataFrame:
    dezenas_cols = [f"d{i}" for i in range(1, n_dezenas + 1)]
    ultimo_concurso: dict[int, int] = {}
    for _, row in df_concursos[["concurso"] + dezenas_cols].iterrows():
        conc = int(row["concurso"])
        for d in row[dezenas_cols]:
            ultimo_concurso[int(d)] = conc
    max_concurso = int(df_concursos["concurso"].max())
    atraso_list = []
    for dezena in range(1, int(freq_df["dezena"].max()) + 1):
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
    return pd.DataFrame(atraso_list)  # [file:1]


def calcular_padroes_par_impar_baixa_alta(
    df_concursos: pd.DataFrame,
    n_dezenas: int,
    limite_baixo: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dezenas_cols = [f"d{i}" for i in range(1, n_dezenas + 1)]
    registros = []
    for _, row in df_concursos[["concurso"] + dezenas_cols].iterrows():
        dezenas = [int(row[c]) for c in dezenas_cols]
        pares = sum(1 for d in dezenas if d % 2 == 0)
        impares = len(dezenas) - pares
        baixos, altos = baixos_altos(dezenas, limite_baixo)
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
    return df_padroes, dist_par_impar, dist_baixa_alta  # [file:1]


def calcular_somas(df_concursos: pd.DataFrame, n_dezenas: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dezenas_cols = [f"d{i}" for i in range(1, n_dezenas + 1)]
    df = df_concursos.copy()
    df["soma"] = df[dezenas_cols].sum(axis=1)
    bins = [0, 150, 200, 250, 300, 350, 500]
    labels = ["0-150", "151-200", "201-250", "251-300", "301-350", "351-500"]
    df["faixa_soma"] = pd.cut(df["soma"], bins=bins, labels=labels, right=True)
    dist_faixas = (
        df["faixa_soma"]
        .value_counts(dropna=False)
        .sort_index()
        .reset_index()
        .rename(columns={"index": "faixa_soma", "faixa_soma": "qtd"})
    )
    return df[["concurso", "soma", "faixa_soma"]], dist_faixas  # [file:1]

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:
    st.title("Lottery Helper")

    modalidade = st.radio(
        "Loteria",
        ["Mega-Sena", "Lotofácil"],
        help="Escolha qual loteria deseja gerar e analisar.",
    )

    if modalidade == "Mega-Sena":
        N_UNIVERSO = 60
        N_MIN = 6
        N_MAX = 15
        N_DEZENAS_HIST = 6
        CSVPATH = "historicomegasena.csv"
        preco_base = PRECO_BASE_MEGA
        COMB_TARGET = math.comb(60, 6)
        LIMITE_BAIXO = 30
    else:
        N_UNIVERSO = 25
        N_MIN = 15
        N_MAX = 20
        N_DEZENAS_HIST = 15
        CSVPATH = "historicolotofacil.csv"
        preco_base = PRECO_BASE_LOTO
        COMB_TARGET = math.comb(25, 15)
        LIMITE_BAIXO = 13

    pagina = st.radio("Navegação", ["Gerar jogos", "Análises estatísticas", "Debug Mega"])


    st.markdown("### Filtros avançados (opcional)")
    with st.expander("Restrições sobre os jogos gerados", expanded=False):
        dezenas_fixas_txt = st.text_input(
            "Dezenas fixas (sempre incluir)", placeholder="Ex: 10, 11, 12"
        )
        dezenas_proibidas_txt = st.text_input(
            "Dezenas proibidas (nunca incluir)", placeholder="Ex: 1, 2, 3"
        )
        soma_min = st.number_input("Soma mínima (opcional)", min_value=0, max_value=600, value=0)
        soma_max = st.number_input("Soma máxima (opcional)", min_value=0, max_value=600, value=0)

    def parse_lista(texto: str) -> list[int]:
        if not texto:
            return []
        return [int(x.strip()) for x in texto.replace(";", ",").split(",") if x.strip().isdigit()]

    dezenas_fixas = parse_lista(dezenas_fixas_txt)
    dezenas_proibidas = parse_lista(dezenas_proibidas_txt)
    soma_min_val = soma_min if soma_min > 0 else None
    soma_max_val = soma_max if soma_max > 0 else None

    if soma_min_val is not None and soma_max_val is not None and soma_min_val > soma_max_val:
        st.warning("A soma mínima é maior que a soma máxima. Filtros de soma serão ignorados.")
        soma_min_val, soma_max_val = None, None

    st.markdown("### Atualização automática (Caixa)")
    if modalidade == "Lotofácil":
        if st.button("Baixar e atualizar Lotofácil - Caixa"):
            try:
                atualizar_base_lotofacil_automatico()
                st.success("Base da Lotofácil baixada da Caixa e atualizada.")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao baixar/atualizar Lotofácil: {e}")
    else:
        if st.button("Baixar e atualizar Mega-Sena - Caixa"):
            try:
                atualizar_base_megasena_automatico()
                st.success("Base da Mega-Sena baixada da Caixa e atualizada.")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao baixar/atualizar Mega-Sena: {e}")

    st.markdown("### Manutenção das bases")
    if st.button("Apagar CSVs locais (reset histórico)"):
        _remover_arquivo_se_existir("historicomegasena.csv")
        _remover_arquivo_se_existir("historicolotofacil.csv")
        st.success("Arquivos CSV locais removidos. Baixe e atualize novamente as bases.")
        st.session_state.clear()
        st.rerun()

    orcamento_max = st.number_input(
        "Orçamento máximo (opcional)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=0.0,
        step=10.0,
    )

# ==========================
# CARREGAR HISTÓRICO
# ==========================

try:
    df_concursos = carregar_concursos(CSVPATH, N_DEZENAS_HIST)
    freq_df = calcular_frequencias(df_concursos, N_DEZENAS_HIST)
except FileNotFoundError:
    st.error(
        f"Arquivo de concursos não encontrado para {modalidade}. "
        "Clique em 'Baixar e atualizar ... - Caixa' na barra lateral primeiro."
    )
    st.stop()
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar/usar os dados: {e}")
    st.stop()

DEZENAS_ULTIMO_CONCURSO: set[int] = set()
if not df_concursos.empty:
    last = df_concursos.sort_values("concurso").iloc[-1]
    DEZENAS_ULTIMO_CONCURSO = {int(last[f"d{i}"]) for i in range(1, N_DEZENAS_HIST + 1)}

if "jogos" not in st.session_state:
    st.session_state["jogos"] = []
if "jogos_info" not in st.session_state:
    st.session_state["jogos_info"] = []

explicacoes = {
    "Aleatório puro": "Sorteia dezenas totalmente aleatórias dentro do universo da loteria.",
    "Balanceado par/ímpar": "Tenta evitar all-in em pares ou ímpares, mantendo alguma mistura.",
    "Quentes/Frias/Mix": "Combina dezenas mais sorteadas, menos sorteadas e neutras.",
    "Sem sequências longas": "Evita jogos com muitas dezenas consecutivas.",
}  # [file:1]

# ==========================
# PÁGINA GERAR JOGOS
# ==========================

if pagina == "Gerar jogos":
    titulo = (
        "Gerador de jogos da Mega-Sena"
        if modalidade == "Mega-Sena"
        else "Gerador de jogos da Lotofácil"
    )
    st.markdown(f"<div class='main-title'>{titulo}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-subtitle'>Escolha o modo e as estratégias na barra lateral, "
        "ajuste filtros/opções de custo e clique em <b>Gerar</b> para ver seus jogos.</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Resumo do histórico")
        st.write("DEBUG antes de tudo:", df_concursos.shape)
        st.write("DEBUG concursos min/max:", df_concursos["concurso"].min(), df_concursos["concurso"].max())
        st.metric("Total de concursos", len(df_concursos))
        st.caption(
            f"De {df_concursos['data'].min().date()} "
            f"até {df_concursos['data'].max().date()}"
        )
    with col2:
        st.markdown("### Parâmetros gerais")
        st.write(f"- Loteria: **{modalidade}**")
        st.write(f"- Universo: 1–{N_UNIVERSO}")
        st.write(f"- Valor aposta base: R$ {preco_base:,.2f}".replace(".", ","))
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

    modo_geracao = st.radio("Modo de geração", ["Uma estratégia", "Misto de estratégias"])
    gerar = False
    gerar_misto = False

    if modo_geracao == "Uma estratégia":
        st.markdown("### Estratégia de geração")
        estrategia = st.selectbox(
            "Estratégia",
            ["Aleatório puro", "Balanceado par/ímpar", "Quentes/Frias/Mix", "Sem sequências longas"],
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
        tam_jogo = st.slider("Dezenas por jogo", N_MIN, N_MAX, N_MIN)

        if estrategia == "Quentes/Frias/Mix":
            st.markdown("### Mix de dezenas")
            colq1, colq2, colq3 = st.columns(3)
            with colq1:
                q_quentes = st.number_input("Quentes", 0, tam_jogo, 5)
            with colq2:
                q_frias = st.number_input("Frias", 0, tam_jogo, 5)
            with colq3:
                q_neutras = st.number_input("Neutras", 0, tam_jogo, 5)
        if estrategia == "Sem sequências longas":
            st.markdown("### Controle de sequência")
            limite_seq = st.slider(
                "Máx. sequência permitida",
                2,
                min(10, tam_jogo),
                3,
            )

        st.markdown("---")
        gerar = st.button("Gerar jogos", type="primary")

    else:
        st.markdown("### Parâmetros básicos do misto")
        tam_jogomix = st.slider(
            "Dezenas por jogo", N_MIN, N_MAX, N_MIN, key="tam_jogomix"
        )
        st.markdown("Quantos jogos por estratégia?")
        jogos_misto: dict[str, int] = {}
        jogos_misto["Aleatório puro"] = st.number_input(
            "Aleatório puro", min_value=0, max_value=500, value=2, step=1, key="mix_qtd_aleatorio"
        )
        jogos_misto["Balanceado par/ímpar"] = st.number_input(
            "Balanceado par/ímpar", min_value=0, max_value=500, value=2, step=1, key="mix_qtd_balanceado"
        )

        with st.expander("Quentes/Frias/Mix (opcional)", expanded=False):
            jogos_misto["Quentes/Frias/Mix"] = st.number_input(
                "Jogos Quentes/Frias/Mix",
                min_value=0, max_value=500, value=2, step=1, key="mix_qtd_qfm"
            )
            colq1m, colq2m, colq3m = st.columns(3)
            with colq1m:
                mix_q_quentes = st.number_input(
                    "Quentes", 0, tam_jogomix, 5, key="mix_q_quentes"
                )
            with colq2m:
                mix_q_frias = st.number_input(
                    "Frias", 0, tam_jogomix, 5, key="mix_q_frias"
                )
            with colq3m:
                mix_q_neutras = st.number_input(
                    "Neutras", 0, tam_jogomix, 5, key="mix_q_neutras"
                )

        with st.expander("Sem sequências longas (opcional)", expanded=False):
            jogos_misto["Sem sequências longas"] = st.number_input(
                "Jogos Sem sequências longas",
                min_value=0, max_value=500, value=2, step=1, key="mix_qtd_semseq"
            )
            mix_limite_seq = st.slider(
                "Máx. sequência permitida (misto)",
                2,
                min(10, tam_jogomix),
                3,
                key="mix_limite_seq",
            )

        st.markdown("---")
        gerar_misto = st.button("Gerar jogos mistos", type="primary")

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

    jogos: list[list[int]] = st.session_state["jogos"]
    jogos_info: list[dict] = st.session_state["jogos_info"]

    if modo_geracao == "Uma estratégia" and gerar and "estrategia" in locals():
        if estrategia == "Aleatório puro":
            jogos = gerar_aleatorio_puro(int(qtd_jogos), tam_jogo, N_UNIVERSO)
        elif estrategia == "Balanceado par/ímpar":
            jogos = gerar_balanceado_par_impar(int(qtd_jogos), tam_jogo, N_UNIVERSO)
        elif estrategia == "Quentes/Frias/Mix":
            jogos = gerar_quentes_frias_mix(
                qtd_jogos=int(qtd_jogos),
                tam_jogo=tam_jogo,
                freq_df=freq_df,
                n_universo=N_UNIVERSO,
                proporcao=(q_quentes, q_frias, q_neutras),
            )
        elif estrategia == "Sem sequências longas":
            jogos = gerar_sem_sequencias(
                qtd_jogos=int(qtd_jogos),
                tam_jogo=tam_jogo,
                n_universo=N_UNIVERSO,
                limite_sequencia=limite_seq,
            )

        jogos = filtrar_jogos(
            jogos,
            dezenas_fixas=dezenas_fixas,
            dezenas_proibidas=dezenas_proibidas,
            soma_min=soma_min_val,
            soma_max=soma_max_val,
        )
        jogos = [j for j in jogos if len(j) >= N_MIN]
        jogos_info = [{"estrategia": estrategia, "jogo": j} for j in jogos]

    if modo_geracao == "Misto de estratégias" and gerar_misto:
        jogos = []
        jogos_info = []

        qtd_ap = jogos_misto.get("Aleatório puro", 0)
        if qtd_ap > 0:
            js = gerar_aleatorio_puro(int(qtd_ap), tam_jogomix, N_UNIVERSO)
            jogos.extend(js)
            jogos_info.extend({"estrategia": "Aleatório puro", "jogo": j} for j in js)

        qtd_bal = jogos_misto.get("Balanceado par/ímpar", 0)
        if qtd_bal > 0:
            js = gerar_balanceado_par_impar(int(qtd_bal), tam_jogomix, N_UNIVERSO)
            jogos.extend(js)
            jogos_info.extend({"estrategia": "Balanceado par/ímpar", "jogo": j} for j in js)

        qtd_qfm = jogos_misto.get("Quentes/Frias/Mix", 0)
        if qtd_qfm > 0:
            js = gerar_quentes_frias_mix(
                qtd_jogos=int(qtd_qfm),
                tam_jogo=tam_jogomix,
                freq_df=freq_df,
                n_universo=N_UNIVERSO,
                proporcao=(mix_q_quentes, mix_q_frias, mix_q_neutras),
            )
            jogos.extend(js)
            jogos_info.extend({"estrategia": "Quentes/Frias/Mix", "jogo": j} for j in js)

        qtd_ss = jogos_misto.get("Sem sequências longas", 0)
        if qtd_ss > 0:
            js = gerar_sem_sequencias(
                qtd_jogos=int(qtd_ss),
                tam_jogo=tam_jogomix,
                n_universo=N_UNIVERSO,
                limite_sequencia=mix_limite_seq,
            )
            jogos.extend(js)
            jogos_info.extend({"estrategia": "Sem sequências longas", "jogo": j} for j in js)

        jogos_filtrados = filtrar_jogos(
            jogos,
            dezenas_fixas=dezenas_fixas,
            dezenas_proibidas=dezenas_proibidas,
            soma_min=soma_min_val,
            soma_max=soma_max_val,
        )
        jogos_filtrados = [j for j in jogos_filtrados if len(j) >= N_MIN]
        novo_info = [info for info in jogos_info if info["jogo"] in jogos_filtrados]
        jogos = [info["jogo"] for info in novo_info]
        jogos_info = novo_info

    avisoorcamento = None
    if (gerar or gerar_misto) and jogos and orcamento_max > 0:
        jogos_dentro: list[list[int]] = []
        jogos_info_dentro: list[dict] = []
        custo_acum = 0.0
        for info in jogos_info:
            jogo = info["jogo"]
            custo_jogo = preco_aposta_loteria(len(jogo), N_MIN, preco_base)
            if custo_acum + custo_jogo > orcamento_max:
                break
            custo_acum += custo_jogo
            jogos_dentro.append(jogo)
            jogos_info_dentro.append(info)
        if len(jogos_dentro) < len(jogos):
            avisoorcamento = (
                f"Foram mantidos {len(jogos_dentro)} jogos dentro do orçamento. "
                f"{len(jogos) - len(jogos_dentro)} jogos foram descartados."
            )
        jogos = jogos_dentro
        jogos_info = jogos_info_dentro

    if (gerar or gerar_misto) and jogos:
        st.session_state["jogos"] = jogos
        st.session_state["jogos_info"] = jogos_info
    else:
        jogos = st.session_state["jogos"]
        jogos_info = st.session_state["jogos_info"]

    if not jogos and (gerar or gerar_misto):
        tab_jogos.warning(
            "Nenhum jogo foi gerado após aplicar filtros e orçamento. "
            "Revise os parâmetros na barra lateral."
        )
        tab_tabela.info("Nenhum dado para exibir ainda.")
        tab_analise.info("Nenhuma análise disponível sem jogos gerados.")
    elif not jogos:
        tab_jogos.write("Ajuste os parâmetros na barra lateral e clique em Gerar.")
        tab_tabela.write("A tabela aparecerá aqui após a geração.")
        tab_analise.write("A análise aparecerá aqui após a geração.")
    else:
        custo_total = calcular_custo_total(jogos, N_MIN, preco_base)
        prob_total = prob_premio_maximo_pacote(jogos, N_MIN, COMB_TARGET)

        with tab_jogos:
            colr1, colr2, colr3 = st.columns(3)
            with colr1:
                st.metric("Quantidade de jogos", len(jogos))
            with colr2:
                st.metric(
                    "Custo total estimado",
                    f"R$ {custo_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                )
            with colr3:
                if prob_total > 0:
                    inv = 1.0 / prob_total
                    st.metric("Chance aprox. prêmio máximo", f"1 em {inv:,.0f}".replace(",", "."))
                else:
                    st.metric("Chance aprox. prêmio máximo", "NA")
            if avisoorcamento:
                st.info(avisoorcamento)

            colleft, colcenter, colright = st.columns([1, 2, 1])
            with colcenter:
                for i, info in enumerate(jogos_info, start=1):
                    st.code(f"{i:02d} - {info['estrategia']} - {formatar_jogo(info['jogo'])}")

        with tab_tabela:
            dados = []
            for idx_global, info in enumerate(jogos_info, start=1):
                jogo = info["jogo"]
                row = {"jogo_id": idx_global, "estrategia": info["estrategia"]}
                for idx, d in enumerate(jogo, start=1):
                    row[f"d{idx}"] = d

                soma_jogo = sum(jogo)
                pares, impares = pares_impares(jogo)
                bax, alt = baixos_altos(jogo, LIMITE_BAIXO)
                nprimos = contar_primos(jogo)
                repet_ultimo = len(set(jogo) & DEZENAS_ULTIMO_CONCURSO)

                if soma_jogo < 150:
                    faixa_soma = "muito baixa"
                elif soma_jogo < 250:
                    faixa_soma = "baixa"
                elif soma_jogo < 350:
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

                row.update(
                    {
                        "soma": soma_jogo,
                        "faixa_soma": faixa_soma,
                        "pares": pares,
                        "impares": impares,
                        "baixos": bax,
                        "altos": alt,
                        "nprimos": nprimos,
                        "repeticoes_ultimo_concurso": repet_ultimo,
                        "padrao_extremo": padrao_extremo,
                    }
                )
                dados.append(row)

            jogos_df = pd.DataFrame(dados)

            def highlight_extreme(row):
                if row.get("padrao_extremo", False):
                    return ["background-color: #fee2e2"] * len(row)
                return [""] * len(row)

            styled = jogos_df.style.apply(highlight_extreme, axis=1)
            st.dataframe(styled, use_container_width=True)

            csv_data = jogos_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar CSV",
                data=csv_data,
                file_name=f"jogos_{modalidade}_{datetime.now().date()}.csv",
                mime="text/csv",
            )

        with tab_analise:
            st.subheader("Resumo do pacote atual")
            scores: list[float] = []
            extremos = 0
            for jogo in jogos:
                soma_jogo = sum(jogo)
                pares, impares = pares_impares(jogo)
                bax, alt = baixos_altos(jogo, LIMITE_BAIXO)
                nprimos = contar_primos(jogo)
                repet_ultimo = len(set(jogo) & DEZENAS_ULTIMO_CONCURSO)

                if soma_jogo < 150:
                    faixa_soma = "muito baixa"
                elif soma_jogo < 250:
                    faixa_soma = "baixa"
                elif soma_jogo < 350:
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
                if padrao_extremo:
                    extremos += 1

                score = 10.0
                if faixa_soma == "dentro do comum":
                    score += 0.5
                elif faixa_soma == "baixa":
                    score -= 0.5
                else:
                    score -= 1.0

                if pares == impares:
                    score += 0.5
                elif abs(pares - impares) <= 2:
                    score += 0.2
                else:
                    score -= 0.3

                if nprimos in (2, 3, 4):
                    score += 0.3
                elif nprimos == 0 or nprimos >= 7:
                    score -= 0.5

                if bax > 0 and alt > 0:
                    score += 0.2
                else:
                    score -= 0.5

                if repet_ultimo >= 5:
                    score -= 0.3

                score = max(0.0, min(10.0, score))
                scores.append(score)

            colr1, colr2, colr3 = st.columns(3)
            with colr1:
                st.metric(
                    "Média do score heurístico",
                    f"{np.mean(scores):.2f}" if scores else "NA",
                )
            with colr2:
                pct_ok = 100.0 * (1 - extremos / len(jogos)) if jogos else 0.0
                st.metric(
                    "% jogos sem padrão extremo",
                    f"{pct_ok:.1f}%" if jogos else "NA",
                )
            with colr3:
                media_dezenas = np.mean([len(j) for j in jogos]) if jogos else 0.0
                st.metric(
                    "Média de dezenas por jogo",
                    f"{media_dezenas:.1f}" if jogos else "NA",
                )

            st.markdown("### Cobertura das dezenas nos jogos gerados")
            todas_dezenas = [d for j in jogos for d in j]
            freq_jogos = pd.Series(todas_dezenas).value_counts().sort_index()
            df_freq_jogos = freq_jogos.reset_index()
            df_freq_jogos.columns = ["dezena", "frequencia"]
            cola1, cola2 = st.columns(2)
            with cola1:
                st.markdown("Frequência das dezenas")
                if not df_freq_jogos.empty:
                    st.bar_chart(df_freq_jogos.set_index("dezena")["frequencia"])
                else:
                    st.info("Nenhuma dezena para exibir.")
            with cola2:
                st.markdown("Top dezenas mais usadas")
                if not df_freq_jogos.empty:
                    topn = min(10, len(df_freq_jogos))
                    st.dataframe(
                        df_freq_jogos.sort_values("frequencia", ascending=False).head(topn),
                        use_container_width=True,
                    )
                else:
                    st.info("Nenhum dado de frequência disponível.")

            st.markdown("---")
            st.subheader("Simulações dos seus jogos")
            st.markdown("Simulação de acertos em um concurso")

            modosim = st.radio(
                "Escolher resultado para simulação",
                ["Usar concurso histórico", "Digitar resultado manual"],
                horizontal=True,
            )

            dezenas_sorteadas: list[int] = []
            if modosim == "Usar concurso histórico":
                ultimos_ids = (
                    df_concursos.sort_values("concurso", ascending=False)["concurso"]
                    .head(100)
                    .tolist()
                )
                concurso_escolhido = st.selectbox(
                    "Concurso para simular",
                    options=ultimos_ids,
                    format_func=lambda c: f"Concurso {c}",
                )
                linha = df_concursos.loc[
                    df_concursos["concurso"] == concurso_escolhido
                ].iloc[0]
                dezenas_sorteadas = [int(linha[f"d{i}"]) for i in range(1, N_DEZENAS_HIST + 1)]
                st.write(f"Dezenas sorteadas: {formatar_jogo(dezenas_sorteadas)}")
            else:
                import re

                placeholder = (
                    "Ex: 05, 12, 23, 34, 45, 60"
                    if modalidade == "Mega-Sena"
                    else "Ex: 01, 02, 03, ..., 25"
                )
                resultado_manual = st.text_input(
                    f"Informe {N_DEZENAS_HIST} dezenas separadas por vírgula ou espaço",
                    placeholder=placeholder,
                )
                tokens = re.split(r"[,\s]+", resultado_manual.strip())
                dezenas_sorteadas = [int(t) for t in tokens if t.isdigit()]
                if resultado_manual and len(dezenas_sorteadas) != N_DEZENAS_HIST:
                    st.warning(
                        f"Informe exatamente {N_DEZENAS_HIST} dezenas numéricas para a simulação."
                    )

            if len(dezenas_sorteadas) == N_DEZENAS_HIST:
                df_sim = simular_premios(jogos, dezenas_sorteadas)
                st.markdown("#### Resumo de acertos no sorteio escolhido")
                dist_acertos = (
                    df_sim["acertos"].value_counts().sort_index().reset_index()
                )
                dist_acertos.columns = ["acertos", "qtd_jogos"]
                st.dataframe(dist_acertos, use_container_width=True)

                st.markdown("#### Detalhamento por jogo")
                st.dataframe(df_sim, use_container_width=True)
            else:
                st.info("Informe dezenas sorteadas para simular acertos.")

# ==========================
# PÁGINA ANÁLISES
# ==========================

elif pagina == "Análises estatísticas":
    st.title(f"Análises estatísticas da {modalidade}")
    st.caption("As análises são descritivas e não garantem aumento de chance em sorteios futuros.")

    limite_baixo = 30 if modalidade == "Mega-Sena" else 13

    tab_freq, tab_padroes, tab_somas, tab_ultimos = st.tabs(
        ["Frequência & Atraso", "Par/ímpar & Baixa/Alta", "Soma das dezenas", "Últimos resultados"]
    )

    with tab_freq:
        st.subheader("Frequência e atraso por dezena")
        atraso_df = calcular_atraso(freq_df, df_concursos, N_DEZENAS_HIST)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Ordenado por frequência total")
            st.dataframe(
                freq_df.sort_values("frequencia", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
        with col2:
            st.markdown("Ordenado por atraso atual")
            st.dataframe(
                atraso_df.sort_values(["atraso_atual", "frequencia"], ascending=[False, False])
                .reset_index(drop=True),
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("Frequência recente vs total")
        nrec = st.slider(
            "Concursos recentes para analisar",
            min_value=20,
            max_value=300,
            value=50,
            step=10,
        )
        df_recent = df_concursos.sort_values("concurso", ascending=False).head(nrec)
        freq_recent = calcular_frequencias(df_recent, N_DEZENAS_HIST)
        freq_recent = freq_recent.rename(columns={"frequencia": "freq_recente"})
        freq_merge = freq_df.merge(freq_recent, on="dezena", how="left")
        freq_merge["freq_recente"] = freq_merge["freq_recente"].fillna(0).astype(int)

        colf1, colf2 = st.columns(2)
        with colf1:
            st.markdown("Top dezenas recentes (freq. recente)")
            st.dataframe(
                freq_merge.sort_values("freq_recente", ascending=False).head(20),
                use_container_width=True,
            )
        with colf2:
            st.markdown("Top dezenas históricas (freq. total)")
            st.dataframe(
                freq_merge.sort_values("frequencia", ascending=False).head(20),
                use_container_width=True,
            )

    with tab_padroes:
        st.subheader("Distribuição de par/ímpar e baixa/alta")
        df_padroes, dist_par_impar, dist_baixa_alta = calcular_padroes_par_impar_baixa_alta(
            df_concursos, N_DEZENAS_HIST, limite_baixo
        )
        st.markdown("Padrões mais frequentes")
        st.dataframe(dist_par_impar, use_container_width=True)
        st.dataframe(dist_baixa_alta, use_container_width=True)
        with st.expander("Tabela detalhada por concurso"):
            st.dataframe(df_padroes.sort_values("concurso"), use_container_width=True)

    with tab_somas:
        st.subheader("Soma das dezenas por concurso")
        df_somas, dist_faixas = calcular_somas(df_concursos, N_DEZENAS_HIST)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Tabela por concurso")
            st.dataframe(
                df_somas.sort_values("concurso").reset_index(drop=True),
                use_container_width=True,
            )
        with col2:
            st.markdown("Distribuição por faixa de soma")
            st.dataframe(dist_faixas, use_container_width=True)

    with tab_ultimos:
        st.subheader("Últimos resultados da modalidade")
        qtd_ultimos = st.slider(
            "Quantidade de concursos para exibir",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
        )
        ultimos = df_concursos.sort_values("concurso", ascending=False).head(qtd_ultimos)
        st.dataframe(
            ultimos.sort_values("concurso", ascending=True),
            use_container_width=True,
        )
        
elif pagina == "Debug Mega":
    st.title("Debug Mega-Sena")

    st.markdown("### 1. Ler XLSX manualmente (arquivo anexado ou baixado)")
    uploaded = st.file_uploader("Envie o arquivo Mega-Sena.xlsx", type=["xlsx"])
    if uploaded is not None:
        df_raw = pd.read_excel(uploaded)
        st.write("Colunas XLSX:", list(df_raw.columns))
        st.write("Total linhas XLSX:", len(df_raw))

        cols = [
            "Concurso", "Data do Sorteio",
            "Bola1", "Bola2", "Bola3", "Bola4", "Bola5", "Bola6",
        ]
        falta = [c for c in cols if c not in df_raw.columns]
        if falta:
            st.error(f"Colunas faltando no XLSX: {falta}")
        else:
            df = df_raw[cols].copy()
            df.rename(
                columns={"Concurso": "concurso", "Data do Sorteio": "data"},
                inplace=True,
            )

            df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
            df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["concurso", "data"])
            hoje = pd.Timestamp.today().normalize()
            df = df[(df["data"] >= "1996-01-01") & (df["data"] <= hoje)]
            df["concurso"] = df["concurso"].astype(int)

            dezenas_cols = [f"Bola{i}" for i in range(1, 7)]
            for c in dezenas_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=dezenas_cols)
            for c in dezenas_cols:
                df[c] = df[c].astype(int)

            st.write("Concursos únicos após limpeza (XLSX):", df["concurso"].nunique())
            st.write("Menor concurso:", int(df["concurso"].min()))
            st.write("Maior concurso:", int(df["concurso"].max()))

            # >>> AQUI entra o botão de sobrescrever <<<
            if st.button("Sobrescrever historicomegasena.csv com XLSX limpo"):
                df_to_save = df.rename(columns={
                    "Bola1": "d1", "Bola2": "d2", "Bola3": "d3",
                    "Bola4": "d4", "Bola5": "d5", "Bola6": "d6",
                })
                df_to_save.to_csv("historicomegasena.csv", sep=";", index=False)
                st.success(f"Salvo {len(df_to_save)} linhas em historicomegasena.csv")



    st.markdown("### 2. Ler CSV gerado pelo app")
    if os.path.exists("historicomegasena.csv"):
        df_csv = pd.read_csv("historicomegasena.csv", sep=";")
        st.write("DEBUG CSV shape:", df_csv.shape)

