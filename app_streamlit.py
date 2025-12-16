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

# endpoints oficiais de download de resultados [web:2]
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

    pagina = st.radio("Navegação", ["Gerar jogos", "Análises estatísticas"])

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
# RESTO DO APP (geração, tabelas, análises)
# ==========================

# A partir daqui, use exatamente o mesmo bloco de código de geração de jogos,
# tabelas e análises que já está no arquivo completo que você colou na última
# versão – não precisa mudar nada, pois ele só depende de df_concursos,
# freq_df, N_UNIVERSO/N_MIN etc., que já estão corretos agora. [file:1]
