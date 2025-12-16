# app.py
import math
import re
from datetime import datetime
from io import BytesIO
from typing import Literal

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ==========================
# CONFIG
# ==========================

st.set_page_config(page_title="Lottery Helper", page_icon="🎰", layout="wide", initial_sidebar_state="expanded")


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0f172a0d; }
        .main-title { font-size: 2.0rem; font-weight: 700; margin-bottom: 0.25rem; }
        .main-subtitle { font-size: 0.9rem; color: #6b7280; margin-bottom: 0.75rem; }
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            padding: 0.75rem 0.9rem;
            border-radius: 0.75rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 3px rgba(15,23,42,0.08);
        }
        div[data-testid="metric-container"] > label { font-size: 0.8rem; color: #6b7280; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
        .stTabs [data-baseweb="tab"] {
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            background-color: #e5e7eb33;
        }
        .stTabs [aria-selected="true"] { background-color: #111827; color: #f9fafb; }
        .stDataFrame thead tr th { font-size: 0.80rem; padding-top: 0.4rem; padding-bottom: 0.4rem; }
        .stDataFrame tbody tr td { font-size: 0.80rem; padding-top: 0.25rem; padding-bottom: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_css()

# Preços base (ajuste se quiser)
PRECO_BASE_MEGA = 6.00
PRECO_BASE_LOTO = 3.50

# Endpoints de download do histórico (XLSX)
URL_LOTOFACIL_DOWNLOAD = (
    "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/download"
    "?modalidade=Lotof%C3%A1cil"
)
URL_MEGA_DOWNLOAD = (
    "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/download"
    "?modalidade=Mega-Sena"
)

Modalidade = Literal["Mega-Sena", "Lotofácil"]


# ==========================
# UTILITÁRIOS
# ==========================

def money_ptbr(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def parse_lista(texto: str) -> list[int]:
    """Aceita: '1 2 3', '1,2,3', '1; 2;3'. Remove duplicados preservando ordem."""
    if not texto:
        return []
    tokens = re.split(r"[,\s;]+", texto.strip())
    out: list[int] = []
    seen: set[int] = set()
    for t in tokens:
        if not t:
            continue
        if t.isdigit():
            v = int(t)
            if v not in seen:
                out.append(v)
                seen.add(v)
    return out


def validar_dezenas(lista: list[int], n_universo: int, nome: str) -> None:
    if len(set(lista)) != len(lista):
        raise ValueError(f"{nome}: há dezenas repetidas.")
    if any((d < 1 or d > n_universo) for d in lista):
        raise ValueError(f"{nome}: há dezenas fora do intervalo 1–{n_universo}.")


def formatar_jogo(jogo: list[int]) -> str:
    return " - ".join(f"{d:02d}" for d in sorted(jogo))


def pares_impares(jogo: list[int]) -> tuple[int, int]:
    pares = sum(1 for d in jogo if d % 2 == 0)
    return pares, len(jogo) - pares


def baixos_altos(jogo: list[int], limite_baixo: int) -> tuple[int, int]:
    baixos = sum(1 for d in jogo if 1 <= d <= limite_baixo)
    return baixos, len(jogo) - baixos


def tem_sequencia_longa(jogo: list[int], limite: int = 3) -> bool:
    j = sorted(jogo)
    atual = 1
    for i in range(1, len(j)):
        if j[i] == j[i - 1] + 1:
            atual += 1
            if atual >= limite:
                return True
        else:
            atual = 1
    return False


PRIMOS_ATE_60 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59}


def contar_primos(jogo: list[int]) -> int:
    return sum(1 for d in jogo if d in PRIMOS_ATE_60)


# ==========================
# HISTÓRICO (IN-MEMORY)
# ==========================

def baixar_xlsx(url: str) -> BytesIO:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return BytesIO(r.content)


def _limpar_concurso_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["concurso", "data"])

    hoje = pd.Timestamp.today().normalize()
    df = df[(df["data"] >= "1996-01-01") & (df["data"] <= hoje)]
    df["concurso"] = df["concurso"].astype(int)

    # Remove duplicados de concurso (mantém último por data)
    df = df.sort_values(["concurso", "data"]).drop_duplicates(subset=["concurso"], keep="last")
    df = df.sort_values("concurso").reset_index(drop=True)
    return df


def normalizar_df_megasena(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["Concurso", "Data do Sorteio", "Bola1", "Bola2", "Bola3", "Bola4", "Bola5", "Bola6"]
    faltando = [c for c in cols if c not in df_raw.columns]
    if faltando:
        raise RuntimeError(f"XLSX Mega-Sena inválido; colunas ausentes: {faltando}")

    df = df_raw[cols].copy()
    df.rename(columns={"Concurso": "concurso", "Data do Sorteio": "data"}, inplace=True)
    df = _limpar_concurso_data(df)

    bolas = [f"Bola{i}" for i in range(1, 7)]
    for c in bolas:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=bolas)
    for c in bolas:
        df[c] = df[c].astype(int)
        df = df[df[c].between(1, 60)]

    df.rename(columns={f"Bola{i}": f"d{i}" for i in range(1, 7)}, inplace=True)
    dezenas = [f"d{i}" for i in range(1, 7)]
    df[dezenas] = np.sort(df[dezenas].values, axis=1)
    df = df[["concurso", "data"] + dezenas].sort_values("concurso").reset_index(drop=True)
    return df


def normalizar_df_lotofacil(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["Concurso", "Data Sorteio"] + [f"Bola{i}" for i in range(1, 16)]
    faltando = [c for c in cols if c not in df_raw.columns]
    if faltando:
        raise RuntimeError(f"XLSX Lotofácil inválido; colunas ausentes: {faltando}")

    df = df_raw[cols].copy()
    df.rename(columns={"Concurso": "concurso", "Data Sorteio": "data"}, inplace=True)
    df = _limpar_concurso_data(df)

    bolas = [f"Bola{i}" for i in range(1, 16)]
    for c in bolas:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=bolas)
    for c in bolas:
        df[c] = df[c].astype(int)
        df = df[df[c].between(1, 25)]

    df.rename(columns={f"Bola{i}": f"d{i}" for i in range(1, 16)}, inplace=True)
    dezenas = [f"d{i}" for i in range(1, 16)]
    df[dezenas] = np.sort(df[dezenas].values, axis=1)
    df = df[["concurso", "data"] + dezenas].sort_values("concurso").reset_index(drop=True)
    return df


def carregar_historico_inmemory(modalidade: Modalidade) -> pd.DataFrame:
    """
    Histórico por sessão: guarda em st.session_state para reaproveitar entre reruns na mesma sessão.
    """
    key = f"historico_df::{modalidade}"
    if key in st.session_state:
        return st.session_state[key]

    if modalidade == "Mega-Sena":
        buf = baixar_xlsx(URL_MEGA_DOWNLOAD)
        df_raw = pd.read_excel(buf)
        df = normalizar_df_megasena(df_raw)
    else:
        buf = baixar_xlsx(URL_LOTOFACIL_DOWNLOAD)
        df_raw = pd.read_excel(buf)
        df = normalizar_df_lotofacil(df_raw)

    st.session_state[key] = df
    return df


def limpar_historico_sessao(modalidade: Modalidade) -> None:
    key = f"historico_df::{modalidade}"
    if key in st.session_state:
        del st.session_state[key]


# ==========================
# GERAÇÃO DE JOGOS
# ==========================

def gerar_aleatorio_puro(qtd: int, tam: int, n_universo: int) -> list[list[int]]:
    universe = np.arange(1, n_universo + 1)
    jogos: list[list[int]] = []
    for _ in range(qtd):
        dezenas = np.random.choice(universe, size=tam, replace=False)
        jogos.append(sorted(dezenas.tolist()))
    return jogos


def gerar_balanceado_par_impar(qtd: int, tam: int, n_universo: int) -> list[list[int]]:
    universe = np.arange(1, n_universo + 1)
    jogos: list[list[int]] = []
    for _ in range(qtd):
        tent = 0
        while True:
            tent += 1
            dezenas = np.random.choice(universe, size=tam, replace=False).tolist()
            p, i = pares_impares(dezenas)
            if p not in (0, tam) and i not in (0, tam):
                jogos.append(sorted(dezenas))
                break
            if tent > 50:
                jogos.append(sorted(dezenas))
                break
    return jogos


def gerar_quentes_frias_mix(
    qtd: int,
    tam: int,
    freq_df: pd.DataFrame,
    n_universo: int,
    proporcao: tuple[int, int, int],
) -> list[list[int]]:
    q_quentes, q_frias, q_neutras = proporcao

    freq_ord = freq_df.sort_values("frequencia", ascending=False)
    quentes = freq_ord["dezena"].values[:10]

    frias_raw = freq_df.sort_values("frequencia", ascending=True)["dezena"].values[:10]
    frias = np.setdiff1d(frias_raw, quentes)  # evita sobreposição

    neutras = np.setdiff1d(np.arange(1, n_universo + 1), np.union1d(quentes, frias))

    jogos: list[list[int]] = []
    for _ in range(qtd):
        dezenas: list[int] = []

        qq = min(q_quentes, tam)
        qf = min(q_frias, max(0, tam - qq))
        qn = min(q_neutras, max(0, tam - qq - qf))

        if qq > 0 and len(quentes) > 0:
            dezenas.extend(np.random.choice(quentes, size=min(qq, len(quentes)), replace=False).tolist())
        if qf > 0 and len(frias) > 0:
            dezenas.extend(np.random.choice(frias, size=min(qf, len(frias)), replace=False).tolist())
        if qn > 0 and len(neutras) > 0:
            dezenas.extend(np.random.choice(neutras, size=min(qn, len(neutras)), replace=False).tolist())

        if len(dezenas) < tam:
            rest = np.setdiff1d(np.arange(1, n_universo + 1), np.array(dezenas, dtype=int))
            extra = np.random.choice(rest, size=tam - len(dezenas), replace=False).tolist()
            dezenas.extend(extra)

        jogos.append(sorted(list(map(int, dezenas))))
    return jogos


def gerar_sem_sequencias(qtd: int, tam: int, n_universo: int, limite: int) -> list[list[int]]:
    universe = np.arange(1, n_universo + 1)
    jogos: list[list[int]] = []
    for _ in range(qtd):
        tent = 0
        while True:
            tent += 1
            dezenas = np.random.choice(universe, size=tam, replace=False).tolist()
            if not tem_sequencia_longa(dezenas, limite=limite):
                jogos.append(sorted(dezenas))
                break
            if tent > 100:
                jogos.append(sorted(dezenas))
                break
    return jogos


def filtrar_jogo(
    jogo: list[int],
    dezenas_fixas: list[int],
    dezenas_proibidas: list[int],
    soma_min: int | None,
    soma_max: int | None,
) -> bool:
    s = set(jogo)
    if dezenas_fixas and not set(dezenas_fixas).issubset(s):
        return False
    if dezenas_proibidas and (set(dezenas_proibidas) & s):
        return False
    soma = sum(jogo)
    if soma_min is not None and soma < soma_min:
        return False
    if soma_max is not None and soma > soma_max:
        return False
    return True


def preco_aposta_loteria(n_dezenas: int, n_min_base: int, preco_base: float) -> float:
    if n_dezenas < n_min_base:
        return 0.0
    return math.comb(n_dezenas, n_min_base) * preco_base


def calcular_custo_total(jogos: list[list[int]], n_min_base: int, preco_base: float) -> float:
    return sum(preco_aposta_loteria(len(j), n_min_base, preco_base) for j in jogos)


def prob_premio_maximo_pacote(jogos: list[list[int]], n_min_base: int, comb_target: int) -> float:
    """
    Aproximação (assume independência entre jogos). Útil como noção de ordem de grandeza.
    """
    prob_nao = 1.0
    for jogo in jogos:
        if len(jogo) < n_min_base:
            continue
        p = math.comb(len(jogo), n_min_base) / comb_target
        prob_nao *= (1.0 - p)
    return 1.0 - prob_nao


# ==========================
# ANÁLISES (HISTÓRICO)
# ==========================

def calcular_frequencias(df: pd.DataFrame, n_dezenas_sorteio: int, n_universo: int) -> pd.DataFrame:
    dezenas_cols = [f"d{i}" for i in range(1, n_dezenas_sorteio + 1)]
    todas = df[dezenas_cols].values.ravel()
    freq = pd.Series(todas).value_counts().reindex(range(1, n_universo + 1), fill_value=0).sort_index()
    out = freq.reset_index()
    out.columns = ["dezena", "frequencia"]
    out["dezena"] = out["dezena"].astype(int)
    out["frequencia"] = out["frequencia"].astype(int)
    return out


def calcular_atraso(freq_df: pd.DataFrame, df: pd.DataFrame, n_dezenas_sorteio: int, n_universo: int) -> pd.DataFrame:
    dezenas_cols = [f"d{i}" for i in range(1, n_dezenas_sorteio + 1)]
    ultimo: dict[int, int] = {}

    for _, row in df[["concurso"] + dezenas_cols].iterrows():
        conc = int(row["concurso"])
        for d in row[dezenas_cols]:
            ultimo[int(d)] = conc

    max_conc = int(df["concurso"].max())
    linhas = []
    for dezena in range(1, n_universo + 1):
        fr = int(freq_df.loc[freq_df["dezena"] == dezena, "frequencia"].iloc[0])
        ult = ultimo.get(dezena)
        atraso = None if ult is None else max_conc - ult
        linhas.append({"dezena": dezena, "frequencia": fr, "ultimo_concurso": ult, "atraso_atual": atraso})
    return pd.DataFrame(linhas)


def calcular_padroes_par_impar_baixa_alta(
    df: pd.DataFrame, n_dezenas_sorteio: int, limite_baixo: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dezenas_cols = [f"d{i}" for i in range(1, n_dezenas_sorteio + 1)]
    registros = []
    for _, row in df[["concurso"] + dezenas_cols].iterrows():
        dezenas = [int(row[c]) for c in dezenas_cols]
        pares = sum(1 for d in dezenas if d % 2 == 0)
        impares = len(dezenas) - pares
        baixos, altos = baixos_altos(dezenas, limite_baixo)
        registros.append({"concurso": int(row["concurso"]), "pares": pares, "impares": impares, "baixos": baixos, "altos": altos})

    dfp = pd.DataFrame(registros)
    dist_pi = dfp.groupby(["pares", "impares"]).size().reset_index(name="qtd").sort_values("qtd", ascending=False).reset_index(drop=True)
    dist_ba = dfp.groupby(["baixos", "altos"]).size().reset_index(name="qtd").sort_values("qtd", ascending=False).reset_index(drop=True)
    return dfp, dist_pi, dist_ba


def calcular_somas(df: pd.DataFrame, n_dezenas_sorteio: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dezenas_cols = [f"d{i}" for i in range(1, n_dezenas_sorteio + 1)]
    dfx = df.copy()
    dfx["soma"] = dfx[dezenas_cols].sum(axis=1)

    bins = [0, 150, 200, 250, 300, 350, 500]
    labels = ["0-150", "151-200", "201-250", "251-300", "301-350", "351-500"]
    dfx["faixa_soma"] = pd.cut(dfx["soma"], bins=bins, labels=labels, right=True)

    dist = (
        dfx["faixa_soma"]
        .value_counts(dropna=False)
        .sort_index()
        .reset_index()
        .rename(columns={"index": "faixa_soma", "faixa_soma": "qtd"})
    )
    return dfx[["concurso", "soma", "faixa_soma"]], dist


def simular_acertos(jogos: list[list[int]], dezenas_sorteadas: list[int]) -> pd.DataFrame:
    s = set(dezenas_sorteadas)
    linhas = []
    for i, jogo in enumerate(jogos, start=1):
        linhas.append({"jogo_id": i, "jogo": formatar_jogo(jogo), "acertos": len(set(jogo) & s)})
    return pd.DataFrame(linhas)


# ==========================
# SIDEBAR (CONTROLES)
# ==========================

with st.sidebar:
    st.title("Lottery Helper")

    modalidade: Modalidade = st.radio("Loteria", ["Mega-Sena", "Lotofácil"])

    if modalidade == "Mega-Sena":
        N_UNIVERSO = 60
        N_MIN = 6
        N_MAX = 15
        N_DEZENAS_HIST = 6
        preco_base = PRECO_BASE_MEGA
        COMB_TARGET = math.comb(60, 6)
        LIMITE_BAIXO = 30
    else:
        N_UNIVERSO = 25
        N_MIN = 15
        N_MAX = 20
        N_DEZENAS_HIST = 15
        preco_base = PRECO_BASE_LOTO
        COMB_TARGET = math.comb(25, 15)
        LIMITE_BAIXO = 13

    pagina = st.radio("Navegação", ["Gerar jogos", "Análises estatísticas"])

    st.markdown("### Histórico (in-memory)")
    colb1, colb2 = st.columns(2)
    with colb1:
        if st.button("Baixar/Recarregar"):
            limpar_historico_sessao(modalidade)
            st.rerun()
    with colb2:
        if st.button("Limpar sessão"):
            limpar_historico_sessao(modalidade)
            # também limpa jogos gerados
            st.session_state.pop("jogos", None)
            st.session_state.pop("jogos_info", None)
            st.rerun()

    st.markdown("### Filtros (opcional)")
    with st.expander("Restrições sobre os jogos gerados", expanded=False):
        dezenas_fixas_txt = st.text_input("Dezenas fixas (sempre incluir)", placeholder="Ex: 10, 11, 12")
        dezenas_proibidas_txt = st.text_input("Dezenas proibidas (nunca incluir)", placeholder="Ex: 1, 2, 3")
        soma_min = st.number_input("Soma mínima", min_value=0, max_value=1000, value=0, step=1)
        soma_max = st.number_input("Soma máxima", min_value=0, max_value=1000, value=0, step=1)

    dezenas_fixas = parse_lista(dezenas_fixas_txt)
    dezenas_proibidas = parse_lista(dezenas_proibidas_txt)
    soma_min_val = soma_min if soma_min > 0 else None
    soma_max_val = soma_max if soma_max > 0 else None

    orcamento_max = st.number_input("Orçamento máximo (opcional)", min_value=0.0, max_value=1_000_000.0, value=0.0, step=10.0)

    # validações de filtros
    try:
        validar_dezenas(dezenas_fixas, N_UNIVERSO, "Dezenas fixas")
        validar_dezenas(dezenas_proibidas, N_UNIVERSO, "Dezenas proibidas")
        conflito = set(dezenas_fixas) & set(dezenas_proibidas)
        if conflito:
            raise ValueError(f"Conflito: {sorted(conflito)} está em fixas e proibidas.")
        if soma_min_val is not None and soma_max_val is not None and soma_min_val > soma_max_val:
            st.warning("Soma mínima > soma máxima. Filtros de soma serão ignorados.")
            soma_min_val, soma_max_val = None, None
    except ValueError as e:
        st.error(str(e))
        st.stop()


# ==========================
# CARREGAR HISTÓRICO (IN-MEMORY)
# ==========================

with st.spinner("Carregando histórico..."):
    try:
        df_concursos = carregar_historico_inmemory(modalidade)
    except Exception as e:
        st.error(f"Falha ao carregar histórico: {e}")
        st.stop()

# checklist (para você ver “total de concursos” correto)
with st.sidebar:
    st.markdown("### Checklist da base")
    st.caption(f"Linhas: {len(df_concursos)}")
    st.caption(f"Concurso min/max: {df_concursos['concurso'].min()} / {df_concursos['concurso'].max()}")
    st.caption(f"Data min/max: {df_concursos['data'].min().date()} / {df_concursos['data'].max().date()}")


freq_df = calcular_frequencias(df_concursos, N_DEZENAS_HIST, N_UNIVERSO)

DEZENAS_ULTIMO_CONCURSO: set[int] = set()
if not df_concursos.empty:
    last = df_concursos.iloc[-1]
    DEZENAS_ULTIMO_CONCURSO = {int(last[f"d{i}"]) for i in range(1, N_DEZENAS_HIST + 1)}


# ==========================
# SESSION STATE (JOGOS)
# ==========================

if "jogos" not in st.session_state:
    st.session_state["jogos"] = []
if "jogos_info" not in st.session_state:
    st.session_state["jogos_info"] = []

explicacoes = {
    "Aleatório puro": "Sorteia dezenas totalmente aleatórias dentro do universo.",
    "Balanceado par/ímpar": "Evita jogos 100% pares ou 100% ímpares (tenta até 50 vezes).",
    "Quentes/Frias/Mix": "Mistura dezenas com maior frequência, menor frequência e neutras.",
    "Sem sequências longas": "Evita sequências consecutivas longas (tenta até 100 vezes).",
}


# ==========================
# PÁGINA: GERAR JOGOS
# ==========================

if pagina == "Gerar jogos":
    titulo = "Gerador de jogos da Mega-Sena" if modalidade == "Mega-Sena" else "Gerador de jogos da Lotofácil"
    st.markdown(f"<div class='main-title'>{titulo}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-subtitle'>Gere jogos e avalie padrões (apenas descritivo). "
        "Use o checklist na sidebar para conferir o total de concursos carregados.</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Resumo do histórico")
        st.metric("Total de concursos", len(df_concursos))
        st.caption(f"De {df_concursos['data'].min().date()} até {df_concursos['data'].max().date()}")
        st.caption(f"Último concurso: {int(df_concursos['concurso'].max())}")

    with col2:
        st.markdown("### Parâmetros")
        st.write(f"- Loteria: **{modalidade}**")
        st.write(f"- Universo: 1–{N_UNIVERSO}")
        st.write(f"- Aposta base: {money_ptbr(preco_base)}")
        if dezenas_fixas:
            st.write(f"- Fixas: {sorted(dezenas_fixas)}")
        if dezenas_proibidas:
            st.write(f"- Proibidas: {sorted(dezenas_proibidas)}")
        if orcamento_max > 0:
            st.write(f"- Orçamento máx.: {money_ptbr(orcamento_max)}")

    st.divider()

    modo = st.radio("Modo de geração", ["Uma estratégia", "Misto de estratégias"])

    gerar = False
    gerar_misto = False

    if modo == "Uma estratégia":
        estrategia = st.selectbox("Estratégia", list(explicacoes.keys()))
        qtd_jogos = st.number_input("Quantidade de jogos", min_value=1, max_value=500, value=10, step=1)
        tam_jogo = st.slider("Dezenas por jogo", N_MIN, N_MAX, N_MIN)

        q_quentes = q_frias = q_neutras = 0
        limite_seq = 3

        if estrategia == "Quentes/Frias/Mix":
            colq1, colq2, colq3 = st.columns(3)
            with colq1:
                q_quentes = st.number_input("Quentes", 0, tam_jogo, min(5, tam_jogo))
            with colq2:
                q_frias = st.number_input("Frias", 0, tam_jogo, min(5, tam_jogo))
            with colq3:
                q_neutras = st.number_input("Neutras", 0, tam_jogo, max(0, tam_jogo - q_quentes - q_frias))

        if estrategia == "Sem sequências longas":
            limite_seq = st.slider("Máx. sequência permitida", 2, min(10, tam_jogo), 3)

        with st.expander("Como funciona?", expanded=False):
            st.write(explicacoes[estrategia])

        gerar = st.button("Gerar jogos", type="primary")

    else:
        tam_jogomix = st.slider("Dezenas por jogo", N_MIN, N_MAX, N_MIN, key="tam_jogomix")

        jogos_misto: dict[str, int] = {}
        st.markdown("Quantidade por estratégia:")
        jogos_misto["Aleatório puro"] = st.number_input("Aleatório puro", min_value=0, max_value=500, value=2, step=1)
        jogos_misto["Balanceado par/ímpar"] = st.number_input("Balanceado par/ímpar", min_value=0, max_value=500, value=2, step=1, key="mix_bal")

        with st.expander("Quentes/Frias/Mix", expanded=False):
            jogos_misto["Quentes/Frias/Mix"] = st.number_input("Jogos Quentes/Frias/Mix", min_value=0, max_value=500, value=2, step=1)
            colq1m, colq2m, colq3m = st.columns(3)
            with colq1m:
                mix_q_quentes = st.number_input("Quentes (misto)", 0, tam_jogomix, min(5, tam_jogomix))
            with colq2m:
                mix_q_frias = st.number_input("Frias (misto)", 0, tam_jogomix, min(5, tam_jogomix))
            with colq3m:
                mix_q_neutras = st.number_input("Neutras (misto)", 0, tam_jogomix, max(0, tam_jogomix - mix_q_quentes - mix_q_frias))

        with st.expander("Sem sequências longas", expanded=False):
            jogos_misto["Sem sequências longas"] = st.number_input("Jogos Sem sequências longas", min_value=0, max_value=500, value=2, step=1)
            mix_limite_seq = st.slider("Máx. sequência (misto)", 2, min(10, tam_jogomix), 3)

        gerar_misto = st.button("Gerar jogos mistos", type="primary")

    tab_jogos, tab_tabela, tab_analise = st.tabs(["Jogos gerados", "Tabela / Exportar", "Análise & Simulação"])

    jogos: list[list[int]] = st.session_state["jogos"]
    jogos_info: list[dict] = st.session_state["jogos_info"]

    # ===== geração
    if modo == "Uma estratégia" and gerar:
        if estrategia == "Aleatório puro":
            jogos = gerar_aleatorio_puro(int(qtd_jogos), int(tam_jogo), N_UNIVERSO)
        elif estrategia == "Balanceado par/ímpar":
            jogos = gerar_balanceado_par_impar(int(qtd_jogos), int(tam_jogo), N_UNIVERSO)
        elif estrategia == "Quentes/Frias/Mix":
            jogos = gerar_quentes_frias_mix(
                qtd=int(qtd_jogos),
                tam=int(tam_jogo),
                freq_df=freq_df,
                n_universo=N_UNIVERSO,
                proporcao=(int(q_quentes), int(q_frias), int(q_neutras)),
            )
        else:
            jogos = gerar_sem_sequencias(int(qtd_jogos), int(tam_jogo), N_UNIVERSO, int(limite_seq))

        # aplica filtros
        jogos = [j for j in jogos if filtrar_jogo(j, dezenas_fixas, dezenas_proibidas, soma_min_val, soma_max_val)]
        jogos_info = [{"estrategia": estrategia, "jogo": j} for j in jogos]

    if modo == "Misto de estratégias" and gerar_misto:
        jogos_info = []

        if int(jogos_misto.get("Aleatório puro", 0)) > 0:
            js = gerar_aleatorio_puro(int(jogos_misto["Aleatório puro"]), int(tam_jogomix), N_UNIVERSO)
            jogos_info.extend({"estrategia": "Aleatório puro", "jogo": j} for j in js)

        if int(jogos_misto.get("Balanceado par/ímpar", 0)) > 0:
            js = gerar_balanceado_par_impar(int(jogos_misto["Balanceado par/ímpar"]), int(tam_jogomix), N_UNIVERSO)
            jogos_info.extend({"estrategia": "Balanceado par/ímpar", "jogo": j} for j in js)

        if int(jogos_misto.get("Quentes/Frias/Mix", 0)) > 0:
            js = gerar_quentes_frias_mix(
                qtd=int(jogos_misto["Quentes/Frias/Mix"]),
                tam=int(tam_jogomix),
                freq_df=freq_df,
                n_universo=N_UNIVERSO,
                proporcao=(int(mix_q_quentes), int(mix_q_frias), int(mix_q_neutras)),
            )
            jogos_info.extend({"estrategia": "Quentes/Frias/Mix", "jogo": j} for j in js)

        if int(jogos_misto.get("Sem sequências longas", 0)) > 0:
            js = gerar_sem_sequencias(int(jogos_misto["Sem sequências longas"]), int(tam_jogomix), N_UNIVERSO, int(mix_limite_seq))
            jogos_info.extend({"estrategia": "Sem sequências longas", "jogo": j} for j in js)

        # filtra mantendo alinhamento
        jogos_info = [
            info for info in jogos_info
            if filtrar_jogo(info["jogo"], dezenas_fixas, dezenas_proibidas, soma_min_val, soma_max_val)
        ]
        jogos = [info["jogo"] for info in jogos_info]

    # ===== orçamento
    avisoorc = None
    if (gerar or gerar_misto) and jogos and orcamento_max > 0:
        dentro_info: list[dict] = []
        custo_acum = 0.0
        for info in jogos_info:
            jogo = info["jogo"]
            custo = preco_aposta_loteria(len(jogo), N_MIN, preco_base)
            if custo_acum + custo > orcamento_max:
                break
            custo_acum += custo
            dentro_info.append(info)

        if len(dentro_info) < len(jogos_info):
            avisoorc = f"Mantidos {len(dentro_info)} jogos no orçamento; descartados {len(jogos_info) - len(dentro_info)}."

        jogos_info = dentro_info
        jogos = [info["jogo"] for info in jogos_info]

    # ===== persistência (session_state)
    if (gerar or gerar_misto) and jogos:
        st.session_state["jogos"] = jogos
        st.session_state["jogos_info"] = jogos_info
    else:
        jogos = st.session_state["jogos"]
        jogos_info = st.session_state["jogos_info"]

    # ===== UI
    if not jogos:
        with tab_jogos:
            st.info("Gere jogos para exibir resultados.")
    else:
        custo_total = calcular_custo_total(jogos, N_MIN, preco_base)
        prob_total = prob_premio_maximo_pacote(jogos, N_MIN, COMB_TARGET)

        with tab_jogos:
            c1, c2, c3 = st.columns(3)
            c1.metric("Quantidade de jogos", len(jogos))
            c2.metric("Custo total estimado", money_ptbr(custo_total))
            c3.metric("Chance aprox. prêmio máximo", ("NA" if prob_total <= 0 else f"1 em {1.0 / prob_total:,.0f}".replace(",", ".")))

            if avisoorc:
                st.info(avisoorc)

            colleft, colcenter, colright = st.columns([1, 2, 1])
            with colcenter:
                for i, info in enumerate(jogos_info, start=1):
                    st.code(f"{i:02d} - {info['estrategia']} - {formatar_jogo(info['jogo'])}")

        with tab_tabela:
            dados = []
            for idx, info in enumerate(jogos_info, start=1):
                jogo = sorted(info["jogo"])
                row = {"jogo_id": idx, "estrategia": info["estrategia"]}

                for jdx, d in enumerate(jogo, start=1):
                    row[f"d{jdx}"] = int(d)

                soma = sum(jogo)
                p, i = pares_impares(jogo)
                b, a = baixos_altos(jogo, LIMITE_BAIXO)
                primos = contar_primos(jogo)
                rep_ult = len(set(jogo) & DEZENAS_ULTIMO_CONCURSO)

                if soma < 150:
                    faixa = "muito baixa"
                elif soma < 250:
                    faixa = "baixa"
                elif soma < 350:
                    faixa = "dentro do comum"
                else:
                    faixa = "alta/muito alta"

                extremo = (p == 0 or i == 0 or b == 0 or a == 0 or faixa in ("muito baixa", "alta/muito alta"))

                row.update(
                    {
                        "soma": soma,
                        "faixa_soma": faixa,
                        "pares": p,
                        "impares": i,
                        "baixos": b,
                        "altos": a,
                        "nprimos": primos,
                        "repeticoes_ultimo_concurso": rep_ult,
                        "padrao_extremo": extremo,
                    }
                )
                dados.append(row)

            jogos_df = pd.DataFrame(dados)

            def highlight_extreme(r):
                return ["background-color: #fee2e2"] * len(r) if r.get("padrao_extremo", False) else [""] * len(r)

            st.dataframe(jogos_df.style.apply(highlight_extreme, axis=1), use_container_width=True)

            csv_data = jogos_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar CSV",
                data=csv_data,
                file_name=f"jogos_{modalidade}_{datetime.now().date()}.csv",
                mime="text/csv",
            )

        with tab_analise:
            st.subheader("Cobertura das dezenas nos jogos gerados")
            todas = [d for j in jogos for d in j]
            if todas:
                s = pd.Series(todas).value_counts().sort_index()
                dff = s.reset_index()
                dff.columns = ["dezena", "frequencia"]
                colA, colB = st.columns(2)
                colA.bar_chart(dff.set_index("dezena")["frequencia"])
                colB.dataframe(dff.sort_values("frequencia", ascending=False).head(10), use_container_width=True)

            st.markdown("---")
            st.subheader("Simulação de acertos")
            modo_sim = st.radio("Escolher resultado", ["Usar concurso histórico", "Digitar resultado manual"], horizontal=True)

            dezenas_sorteadas: list[int] = []
            if modo_sim == "Usar concurso histórico":
                ultimos = df_concursos.sort_values("concurso", ascending=False)["concurso"].head(100).tolist()
                conc = st.selectbox("Concurso", ultimos, format_func=lambda x: f"Concurso {x}")
                linha = df_concursos.loc[df_concursos["concurso"] == conc].iloc[0]
                dezenas_sorteadas = [int(linha[f"d{i}"]) for i in range(1, N_DEZENAS_HIST + 1)]
                st.write("Dezenas sorteadas:", formatar_jogo(dezenas_sorteadas))
            else:
                txt = st.text_input(f"Informe {N_DEZENAS_HIST} dezenas (vírgula/space/;)", placeholder="Ex: 05, 12, 23, ...")
                dezenas_sorteadas = parse_lista(txt)
                if txt and len(dezenas_sorteadas) != N_DEZENAS_HIST:
                    st.warning(f"Informe exatamente {N_DEZENAS_HIST} dezenas.")

            if len(dezenas_sorteadas) == N_DEZENAS_HIST:
                try:
                    validar_dezenas(dezenas_sorteadas, N_UNIVERSO, "Resultado")
                except ValueError as e:
                    st.error(str(e))
                else:
                    dfsim = simular_acertos(jogos, dezenas_sorteadas)
                    dist = dfsim["acertos"].value_counts().sort_index().reset_index()
                    dist.columns = ["acertos", "qtd_jogos"]
                    st.dataframe(dist, use_container_width=True)
                    st.dataframe(dfsim, use_container_width=True)


# ==========================
# PÁGINA: ANÁLISES ESTATÍSTICAS
# ==========================

elif pagina == "Análises estatísticas":
    st.title(f"Análises estatísticas — {modalidade}")
    st.caption("Análises descritivas; não garantem aumento de chance.")

    tab_freq, tab_padroes, tab_somas, tab_ultimos = st.tabs(
        ["Frequência & Atraso", "Par/ímpar & Baixa/Alta", "Soma das dezenas", "Últimos resultados"]
    )

    with tab_freq:
        st.subheader("Frequência e atraso por dezena")
        atraso_df = calcular_atraso(freq_df, df_concursos, N_DEZENAS_HIST, N_UNIVERSO)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Ordenado por frequência total")
            st.dataframe(freq_df.sort_values("frequencia", ascending=False).reset_index(drop=True), use_container_width=True)
        with c2:
            st.markdown("Ordenado por atraso atual")
            st.dataframe(
                atraso_df.sort_values(["atraso_atual", "frequencia"], ascending=[False, False]).reset_index(drop=True),
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("Frequência recente vs total")
        nrec = st.slider("Concursos recentes", min_value=20, max_value=300, value=50, step=10)
        df_recent = df_concursos.sort_values("concurso", ascending=False).head(nrec)
        freq_recent = calcular_frequencias(df_recent, N_DEZENAS_HIST, N_UNIVERSO).rename(columns={"frequencia": "freq_recente"})
        merge = freq_df.merge(freq_recent, on="dezena", how="left")
        merge["freq_recente"] = merge["freq_recente"].fillna(0).astype(int)

        c3, c4 = st.columns(2)
        c3.dataframe(merge.sort_values("freq_recente", ascending=False).head(20), use_container_width=True)
        c4.dataframe(merge.sort_values("frequencia", ascending=False).head(20), use_container_width=True)

    with tab_padroes:
        st.subheader("Distribuição de padrões")
        limite_baixo = LIMITE_BAIXO
        dfp, dist_pi, dist_ba = calcular_padroes_par_impar_baixa_alta(df_concursos, N_DEZENAS_HIST, limite_baixo)
        st.markdown("Par/ímpar")
        st.dataframe(dist_pi, use_container_width=True)
        st.markdown("Baixa/Alta")
        st.dataframe(dist_ba, use_container_width=True)
        with st.expander("Detalhado por concurso"):
            st.dataframe(dfp.sort_values("concurso"), use_container_width=True)

    with tab_somas:
        st.subheader("Soma das dezenas")
        dfs, dist = calcular_somas(df_concursos, N_DEZENAS_HIST)
        c1, c2 = st.columns(2)
        c1.dataframe(dfs.sort_values("concurso").reset_index(drop=True), use_container_width=True)
        c2.dataframe(dist, use_container_width=True)

    with tab_ultimos:
        st.subheader("Últimos resultados")
        qtd = st.slider("Quantidade", min_value=5, max_value=50, value=10, step=5)
        ult = df_concursos.sort_values("concurso", ascending=False).head(qtd)
        st.dataframe(ult.sort_values("concurso", ascending=True), use_container_width=True)
