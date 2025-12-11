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

# valor base padrão da aposta simples (6 dezenas).
# deixe isso fácil de ajustar conforme a tabela oficial / site que você usa.
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
# CUSTO E PROBABILIDADE
# ==========================
def preco_aposta_megasena(n_dezenas: int, preco_6: float) -> float:
    """
    Calcula o valor APROXIMADO de UMA aposta da Mega-Sena com n_dezenas (>= 6),
    usando combinações C(n, 6) * preço_base.
    O valor real pode ter arredondamentos/tabela específica da Caixa.
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


def prob_sena_pacote(jogos: list[list[int]]) -> float:
    """
    Probabilidade aproximada de ACERTAR a sena com pelo menos 1 jogo do pacote.
    Para cada jogo, chance ~ C(n, 6)/C(60, 6). Combinação exata: 1 - prod(1 - p_i).
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

    tab_freq, tab_padroes, tab_somas, tab_pares_trios, tab_ultimos = st.tabs(
        [
            "Frequência & Atraso",
            "Par/Ímpar & Baixa/Alta",
            "Soma das dezenas",
            "Pares & Trios",
            "Últimos resultados",
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

    # ---- ÚLTIMOS RESULTADOS ----
    with tab_ultimos:
        st.subheader("Últimos resultados da Mega-Sena (histórico local)")
        st.markdown(
            "Veja os últimos concursos disponíveis no arquivo e use-os como referência "
            "para montar ou simular jogos."
        )

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

    gerar = False
    gerar_misto = False
    modo_geracao = "Uma estratégia"

    preco_base_6 = st.number_input(
        "Valor aposta simples (6 dezenas)",
        min_value=1.0,
        max_value=50.0,
        value=PRECO_SIMPLES_6_DEFAULT,
        step=0.5,
        help=(
            "Valor usado como base para calcular o custo das apostas. "
            "Ajuste conforme a tabela oficial da Caixa ou da plataforma que você utiliza."
        ),
    )

    orcamento_max = st.number_input(
        "Orçamento máximo (opcional)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=0.0,
        step=10.0,
        help="Se maior que zero, o app tenta limitar o número de jogos ao valor máximo informado.",
    )

    # Filtros avançados (valem para qualquer modo)
    st.markdown("### Filtros avançados (opcional)")
    with st.expander("Restrições sobre os jogos gerados", expanded=False):
        dezenas_fixas_txt = st.text_input(
            "Dezenas fixas (sempre incluir)",
            placeholder="Ex: 10, 53",
            help="Essas dezenas devem aparecer em TODOS os jogos gerados após os filtros.",
        )
        dezenas_proibidas_txt = st.text_input(
            "Dezenas proibidas (nunca incluir)",
            placeholder="Ex: 1, 2, 3",
            help="Essas dezenas serão removidas dos jogos (qualquer jogo que as contenha será descartado).",
        )
        soma_min = st.number_input(
            "Soma mínima (opcional)",
            min_value=0,
            max_value=600,
            value=0,
            help="Use 0 para não aplicar limite mínimo de soma.",
        )
        soma_max = st.number_input(
            "Soma máxima (opcional)",
            min_value=0,
            max_value=600,
            value=0,
            help="Use 0 para não aplicar limite máximo de soma.",
        )

    # Parse de filtros
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
            help=(
                "• Uma estratégia: todos os jogos seguem a mesma lógica.\n"
                "• Misto de estratégias: gera jogos combinando várias estratégias de uma vez."
            ),
        )

        if modo_geracao == "Uma estratégia":
            # ---------- MODO SIMPLES ----------
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
                help="Tipo de lógica usada para montar os jogos.",
            )

            st.markdown("### Parâmetros básicos")

            qtd_jogos = st.slider(
                "Quantidade de jogos",
                1,
                1000,
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

        else:
            # ---------- MODO MISTO ----------
            st.markdown("### Parâmetros básicos do misto")

            tam_jogo_mix = st.slider(
                "Dezenas por jogo (exceto wheeling)",
                6,
                15,
                6,
                key="tam_jogo_mix",
                help="Quantidade de dezenas em cada jogo gerado pelas estratégias (exceto wheeling, que sempre gera 6).",
            )

            st.markdown("### Quantos jogos por estratégia?")
            st.caption(
                "Defina quantos jogos deseja gerar com cada estratégia. "
                "Use 0 para ignorar alguma delas."
            )

            jogos_misto: dict[str, int] = {}

            # Aleatório puro
            jogos_misto["Aleatório puro"] = st.number_input(
                "Aleatório puro",
                min_value=0,
                max_value=30,
                value=1,
                step=1,
                help="Quantidade de jogos totalmente aleatórios.",
                key="mix_qtd_aleatorio",
            )

            # Balanceado
            jogos_misto["Balanceado par/ímpar"] = st.number_input(
                "Balanceado par/ímpar",
                min_value=0,
                max_value=30,
                value=1,
                step=1,
                help="Quantidade de jogos com distribuição equilibrada de pares e ímpares.",
                key="mix_qtd_balanceado",
            )

            # Setorial
            jogos_misto["Setorial (faixas)"] = st.number_input(
                "Setorial (faixas)",
                min_value=0,
                max_value=30,
                value=1,
                step=1,
                help="Quantidade de jogos distribuindo dezenas entre 1–20, 21–40 e 41–60.",
                key="mix_qtd_setorial",
            )

            # Quentes/Frias/Mix
            with st.expander("Quentes/Frias/Mix (opcional)", expanded=False):
                jogos_misto["Quentes/Frias/Mix"] = st.number_input(
                    "Jogos Quentes/Frias/Mix",
                    min_value=0,
                    max_value=30,
                    value=1,
                    step=1,
                    help="Quantidade de jogos misturando dezenas mais sorteadas, atrasadas e neutras.",
                    key="mix_qtd_qfm",
                )
                col_q1m, col_q2m, col_q3m = st.columns(3)
                with col_q1m:
                    mix_q_quentes = st.number_input(
                        "Quentes",
                        0,
                        10,
                        3,
                        help="Dezenas mais sorteadas no histórico.",
                        key="mix_q_quentes",
                    )
                with col_q2m:
                    mix_q_frias = st.number_input(
                        "Frias",
                        0,
                        10,
                        2,
                        help="Dezenas menos sorteadas / mais atrasadas.",
                        key="mix_q_frias",
                    )
                with col_q3m:
                    mix_q_neutras = st.number_input(
                        "Neutras",
                        0,
                        10,
                        1,
                        help="Dezenas que não estão entre as mais nem entre as menos frequentes.",
                        key="mix_q_neutras",
                    )

            # Sem sequências
            with st.expander("Sem sequências longas (opcional)", expanded=False):
                jogos_misto["Sem sequências longas"] = st.number_input(
                    "Jogos Sem sequências longas",
                    min_value=0,
                    max_value=30,
                    value=1,
                    step=1,
                    help="Quantidade de jogos evitando muitas dezenas consecutivas.",
                    key="mix_qtd_sem_seq",
                )
                mix_limite_seq = st.slider(
                    "Máx. sequência permitida (misto)",
                    2,
                    6,
                    3,
                    key="mix_limite_seq",
                    help="Maior quantidade de dezenas consecutivas permitida nos jogos deste grupo.",
                )

            # Wheeling
            with st.expander("Wheeling simples (opcional)", expanded=False):
                jogos_misto["Wheeling simples (base fixa)"] = st.number_input(
                    "Jogos Wheeling simples",
                    min_value=0,
                    max_value=30,
                    value=0,
                    step=1,
                    help="Quantidade de jogos formados a partir de uma base fixa de dezenas.",
                    key="mix_qtd_wheeling",
                )
                st.text_input(
                    "Base fixa (separada por vírgulas)",
                    key="mix_base_wheeling_value",
                    placeholder="Ex: 1, 5, 12, 23, 34, 45, 56",
                    help="Informe a base para montar combinações de 6 dezenas.",
                )

            st.markdown("---")
            gerar_misto = st.button(
                "Gerar jogos mistos",
                type="primary",
                use_container_width=True,
                help="Gera jogos combinando todas as estratégias configuradas acima.",
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
        "1) Escolha o modo e as estratégias na barra lateral. "
        "2) Ajuste filtros e orçamento se quiser refinar. "
        "3) Clique em **Gerar** para ver os jogos, custo e probabilidade aproximada."
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
        st.markdown("### Modo e parâmetros")
        st.write(f"- Modo de geração: **{modo_geracao}**")
        if modo_geracao == "Uma estratégia" and "estrategia" in locals():
            st.write(f"- Estratégia: **{estrategia}**")
            if "qtd_jogos" in locals():
                st.write(f"- Jogos solicitados: **{qtd_jogos}** | Dezenas por jogo: **{tam_jogo}**")
        elif modo_geracao == "Misto de estratégias":
            st.write("- Misto de estratégias configurado na barra lateral.")
        if dezenas_fixas:
            st.write(f"- Dezenas fixas: {sorted(dezenas_fixas)}")
        if dezenas_proibidas:
            st.write(f"- Dezenas proibidas: {sorted(dezenas_proibidas)}")
        if orcamento_max > 0:
            st.write(f"- Orçamento máximo: R$ {orcamento_max:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    st.divider()

    # Explicação da estratégia em expander
    if modo_geracao == "Uma estratégia" and "estrategia" in locals():
        with st.expander("Como funciona esta estratégia?", expanded=False):
            st.write(explicacoes.get(estrategia, ""))
            st.caption(
                "Observação: todas as combinações têm a mesma chance matemática. "
                "As estratégias servem apenas para organizar a forma de jogar."
            )
    elif modo_geracao == "Misto de estratégias":
        with st.expander("O que cada estratégia faz?", expanded=False):
            for nome, desc in explicacoes.items():
                st.markdown(f"**{nome}**")
                st.write(desc)

    # Tabs para jogos, tabela/resumo e análise/simulação
    tab_jogos, tab_tabela, tab_analise = st.tabs(
        ["Jogos gerados", "Tabela / Resumo / Exportar", "Análise & Simulação"]
    )

    jogos: list[list[int]] = []
    jogos_info: list[dict] = []

    # ---------- GERAÇÃO MODO SIMPLES ----------
    if modo_geracao == "Uma estratégia" and gerar and "estrategia" in locals():
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

        # aplica filtros
        jogos = filtrar_jogos(
            jogos,
            dezenas_fixas=dezenas_fixas,
            dezenas_proibidas=dezenas_proibidas,
            soma_min=soma_min_val,
            soma_max=soma_max_val,
        )

        jogos_info = [{"estrategia": estrategia, "jogo": j} for j in jogos]

    # ---------- GERAÇÃO MODO MISTO ----------
    if modo_geracao == "Misto de estratégias" and gerar_misto:
        jogos = []
        jogos_info = []

        # Aleatório puro
        qtd_ap = jogos_misto.get("Aleatório puro", 0)
        if qtd_ap > 0:
            js = gerar_aleatorio_puro(qtd_ap, tam_jogo_mix)
            jogos.extend(js)
            jogos_info.extend(
                {"estrategia": "Aleatório puro", "jogo": j} for j in js
            )

        # Balanceado
        qtd_bal = jogos_misto.get("Balanceado par/ímpar", 0)
        if qtd_bal > 0:
            js = gerar_balanceado_par_impar(qtd_bal, tam_jogo_mix)
            jogos.extend(js)
            jogos_info.extend(
                {"estrategia": "Balanceado par/ímpar", "jogo": j} for j in js
            )

        # Setorial
        qtd_set = jogos_misto.get("Setorial (faixas)", 0)
        if qtd_set > 0:
            js = gerar_setorial(qtd_set, tam_jogo_mix)
            jogos.extend(js)
            jogos_info.extend(
                {"estrategia": "Setorial (faixas)", "jogo": j} for j in js
            )

        # Quentes/Frias/Mix
        qtd_qfm = jogos_misto.get("Quentes/Frias/Mix", 0)
        if qtd_qfm > 0:
            js = gerar_quentes_frias_mix(
                qtd_jogos=qtd_qfm,
                tam_jogo=tam_jogo_mix,
                freq_df=freq_df,
                proporcao=(mix_q_quentes, mix_q_frias, mix_q_neutras),
            )
            jogos.extend(js)
            jogos_info.extend(
                {"estrategia": "Quentes/Frias/Mix", "jogo": j} for j in js
            )

        # Sem sequências
        qtd_ss = jogos_misto.get("Sem sequências longas", 0)
        if qtd_ss > 0:
            js = gerar_sem_sequencias(
                qtd_jogos=qtd_ss,
                tam_jogo=tam_jogo_mix,
                limite_sequencia=mix_limite_seq,
            )
            jogos.extend(js)
            jogos_info.extend(
                {"estrategia": "Sem sequências longas", "jogo": j} for j in js
            )

        # Wheeling
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
                    max_jogos=qtd_wh,
                )
                jogos.extend(js)
                jogos_info.extend(
                    {"estrategia": "Wheeling simples (base fixa)", "jogo": j}
                    for j in js
                )
            except Exception:
                pass

        # aplica filtros
        jogos_filtrados = filtrar_jogos(
            jogos,
            dezenas_fixas=dezenas_fixas,
            dezenas_proibidas=dezenas_proibidas,
            soma_min=soma_min_val,
            soma_max=soma_max_val,
        )

        # re-sincroniza jogos_info com jogos filtrados
        novo_info = []
        for info in jogos_info:
            if info["jogo"] in jogos_filtrados:
                novo_info.append(info)
        jogos = [info["jogo"] for info in novo_info]
        jogos_info = novo_info

    # ---------- APLICAÇÃO DO ORÇAMENTO ----------
    aviso_orcamento = ""
    if jogos and orcamento_max > 0:
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

    # ---------- EXIBIÇÃO ----------
    if not jogos and (gerar or gerar_misto):
        tab_jogos.warning(
            "Nenhum jogo foi gerado após aplicar as regras, filtros e orçamento. "
            "Revise os parâmetros na barra lateral (especialmente filtros avançados e orçamento) e tente novamente."
        )
        tab_tabela.info("Nenhum dado para exibir ainda.")
        tab_analise.info("Nenhuma análise disponível sem jogos gerados.")
    elif not jogos:
        tab_jogos.write(
            "Ajuste os parâmetros na barra lateral e clique em **Gerar** para ver seus jogos aqui."
        )
        tab_tabela.write("A tabela com os jogos aparecerá aqui após a geração.")
        tab_analise.write("A análise dos jogos gerados aparecerá aqui após a geração.")
    else:
        # painel de resumo (custo e probabilidade)
        custo_total = calcular_custo_total(jogos, preco_base_6)
        prob_total = prob_sena_pacote(jogos)

        with tab_jogos:
            st.markdown("#### Lista de jogos")
            st.caption("Cada linha mostra o número do jogo, a estratégia (quando aplicável) e as dezenas.")

            # resumo rápido acima
            colr1, colr2, colr3 = st.columns(3)
            with colr1:
                st.metric("Quantidade de jogos", len(jogos))
            with colr2:
                st.metric(
                    "Custo total estimado",
                    f"R$ {custo_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                    help=(
                        "Valor aproximado calculado a partir do preço base informado. "
                        "Os valores reais podem variar conforme a tabela oficial."
                    ),
                )
            with colr3:
                if prob_total > 0:
                    inv = 1.0 / prob_total
                    st.metric(
                        "Chance aprox. de 1 Sena",
                        f"1 em {inv:,.0f}".replace(",", "."),
                        help=(
                            "Probabilidade aproximada de acertar a sena com pelo menos 1 jogo "
                            "deste pacote, considerando combinações C(n,6)/C(60,6)."
                        ),
                    )
                else:
                    st.metric("Chance aprox. de 1 Sena", "N/A")

            if aviso_orcamento:
                st.info(aviso_orcamento)

            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                for i, info in enumerate(jogos_info, start=1):
                    nome_est = info["estrategia"]
                    jogo = info["jogo"]
                    st.code(f"{i:02d} - {nome_est}: {formatar_jogo(jogo)}")

        with tab_tabela:
            st.markdown("#### Tabela completa, resumo e exportação")
            st.caption("Inclui uma coluna com a estratégia usada em cada jogo.")

            dados = []
            for info in jogos_info:
                row = {"estrategia": info["estrategia"]}
                for idx, d in enumerate(info["jogo"], start=1):
                    row[f"d{idx}"] = d
                row["soma"] = sum(info["jogo"])
                pares, impares = pares_impares(info["jogo"])
                row["pares"] = pares
                row["impares"] = impares
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

        # ---------------- ANALISE & SIMULAÇÃO ----------------
        with tab_analise:
            st.markdown("#### Análise dos jogos gerados")
            st.caption(
                "Veja como os jogos se distribuem em termos de frequência das dezenas, soma e par/ímpar, "
                "e simule quantos acertos teriam em um sorteio específico."
            )

            # frequência das dezenas dentro dos jogos gerados
            todas_dezenas = [d for j in jogos for d in j]
            freq_jogos = pd.Series(todas_dezenas).value_counts().sort_index()
            df_freq_jogos = freq_jogos.reset_index()
            df_freq_jogos.columns = ["dezena", "frequencia"]

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown("**Frequência das dezenas nos jogos gerados**")
                st.bar_chart(
                    df_freq_jogos.set_index("dezena")["frequencia"],
                    use_container_width=True,
                )

            # distribuição de soma
            somas = [sum(j) for j in jogos]
            df_soma_jogos = pd.DataFrame({"soma": somas})
            st.markdown("**Distribuição das somas dos jogos gerados**")
            st.bar_chart(df_soma_jogos["soma"], use_container_width=True)

            # simulação
            st.markdown("---")
            st.markdown("#### Simulação de acertos em um sorteio")
            modo_sim = st.radio(
                "Escolher resultado para simulação",
                ["Usar concurso histórico", "Digitar resultado manual"],
                horizontal=True,
            )

            dezenas_sorteadas: list[int] = []

            if modo_sim == "Usar concurso histórico":
                ultimos_ids = df_concursos.sort_values(
                    "concurso", ascending=False
                )["concurso"].head(50)
                concurso_escolhido = st.selectbox(
                    "Escolha o concurso para simular",
                    options=ultimos_ids,
                    format_func=lambda c: f"Concurso {c}",
                )
                linha = df_concursos.loc[
                    df_concursos["concurso"] == concurso_escolhido
                ].iloc[0]
                dezenas_sorteadas = [int(linha[f"d{i}"]) for i in range(1, 7)]
                st.write(f"Dezenas sorteadas nesse concurso: {formatar_jogo(dezenas_sorteadas)}")
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
                st.markdown("**Resumo de acertos nos jogos gerados**")
                dist_acertos = (
                    df_sim["acertos"].value_counts().sort_index().reset_index()
                )
                dist_acertos.columns = ["acertos", "qtd_jogos"]
                st.dataframe(dist_acertos, hide_index=True, use_container_width=True)

                st.markdown("**Detalhamento por jogo**")
                st.dataframe(df_sim, hide_index=True, use_container_width=True)

else:
    pagina_analises(df_concursos, freq_df)
