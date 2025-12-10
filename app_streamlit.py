import streamlit as st
import pandas as pd
import random
from pathlib import Path


# ---------------- CONFIGURAÇÕES BÁSICAS ----------------

HIST_CSV_PATH = Path("historico_mega_sena.csv")  # ajuste se quiser outro caminho


# ---------------- FUNÇÕES DE DADOS ----------------

def load_data() -> pd.DataFrame:
    """
    Carrega histórico da Mega-Sena de um CSV separado por ';'.
    Espera colunas: concurso, data, dezena1..dezena6.
    """
    if not HIST_CSV_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(HIST_CSV_PATH, sep=";")
    return df



# ---------------- FUNÇÕES DE ANÁLISE ----------------

def _cols_dezenas(df: pd.DataFrame) -> list:
    # se suas colunas são exatamente dezena1, dezena2, ..., force explicitamente
    return [c for c in df.columns if c.lower().startswith("dezena")]
    # ou, se souber o nome exato:
    # return ["dezena1", "dezena2", "dezena3", "dezena4", "dezena5", "dezena6"]

def analyze_frequency(df_hist: pd.DataFrame) -> pd.DataFrame:
    if df_hist.empty:
        return pd.DataFrame()

    dezenas_cols = _cols_dezenas(df_hist)
    if not dezenas_cols:
        st.error(f"Nenhuma coluna de dezenas encontrada. Colunas do histórico: {list(df_hist.columns)}")
        return pd.DataFrame()

    # derrete apenas as colunas de dezenas em uma coluna chamada 'dezena'
    df_melt = df_hist.melt(value_vars=dezenas_cols, value_name="dezena")

    if "dezena" not in df_melt.columns:
        st.error(f"Coluna 'dezena' não existe após melt. Colunas: {list(df_melt.columns)}")
        return pd.DataFrame()

    df_melt = df_melt.dropna(subset=["dezena"])
    df_melt["dezena"] = df_melt["dezena"].astype(str).str.strip()
    df_melt = df_melt[df_melt["dezena"].str.fullmatch(r"\d+")]
    df_melt["dezena"] = df_melt["dezena"].astype(int)

    freq = (
        df_melt["dezena"]
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"index": "dezena", "dezena": "frequencia"})
    )

    return freq.sort_values("dezena")




def analyze_delay(df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula atraso (em concursos) de cada dezena.
    Considera o índice do DataFrame como ordem dos concursos.
    """
    if df_hist.empty:
        return pd.DataFrame()

    df = df_hist.reset_index(drop=True)
    dezenas_cols = _cols_dezenas(df)

    ultima_linha = len(df) - 1
    registros = []

    for dezena in range(1, 61):
        apareceu_em = df[df[dezenas_cols].eq(dezena).any(axis=1)].index
        if len(apareceu_em) == 0:
            atraso = None
        else:
            atraso = ultima_linha - apareceu_em.max()
        registros.append({"dezena": dezena, "atraso": atraso})

    out = pd.DataFrame(registros).sort_values("dezena")
    return out


def analyze_par_impar(df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula distribuição par/ímpar por concurso e frequência de cada composição.
    """
    if df_hist.empty:
        return pd.DataFrame()

    dezenas_cols = _cols_dezenas(df_hist)
    registros = []
    for _, row in df_hist.iterrows():
        dezenas = [int(row[c]) for c in dezenas_cols]
        pares = sum(1 for d in dezenas if d % 2 == 0)
        impares = len(dezenas) - pares
        registros.append({"pares": pares, "impares": impares})

    tmp = pd.DataFrame(registros)
    out = (
        tmp.value_counts(["pares", "impares"])
        .reset_index(name="qtd")
        .sort_values("qtd", ascending=False)
    )
    return out


# ---------------- FUNÇÕES DE PREÇO ----------------

def preco_por_jogo(dezenas_por_jogo: int) -> float:
    """
    Retorna preço de um jogo da Mega-Sena conforme quantidade de dezenas.
    Tabela aproximada baseada em fontes públicas (pode sofrer reajustes). [web:443]
    """
    tabela = {
        6: 5.00,
        7: 35.00,
        8: 140.00,
        9: 378.00,
        10: 945.00,
        11: 2079.00,
        12: 4158.00,
        13: 7722.00,
        14: 13513.50,
        15: 22522.50,
    }  # [web:443]

    if dezenas_por_jogo not in tabela:
        return 0.0
    return tabela[dezenas_por_jogo]


def custo_total(qtd_jogos: int, dezenas_por_jogo: int) -> float:
    return qtd_jogos * preco_por_jogo(dezenas_por_jogo)


# ---------------- GERADOR DE JOGOS ----------------

def gerar_jogo_simples(dezenas_por_jogo: int) -> list[int]:
    return sorted(random.sample(range(1, 61), dezenas_por_jogo))


def gerar_jogos(
    qtd_jogos: int,
    dezenas_por_jogo: int,
    estrategia: str,
    freq_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Gera jogos conforme a estratégia.
    Implementações simples/ilustrativas.
    """
    jogos = []

    if freq_df is not None and not freq_df.empty and "dezena" in freq_df.columns:
        freq_sorted = freq_df.sort_values("frequencia", ascending=False)
        quentes = freq_sorted["dezena"].tolist()[:20]
        frias = freq_sorted["dezena"].tolist()[-20:]
    else:
        quentes = []
        frias = []

    for i in range(1, qtd_jogos + 1):
        if estrategia == "aleatorio_puro":
            dezenas = gerar_jogo_simples(dezenas_por_jogo)

        elif estrategia == "balanceado_par_impar":
            alvo_par = dezenas_por_jogo // 2
            alvo_impar = dezenas_por_jogo - alvo_par
            pares = random.sample([d for d in range(1, 61) if d % 2 == 0], alvo_par)
            impares = random.sample([d for d in range(1, 61) if d % 2 != 0], alvo_impar)
            dezenas = sorted(pares + impares)

        elif estrategia == "faixas":
            faixas = [range(1, 21), range(21, 41), range(41, 61)]
            dezenas = []
            while len(dezenas) < dezenas_por_jogo:
                f = random.choice(faixas)
                n = random.choice(list(f))
                if n not in dezenas:
                    dezenas.append(n)
            dezenas = sorted(dezenas)

        elif estrategia == "sem_sequencias":
            while True:
                dezenas = gerar_jogo_simples(dezenas_por_jogo)
                if not any(
                    dezenas[j] == dezenas[j - 1] + 1
                    for j in range(1, len(dezenas))
                ):
                    break

        elif estrategia == "hot" and quentes:
            dezenas = sorted(random.sample(quentes, min(dezenas_por_jogo, len(quentes))))

        elif estrategia == "cold" and frias:
            dezenas = sorted(random.sample(frias, min(dezenas_por_jogo, len(frias))))

        elif estrategia == "hot_cold_misto" and quentes and frias:
            qtd_hot = max(1, dezenas_por_jogo // 3)
            qtd_cold = max(1, dezenas_por_jogo // 3)
            qtd_rest = dezenas_por_jogo - qtd_hot - qtd_cold

            escolha_hot = random.sample(quentes, min(qtd_hot, len(quentes)))
            escolha_cold = random.sample(frias, min(qtd_cold, len(frias)))
            restantes = [d for d in range(1, 61) if d not in escolha_hot + escolha_cold]
            escolha_rest = random.sample(restantes, max(0, qtd_rest))
            dezenas = sorted(escolha_hot + escolha_cold + escolha_rest)

        else:
            dezenas = gerar_jogo_simples(dezenas_por_jogo)

        jogos.append({"jogo": i, "dezenas": dezenas})

    linhas = []
    for row in jogos:
        base = {"jogo": row["jogo"]}
        for idx, d in enumerate(row["dezenas"], start=1):
            base[f"dezena{idx}"] = d
        linhas.append(base)

    return pd.DataFrame(linhas)


# ---------------- PÁGINA: GERAR JOGOS ----------------

def pagina_gerar_jogos():
    st.header("Gerar jogos")

    try:
        df_hist = load_data()
        freq_df = analyze_frequency(df_hist) if df_hist is not None and not df_hist.empty else None
    except Exception:
        freq_df = None

    estrategias_disponiveis = [
        "aleatorio_puro",
        "balanceado_par_impar",
        "faixas",
        "sem_sequencias",
        "hot",
        "cold",
        "hot_cold_misto",
    ]

    with st.form("form_gerar_jogos"):
        qtd_jogos = st.number_input("Quantidade de jogos", min_value=1, max_value=1000, value=5, step=1)
        dezenas_por_jogo = st.number_input("Dezenas por jogo", min_value=6, max_value=15, value=6, step=1)
        estrategia = st.selectbox("Estratégia", estrategias_disponiveis, index=0)
        gerar_todas = st.checkbox("Gerar 1 jogo para cada estratégia acima", value=False)

        with st.expander("Entenda as estratégias"):
            st.markdown(
                """
**Importante:** todas as combinações têm a mesma probabilidade matemática.
As estratégias abaixo só organizam os números de formas diferentes e não garantem prêmio. [web:343][web:339]

- **aleatorio_puro** – dezenas totalmente aleatórias entre 1 e 60. [web:343]  
- **balanceado_par_impar** – mantém equilíbrio entre pares e ímpares. [web:317][web:320]  
- **faixas** – distribui as dezenas em 1–20, 21–40 e 41–60. [web:319]  
- **sem_sequencias** – evita sequências longas de dezenas consecutivas. [web:486]  
- **hot** – prioriza dezenas mais frequentes no histórico. [web:322]  
- **cold** – prioriza dezenas menos frequentes/atrasadas. [web:322][web:323]  
- **hot_cold_misto** – mistura dezenas quentes, frias e neutras. [web:322]
                """
            )

        submitted = st.form_submit_button("Gerar jogos")

    if not submitted:
        return

    if qtd_jogos <= 0:
        st.error("A quantidade de jogos deve ser maior que zero.")
        return

    try:
        if gerar_todas:
            linhas = []
            for nome_estrategia in estrategias_disponiveis:
                df_tmp = gerar_jogos(
                    qtd_jogos=1,
                    dezenas_por_jogo=int(dezenas_por_jogo),
                    estrategia=nome_estrategia,
                    freq_df=freq_df,
                )
                df_tmp["estrategia"] = nome_estrategia
                linhas.append(df_tmp)
            df_jogos = pd.concat(linhas, ignore_index=True)
        else:
            df_jogos = gerar_jogos(
                int(qtd_jogos),
                int(dezenas_por_jogo),
                estrategia,
                freq_df=freq_df,
            )
    except Exception as e:
        st.error(f"Erro ao gerar jogos: {e}")
        return

    if "jogo" in df_jogos.columns:
        df_jogos["jogo"] = df_jogos["jogo"].apply(lambda x: f"#{int(x)}")

    st.subheader("Jogos gerados")
    st.dataframe(df_jogos, width="stretch", hide_index=True)

    try:
        preco = preco_por_jogo(int(dezenas_por_jogo))
        qtd_para_preco = len(df_jogos) if gerar_todas else int(qtd_jogos)
        total = custo_total(qtd_para_preco, int(dezenas_por_jogo))
        st.info(f"Preço por jogo: R$ {preco:,.2f} | Custo total: R$ {total:,.2f}")
    except Exception as e:
        st.warning(f"Não foi possível calcular o custo: {e}")

    csv = df_jogos.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Baixar jogos em CSV",
        data=csv,
        file_name="jogos_mega_sena.csv",
        mime="text/csv",
    )


# ---------------- PÁGINA: ANÁLISES ----------------

def pagina_analises():
    st.header("Análises estatísticas")

    try:
        df_hist = load_data()
    except Exception as e:
        st.error(f"Erro ao carregar histórico: {e}")
        return

    if df_hist is None or df_hist.empty:
        st.warning("Nenhum histórico carregado para análise (arquivo historico_mega_sena.csv não encontrado ou vazio).")
        return

    with st.expander("Entenda as análises"):
        st.markdown(
            """
Todas as análises são estatísticas descritivas do passado e não aumentam matematicamente a chance de ganhar. [web:343]

- **Frequência das dezenas** – quantas vezes cada número já foi sorteado. [web:322]  
- **Atraso (delay)** – há quantos concursos cada dezena não aparece. [web:323]  
- **Distribuição par/ímpar** – quais composições de pares/ímpares ocorrem mais. [web:322]  
- **Faixas** – como as dezenas se distribuem entre 1–20, 21–40, 41–60. [web:322]
            """
        )

    aba_freq, aba_atraso, aba_par_impar, aba_faixas = st.tabs(
        ["Frequência", "Atraso", "Par/Ímpar", "Faixas"]
    )

    with aba_freq:
        st.subheader("Frequência das dezenas")
        st.info("Mostra quantas vezes cada dezena já foi sorteada no histórico. [web:322]")
        try:
            freq_df = analyze_frequency(df_hist)
            st.dataframe(freq_df, width="stretch")
        except Exception as e:
            st.error(f"Erro ao calcular frequência: {e}")

    with aba_atraso:
        st.subheader("Atraso das dezenas")
        st.info("Mostra há quantos concursos cada dezena não aparece. [web:323]")
        try:
            atraso_df = analyze_delay(df_hist)
            st.dataframe(atraso_df, width="stretch")
        except Exception as e:
            st.error(f"Erro ao calcular atraso: {e}")

    with aba_par_impar:
        st.subheader("Distribuição par/ímpar")
        st.info("Mostra as composições de pares e ímpares que mais ocorrem. [web:322]")
        try:
            par_impar_df = analyze_par_impar(df_hist)
            st.dataframe(par_impar_df, width="stretch")
        except Exception as e:
            st.error(f"Erro ao calcular distribuição par/ímpar: {e}")

    with aba_faixas:
        st.subheader("Distribuição por faixas")
        st.info("Mostra como as dezenas se distribuem entre 1–20, 21–40 e 41–60. [web:322]")
        try:
            df_tmp = df_hist.copy()
            dezenas_cols = _cols_dezenas(df_tmp)
            df_melt = df_tmp.melt(value_vars=dezenas_cols, value_name="dezena")
            df_melt = df_melt.dropna(subset=["dezena"])
            df_melt["dezena"] = df_melt["dezena"].astype(int)
            df_melt["faixa"] = pd.cut(
                df_melt["dezena"],
                bins=[0, 20, 40, 60],
                labels=["1-20", "21-40", "41-60"],
            )
            faixas_df = df_melt.groupby("faixa")["dezena"].count().reset_index(name="qtd")
            st.dataframe(faixas_df, width="stretch")
        except Exception as e:
            st.error(f"Erro ao calcular distribuição por faixas: {e}")


# ---------------- PÁGINA: SOBRE ----------------

def pagina_sobre():
    st.header("Sobre o projeto")
    st.markdown(
        """
Aplicativo para estudo estatístico da Mega-Sena e geração de jogos com diferentes estratégias.  
As análises usam dados históricos apenas para visualização e aprendizado. [web:322]
        """
    )


# ---------------- NAVEGAÇÃO PRINCIPAL ----------------

def main():
    st.set_page_config(page_title="Mega-Sena Helper", layout="wide")

    st.sidebar.title("Navegação")
    pagina = st.sidebar.selectbox(
        "Escolha a página",
        ["Gerar jogos", "Análises", "Sobre"],
    )

    if pagina == "Gerar jogos":
        pagina_gerar_jogos()
    elif pagina == "Análises":
        pagina_analises()
    elif pagina == "Sobre":
        pagina_sobre()


if __name__ == "__main__":
    main()
