import streamlit as st
import pandas as pd

from utils.storage import load_results
from analysis.frequency import (
    analyze_frequency,
    analyze_delay,
    analyze_par_impar,
)
from analysis.generator import gerar_jogos
from analysis.probability import prob_sena
from pricing.pricing_table import preco_por_jogo, custo_total

st.set_page_config(
    page_title="Mega-Sena Analyzer",
    layout="wide",
    page_icon="🎲",
)

# --------------------------------------------------------------------------------------
# Carregamento de dados (com cache)
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return load_results()

# --------------------------------------------------------------------------------------
# Funções auxiliares
# --------------------------------------------------------------------------------------
def tem_colunas_basicas(df: pd.DataFrame) -> bool:
    # aceita "Dezena1"..."Dezena6"
    col_dezenas = [c for c in df.columns if c.lower().startswith("dezena")]
    return len(col_dezenas) >= 6


# --------------------------------------------------------------------------------------
# Páginas
# --------------------------------------------------------------------------------------
def pagina_historico(df: pd.DataFrame):
    st.header("Histórico da Mega-Sena")

    if df.empty:
        st.warning(
            "O arquivo mega_sena.csv está vazio ou sem dados válidos.\n\n"
            "Suba um CSV com o histórico completo em data/mega_sena.csv "
            "e faça o deploy novamente."
        )
        return

    col_concurso = "concurso"
    col_data = "data"

    if col_concurso in df.columns:
        total_concursos = df[col_concurso].nunique()
        st.write(f"Total de concursos carregados: **{total_concursos}**")
    else:
        st.write(f"Total de linhas no arquivo: **{len(df)}**")

    if col_data in df.columns:
        try:
            df[col_data] = pd.to_datetime(df[col_data], errors="coerce")
            ultima_data = df[col_data].max()
            if pd.notna(ultima_data):
                st.write(f"Último concurso em: **{ultima_data.date()}**")
        except Exception:
            pass

    st.subheader("Últimos concursos")
    st.dataframe(df.tail(20), use_container_width=True)


def pagina_analises(df: pd.DataFrame):
    st.header("Análises e estatísticas")

    if df.empty or not tem_colunas_basicas(df):
        st.warning("Histórico indisponível ou incompleto para análises.")
        return

    freq_df = analyze_frequency(df)
    atraso_df = analyze_delay(df)
    pares_df = analyze_par_impar(df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Frequência das dezenas")
        st.dataframe(freq_df, use_container_width=True)
        chart_data = freq_df.set_index("numero")["frequencia"]
        st.bar_chart(chart_data)

    with col2:
        st.subheader("Atraso das dezenas")
        st.dataframe(atraso_df, use_container_width=True)

    st.subheader("Distribuição de pares x ímpares")
    if pares_df is not None and not pares_df.empty:
        st.dataframe(pares_df, use_container_width=True)
    else:
        st.info("Não foi possível calcular pares x ímpares para o histórico atual.")


def pagina_gerar_jogos():
    st.header("Gerar jogos")

    with st.form("form_gerar_jogos"):
        qtd_jogos = st.number_input(
            "Quantidade de jogos",
            min_value=1,
            max_value=1000,
            value=5,
            step=1,
        )
        dezenas_por_jogo = st.number_input(
            "Dezenas por jogo",
            min_value=6,
            max_value=20,
            value=6,
            step=1,
        )
        estrategia = st.selectbox(
            "Estratégia",
            ["aleatorio_puro"],
            index=0,
        )
        submitted = st.form_submit_button("Gerar jogos")

    if not submitted:
        return

    if qtd_jogos <= 0:
        st.error("A quantidade de jogos deve ser maior que zero.")
        return

    try:
        df_jogos = gerar_jogos(
            int(qtd_jogos),
            int(dezenas_por_jogo),
            estrategia,
        )
    except Exception as e:
        st.error(f"Erro ao gerar jogos: {e}")
        return

    st.subheader("Jogos gerados")
    st.dataframe(df_jogos, use_container_width=True)

    try:
        preco = preco_por_jogo(int(dezenas_por_jogo))
        total = custo_total(int(qtd_jogos), int(dezenas_por_jogo))
        st.info(
            f"Preço por jogo: **R$ {preco:,.2f}**  |  "
            f"Custo total: **R$ {total:,.2f}**"
        )
    except Exception as e:
        st.warning(f"Não foi possível calcular o custo: {e}")

    csv = df_jogos.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Baixar jogos em CSV",
        data=csv,
        file_name="jogos_mega_sena.csv",
        mime="text/csv",
    )


def pagina_simulacao():
    st.header("Simulação de probabilidades")

    with st.form("form_simulacao"):
        dezenas_por_jogo = st.number_input(
            "Dezenas por jogo",
            min_value=6,
            max_value=20,
            value=6,
            step=1,
        )
        submitted = st.form_submit_button("Calcular probabilidade de Sena")

    if not submitted:
        return

    try:
        p = prob_sena(int(dezenas_por_jogo))
        if p > 0:
            st.success(
                f"Probabilidade de acertar a **Sena** com "
                f"{dezenas_por_jogo} dezenas em um único jogo:\n\n"
                f"- Valor aproximado: **{p:.12f}**\n"
                f"- Aproximadamente **1 em {1/p:,.0f}** combinações."
            )
        else:
            st.warning("Probabilidade retornou 0. Verifique a função prob_sena.")
    except Exception as e:
        st.error(f"Erro ao calcular probabilidade: {e}")


# --------------------------------------------------------------------------------------
# Função principal
# --------------------------------------------------------------------------------------
def main():
    st.sidebar.title("Mega-Sena App")

    pagina = st.sidebar.radio(
        "Navegação",
        ["Histórico", "Análises", "Gerar jogos", "Simulação"],
        index=0,
    )

    try:
        df = load_data()
    except FileNotFoundError as e:
        st.error(
            "Arquivo de histórico não encontrado.\n\n"
            "Confira se `data/mega_sena.csv` está presente no repositório.\n\n"
            f"Detalhes: {e}"
        )
        df = pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        df = pd.DataFrame()

    if pagina == "Histórico":
        pagina_historico(df)
    elif pagina == "Análises":
        pagina_analises(df)
    elif pagina == "Gerar jogos":
        pagina_gerar_jogos()
    elif pagina == "Simulação":
        pagina_simulacao()

    st.markdown("---")
    st.caption(
        "App de estudo e entretenimento sobre Mega-Sena. "
        "Probabilidades e custos são aproximações; "
        "consulte sempre as regras e valores oficiais."
    )


if __name__ == "__main__":
    main()
