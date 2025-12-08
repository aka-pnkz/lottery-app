import streamlit as st
import pandas as pd

from utils.storage import load_results
from analysis.frequency import analyze_frequency, analyze_delay
from analysis.generator import gerar_jogos
from analysis.probability import prob_sena
from pricing.pricing_table import preco_por_jogo, custo_total

st.set_page_config(page_title="Mega-Sena Analyzer", layout="wide")

@st.cache_data
def load_data():
    return load_results()

def pagina_historico(df: pd.DataFrame):
    st.header("Histórico da Mega-Sena")
    st.write(f"Total de concursos carregados: {len(df)}")
    st.dataframe(df.tail(20), use_container_width=True)

def pagina_analises(df: pd.DataFrame):
    st.header("Análises e estatísticas")
    freq_df = analyze_frequency(df)
    atraso_df = analyze_delay(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Frequência das dezenas")
        st.dataframe(freq_df, use_container_width=True)
        st.bar_chart(freq_df.set_index("numero")["frequencia"])
    with col2:
        st.subheader("Atraso das dezenas")
        st.dataframe(atraso_df, use_container_width=True)

def pagina_gerar_jogos():
    st.header("Gerar jogos")
    qtd_jogos = st.number_input("Quantidade de jogos", min_value=1, max_value=1000, value=5)
    dezenas_por_jogo = st.number_input("Dezenas por jogo", min_value=6, max_value=20, value=6)
    estrategia = st.selectbox("Estratégia", ["aleatorio_puro"])

    if st.button("Gerar"):
        df_jogos = gerar_jogos(int(qtd_jogos), int(dezenas_por_jogo), estrategia)
        st.dataframe(df_jogos, use_container_width=True)

        # Custo estimado
        try:
            preco = preco_por_jogo(int(dezenas_por_jogo))
            total = custo_total(int(qtd_jogos), int(dezenas_por_jogo))
            st.info(f"Preço por jogo: R$ {preco:,.2f} | Custo total: R$ {total:,.2f}")
        except Exception as e:
            st.warning(f"Não foi possível calcular o custo: {e}")

        # Download CSV
        csv = df_jogos.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar jogos em CSV",
            data=csv,
            file_name="jogos_mega_sena.csv",
            mime="text/csv",
        )

def pagina_simulacao():
    st.header("Simulação de chances")
    dezenas_por_jogo = st.number_input("Dezenas por jogo", min_value=6, max_value=20, value=6)

    if st.button("Calcular probabilidade de Sena"):
        try:
            p = prob_sena(int(dezenas_por_jogo))
            st.success(
                f"Probabilidade de acertar a Sena com {dezenas_por_jogo} dezenas "
                f"em um jogo: {p:.12f} (aprox. 1 em {1/p:,.0f})."
            )
        except Exception as e:
            st.error(str(e))

def main():
    st.sidebar.title("Mega-Sena App")
    pagina = st.sidebar.radio(
        "Navegação",
        ["Histórico", "Análises", "Gerar jogos", "Simulação"],
    )

    df = load_data()

    if pagina == "Histórico":
        pagina_historico(df)
    elif pagina == "Análises":
        pagina_analises(df)
    elif pagina == "Gerar jogos":
        pagina_gerar_jogos()
    elif pagina == "Simulação":
        pagina_simulacao()

    st.caption(
        "App para estudo e entretenimento. "
        "Use apenas dados oficiais para decisões reais."
    )

if __name__ == "__main__":
    main()
