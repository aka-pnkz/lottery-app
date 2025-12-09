
import streamlit as st
import pandas as pd

from utils.storage import load_results
from analysis.frequency import analyze_frequency, analyze_delay
from analysis.generator import gerar_jogos
from analysis.probability import prob_sena
from pricing.pricing_table import preco_por_jogo, custo_total


@st.cache_data
def load_data():
    return load_results()

def main():
    st.sidebar.title("Mega-Sena App")
    pagina = st.sidebar.radio("Navegação",
                              ["Histórico", "Análises", "Gerar jogos", "Simulação"])

    df = load_data()

    if df.empty:
        st.warning("O arquivo mega_sena.csv está vazio ou sem dados válidos. "
                   "Suba um CSV com o histórico em data/mega_sena.csv.")
        # Mesmo assim libera as páginas que não dependem do histórico
        if pagina == "Gerar jogos":
            pagina_gerar_jogos()
        elif pagina == "Simulação":
            pagina_simulacao()
        return
