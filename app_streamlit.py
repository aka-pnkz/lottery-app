import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================
# CONFIGURAÇÃO BÁSICA
# ==========================
st.set_page_config(
    page_title="Mega Sena Helper",
    page_icon="🎰",
    layout="wide"
)

# ==========================
# FUNÇÕES AUXILIARES
# ==========================
@st.cache_data
def carregar_concursos(caminho_csv: str = "megasena.csv") -> pd.DataFrame:
    """
    Espera um CSV com colunas:
    concurso, data, d1, d2, d3, d4, d5, d6
    """
    df = pd.read_csv(caminho_csv, sep=";")
    # Garante tipos
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


def gerar_jogos(
    metodo: str,
    qtd_jogos: int,
    tam_jogo: int,
    freq_df: pd.DataFrame | None = None,
) -> list[list[int]]:
    jogos = []
    universe = np.arange(1, 61)

    for _ in range(qtd_jogos):
        if metodo == "Aleatório puro":
            dezenas = np.random.choice(universe, size=tam_jogo, replace=False)

        elif metodo == "Mais frequentes" and freq_df is not None:
            # Ordena por frequência decrescente
            ordenado = freq_df.sort_values("frequencia", ascending=False)
            base = ordenado["dezena"].values[:max(tam_jogo * 3, tam_jogo)]
            dezenas = np.random.choice(base, size=tam_jogo, replace=False)

        elif metodo == "Menos frequentes" and freq_df is not None:
            ordenado = freq_df.sort_values("frequencia", ascending=True)
            base = ordenado["dezena"].values[:max(tam_jogo * 3, tam_jogo)]
            dezenas = np.random.choice(base, size=tam_jogo, replace=False)

        elif metodo == "Misto (mais+menos)" and freq_df is not None:
            ordenado_mais = freq_df.sort_values("frequencia", ascending=False)
            ordenado_menos = freq_df.sort_values("frequencia", ascending=True)
            top = ordenado_mais["dezena"].values[:30]
            bottom = ordenado_menos["dezena"].values[:30]
            base = np.unique(np.concatenate([top, bottom]))
            dezenas = np.random.choice(base, size=tam_jogo, replace=False)

        else:
            # fallback para aleatório
            dezenas = np.random.choice(universe, size=tam_jogo, replace=False)

        jogos.append(sorted(dezenas.tolist()))

    return jogos


def formatar_jogo(jogo: list[int]) -> str:
    return " - ".join(f"{d:02d}" for d in jogo)


# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.title("Mega Sena Helper 🎰")

    st.markdown("### Configurações básicas")
    caminho_csv = st.text_input(
        "Caminho/arquivo CSV dos concursos",
        value="megasena.csv",
        help="Use o mesmo arquivo que você já baixou da Caixa ou do seu script."
    )

    metodo = st.selectbox(
        "Método de geração",
        ["Aleatório puro", "Mais frequentes", "Menos frequentes", "Misto (mais+menos)"],
    )

    qtd_jogos = st.number_input(
        "Quantidade de jogos",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
    )

    tam_jogo = st.slider(
        "Quantidade de dezenas por jogo",
        min_value=6,
        max_value=15,
        value=6,
    )

    mostrar_frequencias = st.checkbox(
        "Mostrar tabela de frequências",
        value=True
    )

# ==========================
# CORPO PRINCIPAL
# ==========================
st.title("Gerador de Jogos para Mega-Sena")

st.markdown(
    "App experimental para estudo estatístico e geração de combinações de jogos "
    "a partir do histórico de concursos da Mega-Sena."
)

# Carrega concursos
try:
    df_concursos = carregar_concursos(caminho_csv)
    freq_df = calcular_frequencias(df_concursos)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Histórico carregado")
        st.write(
            f"Total de concursos: **{len(df_concursos)}** "
            f"(de {df_concursos['data'].min().date()} até {df_concursos['data'].max().date()})"
        )

        if mostrar_frequencias:
            st.subheader("Frequência das dezenas")
            st.dataframe(
                freq_df.style.background_gradient(
                    subset=["frequencia"], cmap="Blues"
                ),
                use_container_width=True,
                height=400,
            )

    with col2:
        st.subheader("Parâmetros selecionados")
        st.write(f"- Método: **{metodo}**")
        st.write(f"- Jogos: **{qtd_jogos}**")
        st.write(f"- Dezenas por jogo: **{tam_jogo}**")

        if st.button("Gerar jogos agora", type="primary"):
            jogos = gerar_jogos(
                metodo=metodo,
                qtd_jogos=qtd_jogos,
                tam_jogo=tam_jogo,
                freq_df=freq_df,
            )

            st.markdown("### Jogos sugeridos")
            for i, jogo in enumerate(jogos, start=1):
                st.code(f"Jogo {i:02d}: {formatar_jogo(jogo)}")

            # Opcional: baixar em CSV
            jogos_df = pd.DataFrame(
                jogos,
                columns=[f"d{i}" for i in range(1, tam_jogo + 1)]
            )
            csv_data = jogos_df.to_csv(index=False, sep=";").encode("utf-8")
            st.download_button(
                label="Baixar jogos em CSV",
                data=csv_data,
                file_name=f"jogos_megasena_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

except FileNotFoundError:
    st.error(
        "Arquivo de concursos não encontrado. "
        "Verifique o caminho/arquivo CSV informado na barra lateral."
    )
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar/usar os dados: {e}")
