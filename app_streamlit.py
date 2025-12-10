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

    col_concurso = "Concurso"
    col_data = "Data Sorteio"

    # Contagem básica
    if col_concurso in df.columns:
        total_concursos = df[col_concurso].nunique()
        st.write(f"Total de concursos carregados: **{total_concursos}**")
    else:
        st.write(f"Total de linhas no arquivo: **{len(df)}**")

    # Conversão de data e ordenação
    if col_data in df.columns:
        df[col_data] = pd.to_datetime(df[col_data], errors="coerce", dayfirst=True)

    # Controles do usuário
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        qtd_mostrar = st.selectbox(
            "Quantidade de resultados a exibir",
            options=[10, 20, 50, 100, 200, 500],
            index=1,  # default 20
        )
    with col_ctrl2:
        ordem_desc = st.checkbox(
            "Mostrar do mais recente para o mais antigo",
            value=True,
        )

    df_view = df.copy()

    # Ordenação por data (se existir), senão por concurso
    if col_data in df_view.columns and pd.api.types.is_datetime64_any_dtype(
        df_view[col_data]
    ):
        df_view = df_view.sort_values(col_data, ascending=not ordem_desc)
    elif col_concurso in df_view.columns:
        df_view = df_view.sort_values(col_concurso, ascending=not ordem_desc)

    # Último concurso (depois da ordenação)
    if col_data in df_view.columns and pd.notna(df_view[col_data]).any():
        ultima_data = df_view[col_data].max()
        st.write(f"Último concurso em: **{ultima_data.date()}**")

    st.subheader("Resultados")
    st.dataframe(df_view.head(qtd_mostrar), width="stretch")



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
        st.dataframe(freq_df, width="stretch", hide_index=True)
        chart_data = freq_df.set_index("numero")["frequencia"]
        st.bar_chart(chart_data)

    with col2:
        st.subheader("Atraso das dezenas")
        st.dataframe(atraso_df, width="stretch", hide_index=True)

    st.subheader("Distribuição de pares x ímpares")
    if pares_df is not None and not pares_df.empty:
        st.dataframe(pares_df, width="stretch", hide_index=True)
    else:
        st.info("Não foi possível calcular pares x ímpares para o histórico atual.")




def pagina_gerar_jogos():
    st.header("Gerar jogos")

    # carrega histórico para estratégias hot/cold
    try:
        df_hist = load_data()
        freq_df = analyze_frequency(df_hist) if not df_hist.empty else None
    except Exception:
        freq_df = None

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
            [
                "aleatorio_puro",
                "balanceado_par_impar",
                "faixas",
                "sem_sequencias",
                "hot",
                "cold",
                "hot_cold_misto",
            ],
            index=0,
        )

        with st.expander("Entenda as estratégias"):
            st.markdown(
                """
**Importante:** todas as combinações têm a mesma probabilidade matemática.
As estratégias abaixo só organizam os números de formas diferentes, para deixar o jogo mais **estruturado** e evitar padrões ruins, mas **não garantem prêmio**. [web:267][web:271]

- **aleatorio_puro**  
  Gera dezenas totalmente aleatórias entre 1 e 60, sem nenhuma regra extra.
  É o jeito mais simples e alinhado com a ideia de que cada combinação tem a mesma chance. [web:295][web:293]

- **balanceado_par_impar**  
  Monta jogos tentando manter um equilíbrio entre pares e ímpares (por exemplo 3 pares e 3 ímpares quando são 6 dezenas), porque historicamente distribuições muito extremas (tudo par ou tudo ímpar) são raras. [web:263][web:305]

- **faixas**  
  Espalha as dezenas pelas faixas 1–20, 21–40 e 41–60, para evitar concentrar tudo em uma parte do volante e cobrir melhor o intervalo completo de números. [web:266][web:260]

- **sem_sequencias**  
  Evita jogos com sequências longas de dezenas consecutivas (como 10–11–12–13), que quase não aparecem nos sorteios e são padrão que muitos jogadores escolhem sem perceber. [web:266][web:275]

- **hot**  
  Dá mais peso às dezenas que mais apareceram no histórico (“números quentes”).  
  É uma forma popular de apostar usando frequência passada, embora isso não mude a probabilidade futura em um sorteio realmente aleatório. [web:263][web:259][web:276]

- **cold**  
  Prioriza dezenas que saíram pouco ou estão há muito tempo sem aparecer (“números frios”), na ideia de que podem estar “atrasadas”.  
  É uma escolha de preferência do jogador, não uma vantagem garantida. [web:259][web:288]

- **hot_cold_misto**  
  Mistura algumas dezenas quentes, algumas frias e algumas neutras, para ter um jogo variado que use informações do histórico sem ficar preso só em um grupo de números. [web:300][web:288]
                """
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
            freq_df=freq_df,   # novo parâmetro
        )
    except Exception as e:
        st.error(f"Erro ao gerar jogos: {e}")
        return

    # formata coluna do jogo
    df_jogos["jogo"] = df_jogos["jogo"].apply(lambda x: f"#{int(x)}")

    st.subheader("Jogos gerados")
    st.dataframe(df_jogos, width="stretch", hide_index=True)

    try:
        preco = preco_por_jogo(int(dezenas_por_jogo))
        total = custo_total(int(qtd_jogos), int(dezenas_por_jogo))
        msg_preco = (
            f"Preço por jogo: **R$ {preco:,.2f}**  |  "
            f"Custo total: **R$ {total:,.2f}**"
        )
        st.info(msg_preco)
    except Exception as e:
        msg_erro = f"Não foi possível calcular o custo: {e}"
        st.warning(msg_erro)

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
            msg = (
                f"Probabilidade de acertar a **Sena** com "
                f"{dezenas_por_jogo} dezenas em um único jogo:\n\n"
                f"- Valor aproximado: **{p:.12f}**\n"
                f"- Aproximadamente **1 em {1/p:,.0f}** combinações."
            )
            st.success(msg)
        else:
            st.warning("Probabilidade retornou 0. Verifique a função prob_sena.")
    except Exception as e:
        st.error(f"Erro ao calcular probabilidade: {e}")


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
