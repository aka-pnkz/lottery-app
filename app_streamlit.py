import streamlit as st
import pandas as pd

# IMPORTS DO SEU PROJETO
from analysis.frequency import analyze_frequency, analyze_delay, analyze_par_impar
from utils.data import load_data              # use o módulo onde HOJE está o load_data
from utils.jogos import gerar_jogos          # módulo onde HOJE está gerar_jogos
from utils.custos import preco_por_jogo, custo_total  # onde HOJE estão essas funções




# ---------------- PAGINA: GERAR JOGOS ----------------

def pagina_gerar_jogos():
    st.header("Gerar jogos")

    # carrega histórico para estratégias hot/cold
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
            estrategias_disponiveis,
            index=0,
        )
        gerar_todas = st.checkbox(
            "Gerar 1 jogo para cada estratégia acima",
            value=False,
        )

        with st.expander("Entenda as estratégias"):
            st.markdown(
                """
**Importante:** todas as combinações têm a mesma probabilidade matemática.
As estratégias abaixo só organizam os números de formas diferentes, para evitar padrões ruins e deixar o jogo mais estruturado, mas **não garantem prêmio**. [web:343][web:339]

- **aleatorio_puro**  
  Gera dezenas totalmente aleatórias entre 1 e 60, sem nenhuma regra extra.
  É o jeito mais simples e alinhado com a ideia de que cada combinação tem a mesma chance. [web:343]

- **balanceado_par_impar**  
  Monta jogos tentando manter um equilíbrio entre pares e ímpares (por exemplo 3 pares e 3 ímpares quando são 6 dezenas), porque distribuições muito extremas (tudo par ou tudo ímpar) são raras nos sorteios. [web:317][web:320]

- **faixas**  
  Espalha as dezenas pelas faixas 1–20, 21–40 e 41–60, para evitar concentrar tudo em uma parte do volante e cobrir melhor o intervalo completo de números. [web:319]

- **sem_sequencias**  
  Evita jogos com sequências longas de dezenas consecutivas (como 10–11–12–13), que quase não aparecem nos sorteios e são padrão que muitos jogadores escolhem sem perceber. [web:319]

- **hot**  
  Dá mais peso às dezenas que mais apareceram no histórico (“números quentes”).
  É uma forma popular de apostar usando frequência passada, embora isso não mude a probabilidade futura em um sorteio realmente aleatório. [web:324][web:335]

- **cold**  
  Prioriza dezenas que saíram pouco ou estão há muito tempo sem aparecer (“números frios”), na ideia de que podem estar “atrasadas”.
  É uma escolha de preferência do jogador, não uma vantagem garantida. [web:323][web:326]

- **hot_cold_misto**  
  Mistura algumas dezenas quentes, algumas frias e algumas neutras, para ter um jogo variado que use informações do histórico sem ficar preso só em um grupo de números. [web:324]
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
            # ignora qtd_jogos e gera 1 por estratégia
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

    # formata coluna do jogo
    if "jogo" in df_jogos.columns:
        df_jogos["jogo"] = df_jogos["jogo"].apply(lambda x: f"#{int(x)}")

    st.subheader("Jogos gerados")
    st.dataframe(df_jogos, use_container_width=True, hide_index=True)

    # custo
    try:
        preco = preco_por_jogo(int(dezenas_por_jogo))
        qtd_para_preco = len(df_jogos) if gerar_todas else int(qtd_jogos)
        total = custo_total(qtd_para_preco, int(dezenas_por_jogo))
        msg_preco = (
            f"Preço por jogo: **R$ {preco:,.2f}**  |  "
            f"Custo total: **R$ {total:,.2f}**"
        )
        st.info(msg_preco)
    except Exception as e:
        msg_erro = f"Não foi possível calcular o custo: {e}"
        st.warning(msg_erro)

    # download
    csv = df_jogos.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Baixar jogos em CSV",
        data=csv,
        file_name="jogos_mega_sena.csv",
        mime="text/csv",
    )


# ---------------- PAGINA: ANALISES ----------------

def pagina_analises():
    st.header("Análises estatísticas")

    # carrega histórico
    try:
        df_hist = load_data()
    except Exception as e:
        st.error(f"Erro ao carregar histórico: {e}")
        return

    if df_hist is None or df_hist.empty:
        st.warning("Nenhum histórico carregado para análise.")
        return

    with st.expander("Entenda as análises"):
        st.markdown(
            """
**Aviso importante:** todas as análises desta página são *estatísticas descritivas* sobre os sorteios passados.  
Elas ajudam a visualizar padrões como frequência, atraso e equilíbrio entre pares/ímpares, mas **não aumentam matematicamente a chance de ganhar** em sorteios futuros, que continuam independentes e aleatórios. [web:343][web:339]

- **Frequência das dezenas**  
  Conta quantas vezes cada número já foi sorteado no histórico, permitindo identificar números mais e menos frequentes. [web:324][web:335]

- **Atraso (delay) das dezenas**  
  Mede há quantos concursos cada dezena não aparece e destaca dezenas consideradas “atrasadas”. [web:323][web:326]

- **Distribuição par/ímpar**  
  Mostra quais composições de pares e ímpares (3/3, 4/2, 5/1 etc.) aparecem com mais frequência. [web:317][web:320]

- **Distribuição por faixas**  
  Agrupa dezenas em intervalos (1–20, 21–40, 41–60) para ver como os resultados se espalham ao longo do volante. [web:319][web:328]
            """
        )

    aba_freq, aba_atraso, aba_par_impar, aba_faixas = st.tabs(
        ["Frequência", "Atraso", "Par/Ímpar", "Faixas"]
    )

    # FREQUÊNCIA
    with aba_freq:
        st.subheader("Frequência das dezenas")
        st.info(
            "Mostra quantas vezes cada dezena já foi sorteada no histórico, "
            "ajudando a identificar números mais e menos frequentes. [web:324][web:335]"
        )
        try:
            freq_df = analyze_frequency(df_hist)
            st.dataframe(freq_df, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao calcular frequência: {e}")

    # ATRASO
    with aba_atraso:
        st.subheader("Atraso das dezenas")
        st.info(
            "Mostra há quantos concursos cada dezena não aparece, além de atrasos médios "
            "e máximos, destacando dezenas consideradas 'atrasadas'. [web:323][web:326]"
        )
        try:
            atraso_df = analyze_delay(df_hist)
            st.dataframe(atraso_df, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao calcular atraso: {e}")

    # PAR / ÍMPAR
    with aba_par_impar:
        st.subheader("Distribuição par/ímpar")
        st.info(
            "Analisa quantos números pares e ímpares saem em cada sorteio, "
            "mostrando quais composições aparecem com mais frequência. [web:317][web:320]"
        )
        try:
            par_impar_df = analyze_par_impar(df_hist)
            st.dataframe(par_impar_df, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao calcular distribuição par/ímpar: {e}")

    # FAIXAS
    with aba_faixas:
        st.subheader("Distribuição por faixas")
        st.info(
            "Agrupa as dezenas em intervalos (1–20, 21–40, 41–60) e mostra "
            "como os resultados se distribuem nessas faixas. [web:319][web:328]"
        )
        try:
            df_tmp = df_hist.copy()
            dezenas_cols = [c for c in df_tmp.columns if c.startswith("d")]
            df_melt = df_tmp.melt(value_vars=dezenas_cols, value_name="dezena")
            df_melt["faixa"] = pd.cut(
                df_melt["dezena"],
                bins=[0, 20, 40, 60],
                labels=["1-20", "21-40", "41-60"],
            )
            faixas_df = (
                df_melt.groupby("faixa")["dezena"].count().reset_index(name="qtd")
            )
            st.dataframe(faixas_df, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao calcular distribuição por faixas: {e}")


# ---------------- PAGINA: SOBRE (EXEMPLO SIMPLES) ----------------

def pagina_sobre():
    st.header("Sobre o projeto")
    st.markdown(
        """
Aplicativo para estudo estatístico da Mega-Sena e geração de jogos com diferentes estratégias.  
As análises são baseadas em dados históricos e servem apenas como apoio visual e educacional. [web:335]
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
