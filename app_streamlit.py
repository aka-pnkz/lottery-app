import streamlit as st
import pandas as pd
import numpy as np
import itertools
from datetime import datetime

# ==========================
# CONFIG GERAL
# ==========================
st.set_page_config(
    page_title="Mega Sena Helper",
    page_icon="🎰",
    layout="wide",
)

# ==========================
# FUNÇÕES DE DADOS
# ==========================
@st.cache_data
def carregar_concursos(caminho_csv: str = "megasena.csv") -> pd.DataFrame:
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


def contar_setores(jogo: list[int]) -> tuple[int, int, int]:
    """
    Setores:
      S1 = 1–20
      S2 = 21–40
      S3 = 41–60
    """
    s1 = sum(1 for d in jogo if 1 <= d <= 20)
    s2 = sum(1 for d in jogo if 21 <= d <= 40)
    s3 = sum(1 for d in jogo if 41 <= d <= 60)
    return s1, s2, s3


def tem_sequencia_longa(jogo: list[int], limite: int = 3) -> bool:
    """
    True se existir sequência com >= limite dezenas consecutivas.
    """
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
# ESTRATÉGIAS
# ==========================
def gerar_aleatorio_puro(qtd_jogos: int, tam_jogo: int) -> list[list[int]]:
    universe = np.arange(1, 61)
    jogos = []
    for _ in range(qtd_jogos):
        dezenas = np.random.choice(universe, size=tam_jogo, replace=False)
        jogos.append(sorted(dezenas.tolist()))
    return jogos


def gerar_balanceado_par_impar(qtd_jogos: int, tam_jogo: int) -> list[list[int]]:
    """
    Tenta manter 3–3 ou 4–2 / 2–4 de par/ímpar. [web:56][web:59]
    """
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
                # Para jogos > 6, aceita se não for extremo (tudo par/ímpar)
                if pares not in (0, tam_jogo) and impares not in (0, tam_jogo):
                    jogos.append(sorted(dezenas.tolist()))
                    break

            if tentativas > 50:
                jogos.append(sorted(dezenas.tolist()))
                break

    return jogos


def gerar_setorial(qtd_jogos: int, tam_jogo: int) -> list[list[int]]:
    """
    Padrão base para 6 dezenas: 2 de cada setor (1–20, 21–40, 41–60). [web:63][web:66]
    Para mais dezenas, distribui proporcionalmente.
    """
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

        dezenas = np.concatenate([
            np.random.choice(s1, size=min(q1, len(s1)), replace=False),
            np.random.choice(s2, size=min(q2, len(s2)), replace=False),
            np.random.choice(s3, size=min(q3, len(s3)), replace=False),
        ])

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
    """
    Mix quentes + atrasadas + intermediárias. [memory:50]
    proporcao = (quentes, frias, neutras) para jogo de 6 dezenas.
    """
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
                np.random.choice(quentes, size=min(q_quentes, len(quentes)), replace=False)
            )
        if len(frias) > 0 and q_frias > 0:
            dezenas.extend(
                np.random.choice(frias, size=min(q_frias, len(frias)), replace=False)
            )
        if len(neutras) > 0 and q_neutras > 0:
            dezenas.extend(
                np.random.choice(neutras, size=min(q_neutras, len(neutras)), replace=False)
            )

        if len(dezenas) < tam_jogo:
            universe = np.setdiff1d(np.arange(1, 61), dezenas)
            extra = np.random.choice(universe, size=tam_jogo - len(dezenas), replace=False)
            dezenas = np.concatenate([dezenas, extra])

        jogos.append(sorted(list(map(int, dezenas))))

    return jogos


def gerar_sem_sequencias(
    qtd_jogos: int,
    tam_jogo: int,
    limite_sequencia: int = 3,
) -> list[list[int]]:
    """
    Sorteia aleatório, rejeitando jogos com sequências longas. [memory:45]
    """
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
    """
    Mini “wheel system”: gera combinações de 6 dezenas dentro de uma base,
    limitado por max_jogos para não explodir. [web:53][web:55][web:60]
    """
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
# SIDEBAR
# ==========================
with st.sidebar:
    st.title("Mega Sena Helper 🎰")

    st.markdown("### Dados")
    caminho_csv = st.text_input(
        "Caminho/arquivo CSV dos concursos",
        value="megasena.csv",
    )

    st.markdown("### Estratégia")
    estrategia = st.selectbox(
        "Escolha a estratégia",
        [
            "Aleatório puro",
            "Balanceado par/ímpar",
            "Setorial (faixas)",
            "Quentes/Frias/Mix",
            "Sem sequências longas",
            "Wheeling simples (base fixa)",
        ],
    )

    qtd_jogos = st.number_input(
        "Quantidade de jogos",
        min_value=1,
        max_value=200,
        value=10,
        step=1,
    )

    tam_jogo = st.slider(
        "Dezenas por jogo",
        min_value=6,
        max_value=15,
        value=6,
    )

    if estrategia == "Sem sequências longas":
        limite_seq = st.number_input(
            "Máx. tamanho de sequência permitida",
            min_value=2,
            max_value=6,
            value=3,
        )
    else:
        limite_seq = 3

    if estrategia == "Quentes/Frias/Mix":
        st.caption("Padrão 3 quentes, 2 frias, 1 neutra para jogo de 6 dezenas.")
        q_quentes = st.number_input("Qtd. quentes", 0, 6, 3)
        q_frias = st.number_input("Qtd. frias", 0, 6, 2)
        q_neutras = st.number_input("Qtd. neutras", 0, 6, 1)
    else:
        q_quentes, q_frias, q_neutras = 3, 2, 1

    if estrategia == "Wheeling simples (base fixa)":
        st.caption("Informe uma base de dezenas (ex.: 10–15 números) separados por vírgula.")
        base_str = st.text_input("Base de dezenas", "1,3,5,7,9,11,13,15,17,19")
    else:
        base_str = ""

    st.markdown("---")
    mostrar_frequencias = st.checkbox("Mostrar tabela de frequências", value=True)


# ==========================
# CORPO PRINCIPAL
# ==========================
st.title("Gerador de Jogos para Mega-Sena")

st.markdown(
    "Ferramenta de estudo estatístico e organização de jogos. "
    "As estratégias não aumentam a probabilidade matemática de cada combinação, "
    "apenas ajudam a evitar padrões extremos e a usar o histórico de forma organizada."
)

# Explicação da estratégia escolhida
explicacoes = {
    "Aleatório puro": (
        "Sorteia dezenas totalmente aleatórias entre 1 e 60. "
        "É a abordagem mais alinhada com o fato de que todas as combinações têm a mesma chance."
    ),
    "Balanceado par/ímpar": (
        "Busca jogos com distribuição equilibrada de pares e ímpares, como 3–3 ou 4–2, "
        "evitando padrões extremos (tudo par ou tudo ímpar), que são raros no histórico."
    ),
    "Setorial (faixas)": (
        "Distribui dezenas entre faixas 1–20, 21–40 e 41–60, normalmente em 2–2–2 para 6 dezenas, "
        "evitando concentrar todos os números em um único trecho do volante."
    ),
    "Quentes/Frias/Mix": (
        "Usa o histórico: combina dezenas mais sorteadas (quentes), mais atrasadas (frias) e neutras. "
        "É uma forma popular de usar frequência sem depender apenas de um tipo de número."
    ),
    "Sem sequências longas": (
        "Gera jogos aleatórios rejeitando combinações com muitas dezenas consecutivas "
        "(por exemplo 10–11–12–13), que muitos apostadores preferem evitar."
    ),
    "Wheeling simples (base fixa)": (
        "Gera combinações de 6 dezenas dentro de uma base fixa de números, formando um mini ‘fechamento’. "
        "Ajuda a cobrir melhor um conjunto de dezenas escolhido, mas aumenta o número de jogos."
    ),
}

st.info(explicacoes.get(estrategia, ""))

# Carrega dados e gera
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
            st.dataframe(freq_df, use_container_width=True, height=400)

    with col2:
        st.subheader("Parâmetros selecionados")
        st.write(f"- Estratégia: **{estrategia}**")
        st.write(f"- Jogos: **{qtd_jogos}**")
        st.write(f"- Dezenas por jogo: **{tam_jogo}**")

        if estrategia == "Quentes/Frias/Mix":
            st.write(
                f"- Mix: **{q_quentes} quentes, {q_frias} frias, {q_neutras} neutras**"
            )
        if estrategia == "Sem sequências longas":
            st.write(f"- Máx. sequência: **{limite_seq}** dezenas seguidas")

        if estrategia == "Wheeling simples (base fixa)":
            st.write(f"- Base: `{base_str}`")

        gerar = st.button("Gerar jogos agora", type="primary")

        if gerar:
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
                    base_dezenas = [
                        int(x.strip())
                        for x in base_str.split(",")
                        if x.strip().isdigit()
                    ]
                    jogos = gerar_wheeling_simples(
                        base_dezenas=base_dezenas,
                        max_jogos=qtd_jogos,
                    )
                except Exception:
                    jogos = []

            else:
                jogos = []

            st.markdown("### Jogos gerados")
            if not jogos:
                st.warning("Nenhum jogo gerado. Verifique os parâmetros.")
            else:
                for i, jogo in enumerate(jogos, start=1):
                    st.code(f"Jogo {i:02d}: {formatar_jogo(jogo)}")

                jogos_df = pd.DataFrame(
                    jogos,
                    columns=[f"d{i}" for i in range(1, len(jogos[0]) + 1)],
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
