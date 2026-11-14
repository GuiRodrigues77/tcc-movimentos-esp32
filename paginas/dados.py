import streamlit as st

from utils import config, dados, graficos, ui

st.title("📈 Dados")
st.caption(
    "Exploração do DataSet: compare os movimentos e veja como acelerômetro e "
    "giroscópio se comportam em cada um."
)

df = ui.exigir_dataset()
classes = dados.ordem_classes(df)

# --- Filtros ----------------------------------------------------------------
with st.container(border=True):
    movimentos = st.multiselect("Movimentos", classes, default=classes)

    faixas_totais = dados.faixas_dos_sensores(df)
    faixas = {}
    with st.expander("Filtrar por faixa de valores dos sensores"):
        colunas = st.columns(3)
        for i, sensor in enumerate(config.COLUNAS_SENSORES):
            minimo, maximo = faixas_totais[sensor]
            with colunas[i % 3]:
                if minimo < maximo:
                    faixas[sensor] = st.slider(sensor, minimo, maximo, (minimo, maximo))
                else:
                    st.caption(f"{sensor}: valor único ({minimo:g})")

if not movimentos:
    st.warning("Selecione ao menos um movimento.")
    st.stop()

filtrado = dados.filtrar(df, movimentos, faixas)
if filtrado.empty:
    st.warning("Nenhum registro atende aos filtros escolhidos.")
    st.stop()

classes_filtro = [c for c in classes if c in movimentos]
c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.metric("Registros no filtro", ui.fmt_int(len(filtrado)))
with c2, st.container(border=True):
    st.metric("Do total do DataSet", f"{len(filtrado) / len(df) * 100:.1f}%")

dados_grafico = graficos.preparar(filtrado)

aba_dist, aba_comp, aba_rel, aba_est, aba_tab = st.tabs(
    ["Distribuição", "Comparação", "Relação entre eixos", "Estatísticas", "Tabela"]
)

# --- Distribuição -----------------------------------------------------------
with aba_dist:
    sensor = st.selectbox("Sensor", config.COLUNAS_SENSORES, key="sensor_hist")
    st.altair_chart(graficos.histograma(dados_grafico, sensor, classes_filtro))
    st.caption("Histograma sobreposto: cada cor é um movimento.")

# --- Comparação entre movimentos -------------------------------------------
with aba_comp:
    grupo = st.radio("Sensor", ["Acelerômetro", "Giroscópio"], horizontal=True)
    eixos = config.COLUNAS_ACC if grupo == "Acelerômetro" else config.COLUNAS_GYRO
    for coluna, eixo in zip(st.columns(3), eixos):
        with coluna:
            st.altair_chart(graficos.boxplot(dados_grafico, eixo, classes_filtro))
    st.caption(
        "Boxplot: a caixa vai do 1º ao 3º quartil, a linha central é a mediana e "
        "os pontos isolados são valores atípicos."
    )

# --- Relação entre eixos ----------------------------------------------------
with aba_rel:
    cx, cy = st.columns(2)
    eixo_x = cx.selectbox("Eixo X", config.COLUNAS_SENSORES, index=0)
    eixo_y = cy.selectbox("Eixo Y", config.COLUNAS_SENSORES, index=1)
    st.altair_chart(graficos.dispersao(dados_grafico, eixo_x, eixo_y, classes_filtro))
    st.caption("Cada ponto é um registro do DataSet. Dá para dar zoom e arrastar.")

# --- Estatísticas -----------------------------------------------------------
with aba_est:
    nome = st.selectbox("Estatística por movimento", list(dados.ESTATISTICAS))
    st.dataframe(dados.estatistica_por_classe(filtrado, nome))
    if len(filtrado) > 1:
        st.markdown("**Correlação entre os sensores (registros filtrados)**")
        st.altair_chart(graficos.mapa_correlacao(dados.correlacao(filtrado)))

# --- Tabela -----------------------------------------------------------------
with aba_tab:
    st.dataframe(filtrado)
    st.download_button(
        "Baixar registros filtrados (CSV)",
        data=filtrado.to_csv(index=False).encode("utf-8"),
        file_name="dados_filtrados.csv",
        mime="text/csv",
    )
