import streamlit as st

from utils import config, dados, modelo, ui

st.title("📊 Visão geral")
st.caption("Panorama do DataSet e do modelo utilizados no classificador de movimentos.")

df = ui.exigir_dataset()
resumo = dados.resumo(df)

# --- Indicadores do DataSet -------------------------------------------------
indicadores = [
    ("Registros", ui.fmt_int(resumo["registros"])),
    ("Movimentos (classes)", str(resumo["classes"])),
    ("Atributos (sensores)", str(resumo["atributos"])),
    ("Valores ausentes", ui.fmt_int(resumo["ausentes"])),
]
for coluna, (rotulo, valor) in zip(st.columns(4), indicadores):
    with coluna, st.container(border=True):
        st.metric(rotulo, valor)

st.caption(
    f"Atributos: {', '.join(config.COLUNAS_SENSORES)} · "
    f"Linhas duplicadas no DataSet: {ui.fmt_int(resumo['duplicadas'])}"
)

# --- Distribuição das classes ----------------------------------------------
st.subheader("Distribuição dos movimentos")
contagem = dados.contagem_por_classe(df)
esq, dir_ = st.columns([3, 2])
with esq:
    st.bar_chart(contagem.set_index("Movimento")["Registros"])
with dir_:
    st.dataframe(contagem, hide_index=True)
    menor = contagem["Registros"].min()
    if menor > 0:
        razao = contagem["Registros"].max() / menor
        st.caption(f"Razão entre a maior e a menor classe: {razao:.2f}×")

# --- Modelo ----------------------------------------------------------------
st.subheader("Modelo utilizado")
mod = ui.exigir_modelo()
info = modelo.descrever(mod)

esq, dir_ = st.columns(2)
with esq, st.container(border=True):
    st.markdown(f"**Algoritmo:** {info['tipo']}")
    if info["n_estimators"] is not None:
        st.markdown(f"**Número de árvores:** {info['n_estimators']}")
        profundidade = info["max_depth"] if info["max_depth"] is not None else "sem limite"
        st.markdown(f"**Profundidade máxima:** {profundidade}")
    if info["random_state"] is not None:
        st.markdown(f"**random_state:** {info['random_state']}")
    if info["classes"]:
        st.markdown(f"**Classes:** {', '.join(info['classes'])}")
    st.markdown(f"**Arquivo:** `{config.CAMINHO_MODELO.name}`")

with dir_, st.container(border=True):
    avaliacao = ui.obter_avaliacao()
    if avaliacao is None:
        st.info("Métricas indisponíveis no momento.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Acurácia", f"{avaliacao['acuracia'] * 100:.2f}%")
        m2.metric("F1-score (macro)", f"{avaliacao['f1_macro'] * 100:.2f}%")
        m3.metric("Amostras de teste", ui.fmt_int(avaliacao["n_teste"]))
        st.caption(
            "Calculado agora, no conjunto de teste (20 %) com o mesmo split de "
            "`1_preparacao_dados.py` (estratificado, random_state=42)."
        )

importancias = modelo.importancias(mod)
if importancias is not None:
    st.subheader("Importância dos atributos")
    st.bar_chart(importancias.set_index("Atributo")["Importância"])
    st.caption(
        "Importância por redução de impureza (Gini) do Random Forest: mostra a "
        "contribuição relativa de cada atributo no modelo, não uma relação causal."
    )

with st.expander("Pré-visualização do DataSet (50 primeiras linhas)"):
    st.dataframe(df.head(50))
