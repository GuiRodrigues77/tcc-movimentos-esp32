import pandas as pd
import streamlit as st

from utils import config, graficos, modelo, ui

st.title("🤖 Modelo e métricas")
st.caption(
    "Desempenho dos algoritmos no conjunto de teste (20 % do DataSet, split estratificado "
    "com random_state=42). Todos os números são calculados a partir dos dados."
)

ui.exigir_dataset()
ui.exigir_modelo()
resultados = ui.obter_comparacao()
if resultados is None:
    st.stop()

nomes = list(resultados)
n_teste = resultados[nomes[0]]["n_teste"]

# --- Comparação entre modelos ----------------------------------------------
st.subheader("Comparação entre algoritmos")
linhas = [
    {
        "Modelo": nome,
        "Acurácia (%)": r["acuracia"] * 100,
        "Precisão macro (%)": r["precisao_macro"] * 100,
        "Recall macro (%)": r["recall_macro"] * 100,
        "F1 macro (%)": r["f1_macro"] * 100,
        "Acertos": f"{round(r['acuracia'] * r['n_teste'])} de {r['n_teste']}",
    }
    for nome, r in resultados.items()
]
tabela = pd.DataFrame(linhas)
esq, dir_ = st.columns([3, 2])
with esq:
    longo = tabela.melt(
        id_vars="Modelo",
        value_vars=["Acurácia (%)", "F1 macro (%)"],
        var_name="Métrica",
        value_name="Valor",
    )
    longo["Métrica"] = longo["Métrica"].str.replace(" (%)", "", regex=False)
    st.altair_chart(graficos.comparacao_modelos(longo))
with dir_:
    st.dataframe(tabela.round(2), hide_index=True)

acertos = {nome: round(r["acuracia"] * r["n_teste"]) for nome, r in resultados.items()}
melhor, pior = max(acertos, key=acertos.get), min(acertos, key=acertos.get)
st.caption(
    f"Cada amostra de teste vale {100 / n_teste:.2f} ponto percentual. {melhor} acertou "
    f"{acertos[melhor]} de {n_teste} e {pior} acertou {acertos[pior]}; diferenças de poucas "
    "amostras devem ser interpretadas com cautela. O Random Forest é o modelo carregado de "
    f"`{config.CAMINHO_MODELO.name}`; KNN (k=5) e SVM são treinados aqui com os mesmos "
    "parâmetros de `4_testar_knn_svm.py`."
)

# --- Detalhe de um modelo ---------------------------------------------------
st.subheader("Detalhes do modelo")
escolhido = st.radio("Algoritmo", nomes, horizontal=True)
r = resultados[escolhido]

indicadores = [
    ("Acurácia", r["acuracia"]),
    ("Precisão (macro)", r["precisao_macro"]),
    ("Recall (macro)", r["recall_macro"]),
    ("F1-score (macro)", r["f1_macro"]),
]
for coluna, (rotulo, valor) in zip(st.columns(4), indicadores):
    with coluna, st.container(border=True):
        st.metric(rotulo, f"{valor * 100:.2f}%")

st.markdown("**Métricas por movimento**")
por_classe = r["por_classe"].copy()
por_classe["Suporte"] = por_classe["Suporte"].astype(int)
st.dataframe(por_classe.round(2))

st.markdown("**Matriz de confusão**")
em_percentual = st.toggle("Mostrar em % de cada movimento real", value=False)
st.altair_chart(graficos.matriz_confusao(r["matriz"], em_percentual))
st.caption("Cada linha é o movimento que realmente foi executado; cada coluna, o que o modelo previu.")

confusoes = modelo.principais_confusoes(r["matriz"])
if confusoes:
    st.markdown("**Principais confusões**")
    for real, previsto, quantidade in confusoes:
        st.markdown(f"- {real} classificado como {previsto}: {quantidade} amostra(s)")
else:
    st.success("Nenhuma confusão no conjunto de teste.")

imagem = config.RAIZ / "matriz_confusao.png"
if escolhido == "Random Forest" and imagem.exists():
    with st.expander("Imagem gerada por 2_treinamento_rf.py"):
        st.image(str(imagem))
