import pandas as pd
import streamlit as st

from utils import banco, config, dados, ui

banco.init_db()
modelo = ui.exigir_modelo()
df = ui.exigir_dataset()
X_teste, y_teste = ui.exigir_conjunto_teste()
classes = dados.ordem_classes(df)

PADRAO = {
    "ACC_X": 1500.0, "ACC_Y": -3200.0, "ACC_Z": 18000.0,
    "GYRO_X": -200.0, "GYRO_Y": 70.0, "GYRO_Z": 150.0,
}
for sensor, valor in PADRAO.items():
    st.session_state.setdefault(f"in_{sensor}", valor)


def sortear():
    escolha = st.session_state.get("classe_amostra", "Qualquer")
    valores, rotulo = dados.sortear_amostra(
        X_teste, y_teste, None if escolha == "Qualquer" else escolha
    )
    for sensor, valor in valores.items():
        st.session_state[f"in_{sensor}"] = valor
    st.session_state["amostra"] = {"valores": valores, "rotulo": rotulo}


st.title("🧪 Teste de classificação")
st.write(
    "Carregue uma amostra real do DataSet ou digite os valores do acelerômetro e do "
    "giroscópio (MPU6050). Nesta versão não há leitura automática do ESP32."
)

# --- Amostra do DataSet -----------------------------------------------------
with st.container(border=True):
    st.markdown("**Amostra do DataSet**")
    st.selectbox("Movimento da amostra", ["Qualquer"] + classes, key="classe_amostra")
    st.button("🎲 Sortear amostra do conjunto de teste", on_click=sortear)
    st.caption(
        f"Sorteia entre as {len(X_teste)} amostras de teste, que o modelo não viu no "
        "treinamento. O movimento real só aparece depois de classificar."
    )

# --- Valores dos sensores ---------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Acelerômetro")
    acc_x = st.number_input("ACC_X", key="in_ACC_X")
    acc_y = st.number_input("ACC_Y", key="in_ACC_Y")
    acc_z = st.number_input("ACC_Z", key="in_ACC_Z")
with col2:
    st.markdown("### Giroscópio")
    gyro_x = st.number_input("GYRO_X", key="in_GYRO_X")
    gyro_y = st.number_input("GYRO_Y", key="in_GYRO_Y")
    gyro_z = st.number_input("GYRO_Z", key="in_GYRO_Z")

valores = {
    "ACC_X": acc_x, "ACC_Y": acc_y, "ACC_Z": acc_z,
    "GYRO_X": gyro_x, "GYRO_Y": gyro_y, "GYRO_Z": gyro_z,
}

fora = dados.fora_da_faixa(df, valores)
if fora:
    linhas = [
        f"- **{f['sensor']}** = {f['valor']:g} (no DataSet: de {f['minimo']:g} a {f['maximo']:g})"
        for f in fora
    ]
    st.warning(
        "Há valores fora da faixa observada no DataSet. O modelo não viu leituras "
        "assim no treinamento, então o resultado pode não ser confiável.\n\n"
        + "\n".join(linhas)
    )

# --- Classificação ----------------------------------------------------------
if st.button("Classificar movimento", type="primary"):
    entrada = pd.DataFrame([valores])[config.COLUNAS_SENSORES]
    resultado = str(modelo.predict(entrada)[0])

    amostra = st.session_state.get("amostra")
    veio_do_teste = amostra is not None and amostra["valores"] == valores
    real = amostra["rotulo"] if veio_do_teste else None
    banco.salvar_predicao(
        valores,
        resultado,
        origem=banco.ORIGEM_AMOSTRA_TESTE if veio_do_teste else banco.ORIGEM_MANUAL,
        movimento_real=real,
    )

    st.success(f"### Movimento identificado pelo modelo: **{resultado}**")
    if real is not None:
        if real == resultado:
            st.success(f"Movimento real no DataSet: **{real}**. O modelo acertou.")
        else:
            st.error(f"Movimento real no DataSet: **{real}**. O modelo errou.")
    else:
        st.caption("Valores digitados: não há movimento real para comparar.")

    proba = None
    if hasattr(modelo, "predict_proba"):
        try:
            proba = dict(zip([str(c) for c in modelo.classes_], modelo.predict_proba(entrada)[0]))
        except Exception:
            proba = None

    if proba:
        st.markdown("**Votos do modelo por movimento**")
        for classe, p in sorted(proba.items(), key=lambda item: -item[1]):
            st.progress(float(p), text=f"{classe}: {p * 100:.0f}%")
        st.caption(
            "No Random Forest, é a média dos votos das árvores: mostra o quanto elas "
            "concordam, não uma probabilidade calibrada."
        )

    with st.expander("Por que esse resultado?", expanded=True):
        n_arvores = getattr(modelo, "n_estimators", None)
        if proba and n_arvores:
            st.write(
                f"O Random Forest combina {n_arvores} árvores de decisão. "
                f"Cerca de {proba.get(resultado, 0) * 100:.0f}% delas indicaram **{resultado}**."
            )
        comparacao = dados.comparar_com_classe(df, valores, resultado)
        if not comparacao.empty:
            st.write(f"Seus valores comparados com a média de **{resultado}** no DataSet:")
            st.dataframe(comparacao, hide_index=True)
            st.caption(
                "Distância próxima de 0 indica uma leitura típica desse movimento; "
                "valores acima de 2 ou abaixo de -2 são incomuns para ele."
            )
