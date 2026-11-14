"""Ponto de entrada do app (navegação). Execute: streamlit run app_tcc.py"""
import streamlit as st

st.set_page_config(
    page_title="TCC - Classificação de Movimentos",
    page_icon="🏃‍♂️",
    layout="wide",
)

navegacao = st.navigation(
    {
        "Análise": [
            st.Page("paginas/visao_geral.py", title="Visão geral", icon="📊", default=True),
            st.Page("paginas/dados.py", title="Dados", icon="📈"),
            st.Page("paginas/modelo.py", title="Modelo e métricas", icon="🤖"),
        ],
        "Aplicação": [
            st.Page("paginas/classificacao.py", title="Classificação", icon="🧪"),
            st.Page("paginas/historico.py", title="Histórico", icon="🗄️"),
        ],
    }
)

with st.sidebar:
    st.divider()
    st.caption("TCC - Engenharia de Computação")
    st.caption("Classificação de Movimentos Humanos utilizando Sensores Inerciais e Aprendizado de Máquina")

navegacao.run()
