"""Camada Streamlit: carregamento com cache e mensagens de erro amigáveis."""
import streamlit as st

from . import config, dados, modelo


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


@st.cache_data(show_spinner=False)
def _dataset(mtime: float):
    return dados.carregar_dataset()


@st.cache_resource(show_spinner=False)
def _modelo(mtime: float):
    return modelo.carregar_modelo()


@st.cache_data(show_spinner="Avaliando o modelo no conjunto de teste...")
def _avaliacao(mtime_dados: float, mtime_modelo: float):
    return modelo.avaliar_holdout(dados.carregar_dataset(), modelo.carregar_modelo())


@st.cache_data(show_spinner=False)
def _conjunto_teste(mtime: float):
    _, X_teste, _, y_teste = dados.separar_treino_teste(dados.carregar_dataset())
    return X_teste, y_teste


@st.cache_data(show_spinner="Avaliando os modelos no conjunto de teste...")
def _comparacao(mtime_dados: float, mtime_modelo: float):
    df = dados.carregar_dataset()
    X_treino, X_teste, y_treino, y_teste = dados.separar_treino_teste(df)
    modelos = {"Random Forest": modelo.carregar_modelo()}
    modelos.update(modelo.treinar_alternativos(X_treino, y_treino))
    return modelo.comparar_modelos(modelos, X_teste, y_teste, dados.ordem_classes(df))


def obter_comparacao():
    """Avaliação dos 3 modelos, ou None se não for possível calcular."""
    try:
        return _comparacao(
            config.CAMINHO_DATASET.stat().st_mtime,
            config.CAMINHO_MODELO.stat().st_mtime,
        )
    except Exception as erro:
        st.warning(f"Não foi possível avaliar os modelos: {erro}")
        return None


def exigir_conjunto_teste():
    """(X_teste, y_teste): as mesmas amostras de teste usadas na avaliação."""
    exigir_dataset()
    return _conjunto_teste(config.CAMINHO_DATASET.stat().st_mtime)


def exigir_dataset():
    caminho = config.CAMINHO_DATASET
    if not caminho.exists():
        st.error(f"DataSet não encontrado: `{caminho.name}` (esperado na raiz do projeto).")
        st.stop()
    df = _dataset(caminho.stat().st_mtime)
    faltando = dados.colunas_ausentes(df)
    if faltando:
        st.error(f"O DataSet não possui as colunas esperadas: {', '.join(faltando)}.")
        st.stop()
    return df


def exigir_modelo():
    caminho = config.CAMINHO_MODELO
    if not caminho.exists():
        st.error(
            f"Modelo não encontrado: `{caminho.name}`. "
            "Execute `3_salvar_modelo.py` para gerá-lo."
        )
        st.stop()
    try:
        return _modelo(caminho.stat().st_mtime)
    except Exception as erro:
        st.error(f"Não foi possível carregar o modelo: {erro}")
        st.caption(
            "Se o erro citar versões, gere o modelo novamente com "
            "`3_salvar_modelo.py` no mesmo ambiente em que o app roda."
        )
        st.stop()


def obter_avaliacao():
    """Métricas no conjunto de teste, ou None se não for possível calcular."""
    try:
        return _avaliacao(
            config.CAMINHO_DATASET.stat().st_mtime,
            config.CAMINHO_MODELO.stat().st_mtime,
        )
    except Exception as erro:
        st.warning(f"Não foi possível calcular as métricas: {erro}")
        return None
