"""Funções puras (sem Streamlit) para ler e resumir o DataSet."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def carregar_dataset() -> pd.DataFrame:
    return pd.read_csv(config.CAMINHO_DATASET)


def colunas_ausentes(df: pd.DataFrame) -> list[str]:
    obrigatorias = config.COLUNAS_SENSORES + [config.COLUNA_ALVO]
    return [c for c in obrigatorias if c not in df.columns]


def ordem_classes(df: pd.DataFrame) -> list[str]:
    """Classes presentes no DataSet: primeiro as conhecidas, depois as demais."""
    presentes = [str(c) for c in df[config.COLUNA_ALVO].dropna().unique()]
    conhecidas = [m for m in config.MOVIMENTOS if m in presentes]
    outras = sorted(c for c in presentes if c not in config.MOVIMENTOS)
    return conhecidas + outras


def resumo(df: pd.DataFrame) -> dict:
    return {
        "registros": int(len(df)),
        "atributos": len(config.COLUNAS_SENSORES),
        "classes": len(ordem_classes(df)),
        "ausentes": int(df[config.COLUNAS_SENSORES + [config.COLUNA_ALVO]].isna().sum().sum()),
        "duplicadas": int(df.duplicated().sum()),
    }


def contagem_por_classe(df: pd.DataFrame) -> pd.DataFrame:
    contagem = df[config.COLUNA_ALVO].astype(str).value_counts()
    ordem = ordem_classes(df)
    out = pd.DataFrame(
        {"Movimento": ordem, "Registros": [int(contagem[c]) for c in ordem]}
    )
    out["Percentual (%)"] = (out["Registros"] / out["Registros"].sum() * 100).round(2)
    return out


def separar_treino_teste(df: pd.DataFrame):
    """Mesmo split de 1_preparacao_dados.py (80/20, estratificado, seed 42)."""
    X = df[config.COLUNAS_SENSORES]
    y = df[config.COLUNA_ALVO]
    return train_test_split(
        X, y,
        test_size=config.TAMANHO_TESTE,
        random_state=config.SEMENTE,
        stratify=y,
    )


def faixas_dos_sensores(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Mínimo e máximo observados em cada sensor (usados nos filtros por faixa)."""
    return {
        c: (float(df[c].min()), float(df[c].max())) for c in config.COLUNAS_SENSORES
    }


def filtrar(
    df: pd.DataFrame,
    movimentos: list[str],
    faixas: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Filtra por movimento e, opcionalmente, por faixa de valor de cada sensor.

    Retorna uma cópia; o DataFrame original nunca é alterado.
    """
    mascara = df[config.COLUNA_ALVO].astype(str).isin(movimentos)
    for coluna, (minimo, maximo) in (faixas or {}).items():
        mascara &= df[coluna].between(minimo, maximo)
    return df[mascara].copy()


ESTATISTICAS = {
    "Média": "mean",
    "Mediana": "median",
    "Desvio padrão": "std",
    "Mínimo": "min",
    "Máximo": "max",
}


def estatistica_por_classe(df: pd.DataFrame, nome: str) -> pd.DataFrame:
    """Uma estatística de cada sensor, por movimento (linhas na ordem das classes)."""
    tabela = (
        df.assign(**{config.COLUNA_ALVO: df[config.COLUNA_ALVO].astype(str)})
        .groupby(config.COLUNA_ALVO)[config.COLUNAS_SENSORES]
        .agg(ESTATISTICAS[nome])
    )
    ordem = [c for c in ordem_classes(df) if c in tabela.index]
    tabela = tabela.loc[ordem].round(2)
    tabela.index.name = "Movimento"
    return tabela


def correlacao(df: pd.DataFrame) -> pd.DataFrame:
    return df[config.COLUNAS_SENSORES].corr()


def sortear_amostra(X_teste: pd.DataFrame, y_teste: pd.Series, classe: str | None = None):
    """Sorteia uma amostra real do conjunto de teste (opcionalmente de uma classe).

    Retorna (valores dos sensores, movimento real).
    """
    rotulos = y_teste.astype(str)
    candidatos = rotulos[rotulos == classe].index if classe else rotulos.index
    if len(candidatos) == 0:
        raise ValueError(f"Não há amostras de teste da classe '{classe}'.")
    escolhido = pd.Series(candidatos).sample(1).iloc[0]
    valores = {c: float(X_teste.loc[escolhido, c]) for c in config.COLUNAS_SENSORES}
    return valores, str(rotulos.loc[escolhido])


def fora_da_faixa(df: pd.DataFrame, valores: dict) -> list[dict]:
    """Sensores cujo valor informado está fora da faixa observada no DataSet."""
    fora = []
    for sensor, (minimo, maximo) in faixas_dos_sensores(df).items():
        if not (minimo <= valores[sensor] <= maximo):
            fora.append(
                {"sensor": sensor, "valor": valores[sensor], "minimo": minimo, "maximo": maximo}
            )
    return fora


def comparar_com_classe(df: pd.DataFrame, valores: dict, classe: str) -> pd.DataFrame:
    """Compara os valores informados com a média da classe, em desvios-padrão."""
    sub = df[df[config.COLUNA_ALVO].astype(str) == classe][config.COLUNAS_SENSORES]
    if sub.empty:
        return pd.DataFrame()
    media = sub.mean()
    desvio = sub.std().replace(0, float("nan"))
    informado = pd.Series(valores)[config.COLUNAS_SENSORES]
    return pd.DataFrame(
        {
            "Sensor": config.COLUNAS_SENSORES,
            "Valor informado": informado.round(2).values,
            "Média da classe prevista": media.round(2).values,
            "Distância da média (desvios-padrão)": ((informado - media) / desvio).round(2).values,
        }
    )
