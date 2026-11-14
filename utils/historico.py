"""Funções puras (sem Streamlit) para preparar e filtrar o histórico do SQLite."""
import pandas as pd

from . import banco, config

ROTULOS_ORIGEM = {
    banco.ORIGEM_MANUAL: "Digitado manualmente",
    banco.ORIGEM_AMOSTRA_TESTE: "Amostra do DataSet (teste)",
}
ROTULO_SEM_ORIGEM = "Sem origem registrada (versão anterior)"

ACERTOU, ERROU, SEM_REFERENCIA = "Acertou", "Errou", "Sem referência"

_RENOMEAR = {c.lower(): c for c in config.COLUNAS_SENSORES}


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta Origem e Resultado e converte a data/hora."""
    saida = df.copy()
    saida["data_hora"] = pd.to_datetime(saida["data_hora"], errors="coerce")
    saida["Origem"] = saida["origem"].map(ROTULOS_ORIGEM).fillna(ROTULO_SEM_ORIGEM)

    real, previsto = saida["movimento_real"], saida["movimento_previsto"]
    saida["Resultado"] = SEM_REFERENCIA
    saida.loc[real.notna() & (real == previsto), "Resultado"] = ACERTOU
    saida.loc[real.notna() & (real != previsto), "Resultado"] = ERROU
    return saida


def filtrar(
    df: pd.DataFrame,
    origens: list[str],
    movimentos: list[str],
    resultados: list[str],
    periodo: tuple | None = None,
) -> pd.DataFrame:
    mascara = (
        df["Origem"].isin(origens)
        & df["movimento_previsto"].isin(movimentos)
        & df["Resultado"].isin(resultados)
    )
    if periodo is not None:
        inicio, fim = periodo
        datas = df["data_hora"].dt.date
        mascara &= ((datas >= inicio) & (datas <= fim)) | df["data_hora"].isna()
    return df[mascara].copy()


def resumo(df: pd.DataFrame) -> dict:
    com_referencia = df[df["Resultado"] != SEM_REFERENCIA]
    return {
        "total": int(len(df)),
        "amostras_teste": int((df["Origem"] == ROTULOS_ORIGEM[banco.ORIGEM_AMOSTRA_TESTE]).sum()),
        "digitados": int((df["Origem"] == ROTULOS_ORIGEM[banco.ORIGEM_MANUAL]).sum()),
        "sem_origem": int((df["Origem"] == ROTULO_SEM_ORIGEM).sum()),
        "com_referencia": int(len(com_referencia)),
        "acertos": int((com_referencia["Resultado"] == ACERTOU).sum()),
    }


def tabela_para_exibir(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas na ordem e com os nomes mostrados ao usuário."""
    saida = df.rename(
        columns={
            **_RENOMEAR,
            "data_hora": "Data/hora",
            "movimento_previsto": "Previsto",
            "movimento_real": "Movimento real",
        }
    )
    colunas = [
        "id", "Data/hora", "Origem", "Previsto", "Movimento real", "Resultado",
        *config.COLUNAS_SENSORES,
    ]
    return saida[colunas]
