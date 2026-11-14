"""Gráficos Altair (Altair já vem instalado junto com o Streamlit)."""
import altair as alt
import pandas as pd

from . import config

PALETA = ["#2563EB", "#F97316", "#10B981", "#8B5CF6", "#EF4444", "#14B8A6"]


def _cor(classes: list[str], legenda: bool = True) -> alt.Color:
    cores = [PALETA[i % len(PALETA)] for i in range(len(classes))]
    return alt.Color(
        f"{config.COLUNA_ALVO}:N",
        scale=alt.Scale(domain=list(classes), range=cores),
        title="Movimento",
        legend=alt.Legend() if legenda else None,
    )


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Só as colunas usadas, com o movimento como texto simples."""
    saida = df[config.COLUNAS_SENSORES + [config.COLUNA_ALVO]].copy()
    saida[config.COLUNA_ALVO] = saida[config.COLUNA_ALVO].astype(str)
    return saida


def histograma(df: pd.DataFrame, coluna: str, classes: list[str]) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(opacity=0.65)
        .encode(
            x=alt.X(f"{coluna}:Q", bin=alt.Bin(maxbins=40), title=coluna),
            y=alt.Y("count()", stack=None, title="Registros"),
            color=_cor(classes),
        )
        .properties(height=320)
    )


def boxplot(df: pd.DataFrame, coluna: str, classes: list[str]) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_boxplot()
        .encode(
            x=alt.X(f"{config.COLUNA_ALVO}:N", sort=list(classes), title=None),
            y=alt.Y(f"{coluna}:Q", title=coluna),
            color=_cor(classes, legenda=False),
        )
        .properties(height=280, title=coluna)
    )


def dispersao(df: pd.DataFrame, x: str, y: str, classes: list[str]) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_circle(size=40, opacity=0.6)
        .encode(
            x=alt.X(f"{x}:Q", scale=alt.Scale(zero=False), title=x),
            y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False), title=y),
            color=_cor(classes),
            tooltip=[config.COLUNA_ALVO, x, y],
        )
        .properties(height=420)
        .interactive()
    )


def mapa_correlacao(corr: pd.DataFrame) -> alt.Chart:
    colunas = list(corr.columns)
    longo = (
        corr.reset_index(names="Eixo_X")
        .melt(id_vars="Eixo_X", var_name="Eixo_Y", value_name="Correlacao")
        .dropna()
    )
    base = alt.Chart(longo).encode(
        x=alt.X("Eixo_X:N", sort=colunas, title=None),
        y=alt.Y("Eixo_Y:N", sort=colunas, title=None),
    )
    celulas = base.mark_rect().encode(
        color=alt.Color(
            "Correlacao:Q",
            scale=alt.Scale(domain=[-1, 1], scheme="blueorange"),
            title="Correlação",
        )
    )
    textos = base.mark_text().encode(
        text=alt.Text("Correlacao:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.Correlacao) > 0.6", alt.value("white"), alt.value("black")
        ),
    )
    return (celulas + textos).properties(height=360)


def comparacao_modelos(tabela: pd.DataFrame) -> alt.Chart:
    """Barras agrupadas. `tabela`: colunas Modelo, Métrica, Valor (em %)."""
    base = alt.Chart(tabela).encode(
        x=alt.X("Modelo:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="Métrica:N",
        y=alt.Y("Valor:Q", title="%", scale=alt.Scale(domain=[0, 100])),
    )
    barras = base.mark_bar().encode(color=alt.Color("Métrica:N", title=None))
    textos = base.mark_text(dy=-6, fontSize=11).encode(text=alt.Text("Valor:Q", format=".1f"))
    return (barras + textos).properties(height=340)


def matriz_confusao(matriz: pd.DataFrame, em_percentual: bool = False) -> alt.Chart:
    classes = list(matriz.index)
    longo = matriz.reset_index(names="Real").melt(
        id_vars="Real", var_name="Previsto", value_name="Quantidade"
    )
    total_linha = longo.groupby("Real")["Quantidade"].transform("sum").replace(0, 1)
    if em_percentual:
        longo["Valor"] = (longo["Quantidade"] / total_linha * 100).round(1)
        longo["Rotulo"] = longo["Valor"].map(lambda v: f"{v:.0f}%")
    else:
        longo["Valor"] = longo["Quantidade"]
        longo["Rotulo"] = longo["Quantidade"].astype(str)
    limite = float(longo["Valor"].max()) / 2

    base = alt.Chart(longo).encode(
        x=alt.X("Previsto:N", sort=classes, title="Movimento previsto pelo modelo",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Real:N", sort=classes, title="Movimento real"),
    )
    celulas = base.mark_rect().encode(
        color=alt.Color("Valor:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["Real", "Previsto", "Quantidade"],
    )
    textos = base.mark_text(fontSize=16).encode(
        text="Rotulo:N",
        color=alt.condition(f"datum.Valor > {limite}", alt.value("white"), alt.value("black")),
    )
    return (celulas + textos).properties(height=380)
