"""Funções puras (sem Streamlit) para descrever e avaliar o modelo salvo."""
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from . import config, dados


def carregar_modelo():
    return joblib.load(config.CAMINHO_MODELO)


def descrever(modelo) -> dict:
    """Informações lidas do próprio objeto do modelo (nada é digitado à mão)."""
    params = modelo.get_params() if hasattr(modelo, "get_params") else {}
    return {
        "tipo": type(modelo).__name__,
        "n_estimators": params.get("n_estimators"),
        "max_depth": params.get("max_depth"),
        "random_state": params.get("random_state"),
        "classes": [str(c) for c in getattr(modelo, "classes_", [])],
        "atributos": [str(a) for a in getattr(modelo, "feature_names_in_", [])],
    }


def importancias(modelo) -> pd.DataFrame | None:
    """Importância dos atributos (Gini), se o modelo a oferecer."""
    valores = getattr(modelo, "feature_importances_", None)
    if valores is None:
        return None
    nomes = list(getattr(modelo, "feature_names_in_", []))
    if len(nomes) != len(valores):
        if len(config.COLUNAS_SENSORES) != len(valores):
            return None
        nomes = config.COLUNAS_SENSORES
    return (
        pd.DataFrame({"Atributo": nomes, "Importância": [float(v) for v in valores]})
        .sort_values("Importância", ascending=False)
        .reset_index(drop=True)
    )


def avaliar_holdout(modelo, df: pd.DataFrame) -> dict:
    """Avalia o modelo no conjunto de teste (20 %), idêntico ao dos scripts."""
    _, X_test, _, y_test = dados.separar_treino_teste(df)
    y_pred = modelo.predict(X_test)
    return {
        "n_teste": int(len(y_test)),
        "acuracia": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "relatorio": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }


def treinar_alternativos(X_treino, y_treino) -> dict:
    """KNN e SVM com os mesmos parâmetros de 4_testar_knn_svm.py."""
    y = y_treino.astype(str)
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_treino, y),
        "SVM": SVC(random_state=config.SEMENTE).fit(X_treino, y),
    }


def avaliar_completo(modelo, X_teste, y_teste, classes: list[str]) -> dict:
    """Métricas gerais, por classe e matriz de confusão no conjunto de teste."""
    y_real = y_teste.astype(str)
    y_pred = pd.Series(modelo.predict(X_teste), index=y_real.index).astype(str)
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y_real, y_pred, labels=classes, zero_division=0
    )
    por_classe = pd.DataFrame(
        {"Precisão": precisao, "Recall": recall, "F1-score": f1, "Suporte": suporte},
        index=pd.Index(classes, name="Movimento"),
    )
    matriz = pd.DataFrame(
        confusion_matrix(y_real, y_pred, labels=classes),
        index=pd.Index(classes, name="Real"),
        columns=pd.Index(classes, name="Previsto"),
    )
    return {
        "n_teste": int(len(y_real)),
        "acuracia": float(accuracy_score(y_real, y_pred)),
        "precisao_macro": float(precisao.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "por_classe": por_classe,
        "matriz": matriz,
    }


def comparar_modelos(modelos: dict, X_teste, y_teste, classes: list[str]) -> dict:
    return {
        nome: avaliar_completo(m, X_teste, y_teste, classes) for nome, m in modelos.items()
    }


def principais_confusoes(matriz: pd.DataFrame, limite: int = 3) -> list[tuple[str, str, int]]:
    """Maiores erros (real -> previsto), ignorando a diagonal."""
    erros = [
        (real, previsto, int(matriz.loc[real, previsto]))
        for real in matriz.index
        for previsto in matriz.columns
        if real != previsto and matriz.loc[real, previsto] > 0
    ]
    return sorted(erros, key=lambda e: -e[2])[:limite]
