"""Constantes e caminhos do projeto.

Todos os caminhos partem da raiz do repositório, então o app funciona
independentemente da pasta de onde o comando `streamlit run` é executado.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

CAMINHO_DATASET = RAIZ / "data/dataset_movimentos_limpo.csv"
CAMINHO_MODELO = RAIZ / "modelo_classificador_movimentos.pkl"
CAMINHO_BANCO = RAIZ / "historico_movimentos.db"

COLUNAS_ACC = ["ACC_X", "ACC_Y", "ACC_Z"]
COLUNAS_GYRO = ["GYRO_X", "GYRO_Y", "GYRO_Z"]
COLUNAS_SENSORES = COLUNAS_ACC + COLUNAS_GYRO
COLUNA_ALVO = "Movimento_Classificado"

# Ordem de exibição preferida (as classes reais vêm do DataSet).
MOVIMENTOS = ["Andar", "Correr", "Sentar", "Levantar"]

# Mesmos parâmetros de 1_preparacao_dados.py / 2_treinamento_rf.py /
# 3_salvar_modelo.py. Se mudar lá, mude aqui.
TAMANHO_TESTE = 0.2
SEMENTE = 42
