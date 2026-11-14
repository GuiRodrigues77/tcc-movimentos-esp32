
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Carregar e preparar os dados
df = pd.read_csv('data/dataset_movimentos_limpo.csv')

colunas_sensores = [
    'ACC_X', 'ACC_Y', 'ACC_Z',
    'GYRO_X', 'GYRO_Y', 'GYRO_Z'
]

X = df[colunas_sensores]
y = df['Movimento_Classificado']

# 2. Divisão de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Treinamento do modelo
print("Treinando o modelo Random Forest...")

modelo_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modelo_rf.fit(X_train, y_train)

# 4. Teste do modelo
print("Realizando previsões...")
y_pred = modelo_rf.predict(X_test)

# 5. Avaliação de desempenho
acuracia = accuracy_score(y_test, y_pred)

print(f"\nAcurácia do Random Forest: {acuracia * 100:.2f}%")

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

# 6. Gerar e salvar matriz de confusão
print("Gerando matriz de confusão...")

movimentos = ['Andar', 'Correr', 'Sentar', 'Levantar']

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=movimentos
)

os.makedirs('resultados', exist_ok=True)

plt.figure(figsize=(8, 6))

sns.heatmap(
    matriz,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=movimentos,
    yticklabels=movimentos
)

plt.title('Matriz de Confusão - Random Forest')
plt.ylabel('Movimento Real')
plt.xlabel('Movimento Previsto')
plt.tight_layout()

plt.savefig('resultados/matriz_confusao_rf.png', dpi=300)

print("Matriz salva em resultados/matriz_confusao_rf.png")

plt.show()
