import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Carregar e Preparar (mesmo processo anterior)
df = pd.read_csv('dataset_movimentos_unificado.csv')
X = df[['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']]
y = df['Movimento_Classificado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Treinamento do Modelo
print("Treinando o modelo Random Forest...")
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# 3. Teste do Modelo
print("Realizando previsões com os dados de teste...")
y_pred = modelo_rf.predict(X_test)

# 4. Avaliação de Desempenho
acuracia = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do Modelo: {acuracia * 100:.2f}%")

print("\nRelatório de Classificação detalhado:")
print(classification_report(y_test, y_pred))

# 5. Gerar e Salvar Matriz de Confusão
print("Gerando Matriz de Confusão...")
movimentos = ['Andar', 'Correr', 'Sentar', 'Levantar']
matriz = confusion_matrix(y_test, y_pred, labels=movimentos)

plt.figure(figsize=(8, 6))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
            xticklabels=movimentos, yticklabels=movimentos)
plt.title('Matriz de Confusão - Sensores Inerciais')
plt.ylabel('Movimento Real executado')
plt.xlabel('Movimento Previsto pelo modelo')
plt.tight_layout()

# Salva a imagem na mesma pasta
plt.savefig('matriz_confusao.png')
print("Gráfico salvo como 'matriz_confusao.png' na sua pasta!")

# Exibe o gráfico na tela
plt.show()