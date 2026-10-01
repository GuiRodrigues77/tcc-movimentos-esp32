import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Carregar os dados
df = pd.read_csv('dataset_movimentos_unificado.csv')
X = df[['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']]
y = df['Movimento_Classificado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Treinar o modelo com todos os dados de treino
print("Treinando o modelo definitivo...")
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# 3. Salvar o modelo treinado em um arquivo
arquivo_modelo = 'modelo_classificador_movimentos.pkl'
joblib.dump(modelo_rf, arquivo_modelo)

print(f"\nSucesso! Modelo salvo com o nome: '{arquivo_modelo}'")
print("Este arquivo será utilizado pelo sistema web do protótipo.")