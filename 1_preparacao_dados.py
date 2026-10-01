import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carregar o dataset
print("Carregando o dataset...")
df = pd.read_csv('dataset_movimentos_unificado.csv')

# 2. Separar as características (X) e o alvo (y)
# X = Dados dos sensores (Acelerômetro e Giroscópio)
# y = O que queremos prever (Andar, Correr, Sentar, Levantar)
colunas_sensores = ['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']
X = df[colunas_sensores]
y = df['Movimento_Classificado']

# 3. Dividir em Treinamento (80%) e Teste (20%)
# O parâmetro stratify=y garante que todas as classes fiquem balanceadas nas divisões
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Exibir o resultado para conferência
print("\n--- Separação Concluída ---")
print(f"Total de amostras originais: {len(df)}")
print(f"Amostras para TREINAMENTO do modelo: {len(X_train)}")
print(f"Amostras para TESTE do modelo: {len(X_test)}")

print("\nDistribuição dos movimentos no Treinamento:")
print(y_train.value_counts())