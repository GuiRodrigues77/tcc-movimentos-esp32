
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carregar o dataset limpo
print("Carregando o dataset...")
df = pd.read_csv('data/dataset_movimentos_limpo.csv')

# 2. Separar as características (X) e o alvo (y)
colunas_sensores = [
    'ACC_X', 'ACC_Y', 'ACC_Z',
    'GYRO_X', 'GYRO_Y', 'GYRO_Z'
]

X = df[colunas_sensores]
y = df['Movimento_Classificado']

# 3. Dividir em Treinamento (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Exibir resultados
print("\n--- Separação Concluída ---")
print(f"Total de amostras originais: {len(df)}")
print(f"Amostras para TREINAMENTO: {len(X_train)}")
print(f"Amostras para TESTE: {len(X_test)}")

print("\nDistribuição dos movimentos no Treinamento:")
print(y_train.value_counts())

print("\nDistribuição dos movimentos no Teste:")
print(y_test.value_counts())
