
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Carregar o dataset limpo
df = pd.read_csv('data/dataset_movimentos_limpo.csv')

# 2. Separar características e classificações
colunas_sensores = [
    'ACC_X', 'ACC_Y', 'ACC_Z',
    'GYRO_X', 'GYRO_Y', 'GYRO_Z'
]

X = df[colunas_sensores]
y = df['Movimento_Classificado']

# 3. Separar os dados para manter a mesma configuração da avaliação
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Treinar o modelo Random Forest
print("Treinando o modelo definitivo...")

modelo_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modelo_rf.fit(X_train, y_train)

# 5. Salvar o modelo treinado
arquivo_modelo = 'modelo_classificador_movimentos.pkl'

joblib.dump(modelo_rf, arquivo_modelo)

print(f"\nSucesso! Modelo salvo: '{arquivo_modelo}'")
print("O modelo será utilizado pelo sistema web do TCC.")
