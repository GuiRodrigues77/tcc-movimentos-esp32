import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregar os dados
df = pd.read_csv('dataset_movimentos_unificado.csv')
X = df[['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']]
y = df['Movimento_Classificado']

# 2. Divisão de Treino e Teste (mesma semente para ser justo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("="*50)
print(" AVALIAÇÃO DO KNN (K-Nearest Neighbors)")
print("="*50)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

acuracia_knn = accuracy_score(y_test, y_pred_knn)
print(f"Acurácia do KNN: {acuracia_knn * 100:.2f}%\n")
print(classification_report(y_test, y_pred_knn))

print("="*50)
print(" AVALIAÇÃO DO SVM (Support Vector Machine)")
print("="*50)
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

acuracia_svm = accuracy_score(y_test, y_pred_svm)
print(f"Acurácia do SVM: {acuracia_svm * 100:.2f}%\n")
print(classification_report(y_test, y_pred_svm))