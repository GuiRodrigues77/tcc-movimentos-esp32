import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

# Configuração da página do sistema
st.set_page_config(
    page_title="TCC - Classificação de Movimentos",
    page_icon="🏃‍♂️",
    layout="wide"
)

# Carregar o modelo de Machine Learning salvo anteriormente
@st.cache_resource
def carregar_modelo():
    return joblib.load('modelo_classificador_movimentos.pkl')

modelo = carregar_modelo()

# Configuração do Banco de Dados SQLite (Backend local)
def init_db():
    conn = sqlite3.connect('historico_movimentos.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predicoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            acc_x REAL,
            acc_y REAL,
            acc_z REAL,
            gyro_x REAL,
            gyro_y REAL,
            gyro_z REAL,
            movimento_previsto TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def salvar_no_banco(dados, movimento):
    conn = sqlite3.connect('historico_movimentos.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predicoes (data_hora, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, movimento_previsto)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        dados['ACC_X'], dados['ACC_Y'], dados['ACC_Z'],
        dados['GYRO_X'], dados['GYRO_Y'], dados['GYRO_Z'],
        movimento
    ))
    conn.commit()
    conn.close()

# Interface Visual do Protótipo (Telas)
st.title("🏃‍♂️ Sistema Inteligente de Classificação de Movimentos")
st.markdown("**Protótipo Funcional - TCC (Sensores Inerciais + Machine Learning)**")

# Abas do sistema
aba1, aba2 = st.tabs(["📊 Classificador em Tempo Real", "🗄️ Banco de Dados (Histórico)"])

with aba1:
    st.subheader("Simulação de Leitura dos Sensores (ESP32 + MPU6050)")
    st.write("Ajuste os valores dos eixos abaixo ou utilize os botões de exemplo para testar a classificação em tempo real:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Acelerômetro")
        acc_x = st.number_input("ACC_X", value=1500.0)
        acc_y = st.number_input("ACC_Y", value=-3200.0)
        acc_z = st.number_input("ACC_Z", value=18000.0)

    with col2:
        st.markdown("### Giroscópio")
        gyro_x = st.number_input("GYRO_X", value=-200.0)
        gyro_y = st.number_input("GYRO_Y", value=70.0)
        gyro_z = st.number_input("GYRO_Z", value=150.0)

    # Botão para realizar a predição
    if st.button("Classificar Movimento", type="primary"):
        # Montar o DataFrame com os dados inseridos
        entrada = pd.DataFrame([{
            'ACC_X': acc_x,
            'ACC_Y': acc_y,
            'ACC_Z': acc_z,
            'GYRO_X': gyro_x,
            'GYRO_Y': gyro_y,
            'GYRO_Z': gyro_z
        }])

        # Predição com o Random Forest
        resultado = modelo.predict(entrada)[0]
        
        # Salvar automaticamente no banco de dados SQLite
        salvar_no_banco(entrada.iloc[0], resultado)

        # Exibir resultado na tela com destaque
        st.success(f"### Movimento Identificado pelo Modelo: **{resultado}**")
        st.balloons()

with aba2:
    st.subheader("Histórico de Classificações Armazenadas no Banco de Dados")
    st.write("Estes dados foram salvos no banco SQLite local a cada predição realizada pelo sistema.")

    # Botão para atualizar dados da tabela
    if st.button("Atualizar Histórico"):
        st.rerun()

    # Ler dados do banco
    conn = sqlite3.connect('historico_movimentos.db')
    df_historico = pd.read_sql_query("SELECT * FROM predicoes ORDER BY id DESC", conn)
    conn.close()

    if not df_historico.empty:
        st.dataframe(df_historico, use_container_width=True)
    else:
        st.info("Nenhum registro encontrado no banco de dados ainda. Faça uma classificação na aba anterior!")