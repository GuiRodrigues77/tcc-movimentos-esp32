# Classificação de Movimentos Humanos utilizando Sensores Inerciais e Aprendizado de Máquina

Trabalho de Conclusão de Curso de Engenharia de Computação (Unilins, 2026).

O projeto classifica quatro movimentos humanos (**Andar, Correr, Sentar e Levantar**) a partir de leituras de acelerômetro e giroscópio. Os dados foram coletados com um sensor inercial **MPU6050** ligado a um **ESP32**, e a classificação é feita por modelos de Aprendizado de Máquina (Random Forest, KNN e SVM). Uma aplicação web em **Streamlit** permite explorar o DataSet, avaliar os modelos, testar classificações e consultar o histórico.

## O que o projeto é (e o que não é)

- O treinamento, a avaliação e a classificação usam o DataSet `dataset_movimentos_unificado.csv` (2.000 amostras, 500 por movimento, coletadas pelo próprio autor).
- **Não há leitura em tempo real do ESP32 na aplicação.** Os valores dos sensores vêm de amostras do DataSet ou são digitados manualmente.
- Cada linha do DataSet é uma leitura isolada dos seis eixos, sem janelas temporais.

## Aplicação

| Página | O que faz |
|---|---|
| **Visão geral** | Indicadores do DataSet, distribuição dos movimentos, informações do modelo, acurácia e F1 no conjunto de teste e importância dos atributos. |
| **Dados** | Filtros por movimento e por faixa de cada sensor; histogramas, boxplots, dispersão entre eixos, estatísticas e correlação; download do CSV filtrado. |
| **Modelo e métricas** | Comparação entre Random Forest, KNN e SVM; métricas por movimento; matriz de confusão (contagem ou %); principais confusões. |
| **Classificação** | Sorteia uma amostra real do conjunto de teste (ou aceita valores digitados), classifica, compara com o movimento real, mostra os votos das árvores e avisa quando um valor está fora da faixa do DataSet. |
| **Histórico** | Consulta as classificações salvas no SQLite, com filtros por origem, movimento, resultado e período, e download em CSV. |

Todas as métricas são calculadas a partir do DataSet e do modelo, no mesmo split dos scripts de treino (80% treino, 20% teste, estratificado, `random_state=42`).

## Como executar

Requisitos: Python 3 e `pip`. O projeto foi desenvolvido com Python 3.14.3 e scikit-learn 1.9.0.

```bash
# 1. (opcional) criar e ativar um ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. instalar as dependências
pip install -r requirements.txt

# 3. iniciar a aplicação
streamlit run app_tcc.py
```

A aplicação abre no navegador (normalmente em `http://localhost:8501`). Para abrir no celular, use o endereço de rede que o Streamlit mostra no terminal (o celular precisa estar na mesma rede).

> O arquivo `modelo_classificador_movimentos.pkl` só carrega com a mesma versão do scikit-learn usada para gerá-lo. Se aparecer erro de versão, gere o modelo novamente (veja abaixo).

## Scripts de treinamento

Rode na raiz do projeto, nesta ordem:

| Script | Função |
|---|---|
| `1_preparacao_dados.py` | Lê o DataSet e faz a divisão treino/teste, mostrando a conferência. |
| `2_treinamento_rf.py` | Treina o Random Forest, imprime as métricas e salva `matriz_confusao.png`. |
| `3_salvar_modelo.py` | Treina o modelo definitivo e salva `modelo_classificador_movimentos.pkl`, usado pela aplicação. |
| `4_testar_knn_svm.py` | Avalia KNN e SVM com o mesmo split. |

O script 3 **sobrescreve** o `.pkl`. Os scripts 2 e 4 usam `matplotlib` e `seaborn`, que estão no `requirements.txt`.

## Estrutura do repositório

```
app_tcc.py                       # entrada da aplicação (navegação)
paginas/                         # páginas: visao_geral, dados, modelo, classificacao, historico
utils/                           # config, dados, modelo, graficos, banco, historico, ui
.streamlit/config.toml           # tema
1_preparacao_dados.py            # scripts de treinamento
2_treinamento_rf.py
3_salvar_modelo.py
4_testar_knn_svm.py
dataset_movimentos_unificado.csv # DataSet (6 sensores + Movimento_Classificado)
modelo_classificador_movimentos.pkl
historico_movimentos.db          # SQLite com as classificações feitas na aplicação
matriz_confusao.png              # gerada por 2_treinamento_rf.py
requirements.txt
```

## Banco de dados

O histórico fica em `historico_movimentos.db` (SQLite), na tabela `predicoes`. A tabela é criada, e atualizada se necessário, automaticamente ao abrir a aplicação. Cada registro guarda data/hora, os seis valores dos sensores, o movimento previsto, a **origem** (`amostra_teste` ou `manual`) e, quando existe, o **movimento real**.

## Limitações

- DataSet pequeno e próprio (2.000 amostras), com leituras isoladas e sem janelas temporais.
- A divisão treino/teste é aleatória por linha; o desempenho em outros usuários ou sessões de coleta não foi avaliado.
- Sem integração automática entre o ESP32 e a aplicação.
- O UCI HAR Dataset está previsto como etapa futura e não foi utilizado.

## Licença

MIT. Veja o arquivo `LICENSE`.
