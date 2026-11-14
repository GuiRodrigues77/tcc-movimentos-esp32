import streamlit as st

from utils import banco, historico, ui

banco.init_db()

st.title("🗄️ Histórico")
st.caption(
    "Classificações realizadas no aplicativo e salvas no banco SQLite local. "
    "Os dados do DataSet ficam no CSV e não são misturados a este histórico."
)

df_dataset = ui.exigir_dataset()
_, y_teste = ui.exigir_conjunto_teste()

if st.button("Atualizar histórico"):
    st.rerun()

bruto = banco.ler_historico()
if bruto.empty:
    st.info(
        "Nenhum registro encontrado no banco de dados ainda. "
        "Faça uma classificação na página Classificação!"
    )
    st.stop()

dados_hist = historico.preparar(bruto)
r = historico.resumo(dados_hist)

# --- Resumo por origem ------------------------------------------------------
cartoes = [
    ("Registros do DataSet", ui.fmt_int(len(df_dataset)),
     f"{ui.fmt_int(len(df_dataset) - len(y_teste))} treino · {ui.fmt_int(len(y_teste))} teste"),
    ("Classificações no app", ui.fmt_int(r["total"]),
     f"{r['digitados']} digitadas · {r['sem_origem']} sem origem registrada"),
    ("Amostras de teste usadas", ui.fmt_int(r["amostras_teste"]), "sorteadas do DataSet"),
    ("Acertos com referência", f"{r['acertos']} de {r['com_referencia']}",
     "só onde há movimento real"),
]
for coluna, (rotulo, valor, legenda) in zip(st.columns(4), cartoes):
    with coluna, st.container(border=True):
        st.metric(rotulo, valor)
        st.caption(legenda)

# --- Filtros ----------------------------------------------------------------
with st.container(border=True):
    f1, f2, f3 = st.columns(3)
    opcoes_origem = sorted(dados_hist["Origem"].unique())
    origens = f1.multiselect("Origem", opcoes_origem, default=opcoes_origem)
    opcoes_mov = sorted(dados_hist["movimento_previsto"].dropna().unique())
    movimentos = f2.multiselect("Movimento previsto", opcoes_mov, default=opcoes_mov)
    opcoes_res = [historico.ACERTOU, historico.ERROU, historico.SEM_REFERENCIA]
    resultados = f3.multiselect("Resultado", opcoes_res, default=opcoes_res)

    periodo = None
    datas = dados_hist["data_hora"].dropna()
    if not datas.empty:
        d_min, d_max = datas.min().date(), datas.max().date()
        escolha = st.date_input("Período", value=(d_min, d_max), min_value=d_min, max_value=d_max)
        if isinstance(escolha, tuple) and len(escolha) == 2:
            periodo = escolha

filtrado = historico.filtrar(dados_hist, origens, movimentos, resultados, periodo)

st.markdown(f"**{len(filtrado)} registro(s)** no filtro")
if filtrado.empty:
    st.warning("Nenhum registro atende aos filtros escolhidos.")
    st.stop()

aba_tabela, aba_grafico = st.tabs(["Tabela", "Resumo gráfico"])
with aba_tabela:
    st.dataframe(historico.tabela_para_exibir(filtrado), hide_index=True)
    st.caption(
        "Resultado só existe para amostras sorteadas do DataSet, que têm movimento real "
        "para comparar. Registros digitados ficam como \"Sem referência\"."
    )
    st.download_button(
        "Baixar registros filtrados (CSV)",
        data=historico.tabela_para_exibir(filtrado).to_csv(index=False).encode("utf-8"),
        file_name="historico_filtrado.csv",
        mime="text/csv",
    )
with aba_grafico:
    st.markdown("**Classificações por movimento previsto**")
    st.bar_chart(filtrado["movimento_previsto"].value_counts())
    st.markdown("**Classificações por origem**")
    st.bar_chart(filtrado["Origem"].value_counts())
