import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import altair as alt
import os
import io
from classifier import (
    classificar_gasto, classificar_lote, CATEGORIAS,
    carregar_dados, salvar_dados, validar_csv, adicionar_gasto,
    atualizar_categoria, filtrar_dados, calcular_estatisticas
)

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Gastos com IA", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e descrição
st.title("💸 Dashboard de Gastos com Classificação IA")
st.markdown("""
Este dashboard usa inteligência artificial para classificar automaticamente seus gastos.
Upload um arquivo CSV com suas transações ou adicione gastos manualmente.
""")

# Carregar dados existentes
df = carregar_dados()

# Barra lateral para filtros e ações
with st.sidebar:
    st.header("📋 Ações")
    
    # Seção de upload de arquivo
    st.subheader("📤 Upload de dados")
    uploaded_file = st.file_uploader("Envie seu CSV", type="csv")
    
    if uploaded_file:
        try:
            df_novo = pd.read_csv(uploaded_file)
            df_validado = validar_csv(df_novo)
            
            if df_validado is not None:
                if st.button("Importar dados"):
                    with st.spinner("Processando..."):
                        # Juntar com dados existentes
                        df_combinado = pd.concat([df, df_validado], ignore_index=True)
                        # Remover possíveis duplicatas
                        df_combinado = df_combinado.drop_duplicates(subset=["data", "descricao", "valor"])
                        # Salvar dados
                        salvar_dados(df_combinado)
                        st.success(f"{len(df_validado)} transações importadas!")
                        # Recarregar dados
                        df = carregar_dados()
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
    
    # Filtros
    st.subheader("🔍 Filtros")
    filtro_categoria = st.selectbox("Categoria:", ["Todas"] + CATEGORIAS)
    
    col1, col2 = st.columns(2)
    with col1:
        data_inicio = st.date_input("De:", value=None)
    with col2:
        data_fim = st.date_input("Até:", value=None)
        
    if st.button("Aplicar filtros"):
        df_filtrado = filtrar_dados(df, filtro_categoria, data_inicio, data_fim)
    else:
        df_filtrado = df
    
    # Exportar dados
    st.subheader("📥 Exportar dados")
    if not df.empty:
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        st.download_button(
            label="Baixar CSV",
            data=buffer.getvalue(),
            file_name="meus_gastos.csv",
            mime="text/csv"
        )
    
    # Classificar todos
    st.subheader("🤖 Classificar todos")
    if not df.empty and "Não classificado" in df["categoria"].values:
        if st.button("Classificar gastos não categorizados"):
            with st.spinner("Classificando todos os gastos..."):
                # Filtrar registros não classificados
                indices_nao_classificados = df[df["categoria"] == "Não classificado"].index
                descricoes_nao_classificadas = df.loc[indices_nao_classificados, "descricao"].tolist()
                
                # Barra de progresso
                progresso = st.progress(0)
                categorias = classificar_lote(descricoes_nao_classificadas, progresso)
                
                # Atualizar categorias
                for i, idx in enumerate(indices_nao_classificados):
                    df.at[idx, "categoria"] = categorias[i]
                
                salvar_dados(df)
                st.success(f"{len(indices_nao_classificados)} gastos classificados!")

# Criar layout principal com duas colunas
col_esq, col_dir = st.columns([3, 2])

# Coluna da esquerda: gráficos e estatísticas
with col_esq:
    st.header("📊 Análise de Gastos")
    
    if not df_filtrado.empty:
        # Calcular estatísticas
        estatisticas = calcular_estatisticas(df_filtrado)
        
        # Exibir cards de estatísticas
        metricas = st.columns(3)
        with metricas[0]:
            st.metric("Total de Gastos", f"R$ {estatisticas['total_gastos']:.2f}")
        with metricas[1]:
            st.metric("Total de Receitas", f"R$ {estatisticas['total_receitas']:.2f}")
        with metricas[2]:
            valor_saldo = estatisticas['saldo']
            delta_cor = "normal" if valor_saldo >= 0 else "inverse"
            st.metric("Saldo", f"R$ {valor_saldo:.2f}", delta_color=delta_cor)
        
        # Gráfico de pizza para distribuição de categorias
        st.subheader("Distribuição por Categoria")
        df_gastos = df_filtrado[df_filtrado["valor"] < 0].copy()
        if not df_gastos.empty:
            df_gastos["valor_abs"] = df_gastos["valor"].abs()
            resumo = df_gastos.groupby("categoria")["valor_abs"].sum()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            wedges, texts, autotexts = ax.pie(
                resumo, 
                labels=resumo.index, 
                autopct="%1.1f%%", 
                startangle=90,
                explode=[0.05] * len(resumo)
            )
            # Adicionar legenda e estilo
            plt.setp(autotexts, size=9, weight="bold")
            ax.axis("equal")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabela de resumo por categoria
            st.subheader("Resumo por Categoria")
            resumo_df = pd.DataFrame({
                "Categoria": resumo.index,
                "Valor (R$)": resumo.values,
                "Porcentagem": (resumo / resumo.sum() * 100).round(1).astype(str) + '%'
            })
            st.dataframe(resumo_df.sort_values("Valor (R$)", ascending=False), hide_index=True)
        
        # Gráfico de tendência de gastos ao longo do tempo
        st.subheader("Tendência de Gastos")
        if len(df_gastos) > 1:
            # Agrupar por mês
            df_gastos['ano_mes'] = pd.to_datetime(df_gastos['data']).dt.to_period('M').astype(str)
            gastos_mensais = df_gastos.groupby('ano_mes')['valor_abs'].sum().reset_index()
            
            # Criar gráfico com Altair
            chart = alt.Chart(gastos_mensais).mark_line(point=True).encode(
                x=alt.X('ano_mes:O', title='Mês'),
                y=alt.Y('valor_abs:Q', title='Total de Gastos (R$)'),
                tooltip=['ano_mes', 'valor_abs']
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Nenhum dado disponível para análise. Adicione gastos ou importe um CSV.")

# Coluna da direita: inserção de dados e tabela
with col_dir:
    st.header("✍️ Adicionar Gasto")
    with st.form(key="form_manual"):
        data = st.date_input("Data do gasto", value=datetime.now())
        descricao = st.text_input("Descrição")
        valor = st.number_input("Valor (negativo para gasto)", step=0.01, format="%.2f")
        categoria_manual = st.selectbox("Categoria", ["Auto-classificar"] + CATEGORIAS)
        
        submit = st.form_submit_button("Adicionar gasto")
    
    if submit and descricao:
        categoria_final = categoria_manual
        if categoria_manual == "Auto-classificar":
            with st.spinner("Classificando..."):
                categoria_final = classificar_gasto(descricao)
        
        df = adicionar_gasto(df, data, descricao, valor, categoria_final)
        st.success(f"Gasto adicionado na categoria: {categoria_final}")
    
    # Tabela de dados com opções de edição
    st.header("📋 Registros")
    if not df_filtrado.empty:
        st.dataframe(
            df_filtrado.sort_values("data", ascending=False),
            column_config={
                "data": st.column_config.DateColumn("Data"),
                "descricao": "Descrição",
                "valor": st.column_config.NumberColumn("Valor", format="R$ %.2f"),
                "categoria": "Categoria"
            },
            hide_index=True
        )
        
        # Editar categoria de um registro
        st.subheader("Editar Categoria")
        col1, col2 = st.columns(2)
        with col1:
            descricao_selecionada = st.selectbox(
                "Selecione a transação:",
                df_filtrado["descricao"].tolist()
            )
        
        if descricao_selecionada:
            indice = df[df["descricao"] == descricao_selecionada].index[0]
            categoria_atual = df.loc[indice, "categoria"]
            
            with col2:
                nova_categoria = st.selectbox(
                    "Nova categoria:",
                    CATEGORIAS,
                    index=CATEGORIAS.index(categoria_atual) if categoria_atual in CATEGORIAS else 0
                )
            
            if st.button("Atualizar categoria"):
                df = atualizar_categoria(df, indice, nova_categoria)
                st.success(f"Categoria atualizada para: {nova_categoria}")
    else:
        st.info("Nenhum registro para exibir.")

# Footer
st.markdown("---")
st.markdown("💡 **Dica:** Exporte seus dados regularmente para evitar perdas.")
#with open(f"{project_name}/README.md", "w") as f:
    f.write(readme_md)

print("Projeto criado com sucesso!")
