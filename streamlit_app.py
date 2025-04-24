import os

# Criar estrutura de diretórios
project_name = "gastos-dashboard-ia-v2"
os.makedirs(f"{project_name}/utils", exist_ok=True)
os.makedirs(f"{project_name}/data", exist_ok=True)
os.makedirs(f"{project_name}/.streamlit", exist_ok=True)

# Arquivo secrets.toml para gerenciar tokens de forma segura
secrets_toml = '''
[huggingface]
token = "hf_seu_token_real_aqui"
'''

with open(f"{project_name}/.streamlit/secrets.toml", "w") as f:
    f.write(secrets_toml)

# Arquivo .gitignore para não vazar dados sensíveis
gitignore = '''
.streamlit/secrets.toml
__pycache__/
*.py[cod]
*$py.class
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.DS_Store
data/*.csv
'''

with open(f"{project_name}/.gitignore", "w") as f:
    f.write(gitignore)

# utils/classifier.py - Classificador com cache e gerenciamento de erros
classifier_code = '''import requests
import json
import os
import streamlit as st
from datetime import datetime

# Usar o token do Streamlit Secrets
def get_token():
    try:
        return st.secrets["huggingface"]["token"]
    except Exception:
        st.error("Token de autenticação não encontrado. Verifique o arquivo .streamlit/secrets.toml")
        return None

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

# Cache para evitar chamadas repetidas à API
@st.cache_data(ttl=3600)  # Cache válido por 1 hora
def classificar_gasto(descricao):
    if not descricao or len(descricao.strip()) == 0:
        return "Outros"
    
    token = get_token()
    if not token:
        return "Outros"
        
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    CATEGORIAS = [
        "Alimentação", "Transporte", "Lazer", 
        "Moradia", "Saúde", "Educação", "Outros"
    ]
    
    try:
        payload = {
            "inputs": descricao,
            "parameters": {
                "candidate_labels": CATEGORIAS
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if "labels" in result and len(result["labels"]) > 0:
                return result["labels"][0]
        else:
            st.warning(f"Erro na API: {response.status_code}")
            
    except Exception as e:
        st.warning(f"Erro ao classificar: {str(e)}")
    
    return "Outros"

# Função para classificar em lote
def classificar_lote(descricoes, progresso=None):
    resultados = []
    total = len(descricoes)
    
    for i, desc in enumerate(descricoes):
        cat = classificar_gasto(desc)
        resultados.append(cat)
        if progresso:
            progresso.progress((i+1)/total)
    
    return resultados
'''

# utils/data_manager.py - Gerenciamento de dados
data_manager_code = '''import pandas as pd
import os
import json
from datetime import datetime
import streamlit as st

# Pasta para armazenar dados
DATA_DIR = "data"
CATEGORIAS = ["Alimentação", "Transporte", "Lazer", "Moradia", "Saúde", "Educação", "Outros"]

def carregar_dados():
    """Carrega dados do CSV ou cria um DataFrame vazio se não existir"""
    try:
        filepath = os.path.join(DATA_DIR, "gastos.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Converter data para datetime
            df['data'] = pd.to_datetime(df['data']).dt.date
            return df
        else:
            return pd.DataFrame(columns=["data", "descricao", "valor", "categoria"])
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame(columns=["data", "descricao", "valor", "categoria"])

def salvar_dados(df):
    """Salva os dados em CSV"""
    try:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        filepath = os.path.join(DATA_DIR, "gastos.csv")
        df.to_csv(filepath, index=False)
        return True
    except Exception as e:
        st.error(f"Erro ao salvar dados: {str(e)}")
        return False

def validar_csv(df):
    """Valida e formata o DataFrame carregado de CSV"""
    colunas_necessarias = ["data", "descricao", "valor"]
    
    # Verificar colunas necessárias
    for col in colunas_necessarias:
        if col not in df.columns:
            st.error(f"Coluna '{col}' não encontrada no CSV")
            return None
    
    # Manter apenas as colunas necessárias e adicionar categoria se não existir
    df = df[colunas_necessarias]
    if "categoria" not in df.columns:
        df["categoria"] = "Não classificado"
    
    # Converter tipos
    try:
        df['data'] = pd.to_datetime(df['data']).dt.date
        df['valor'] = pd.to_numeric(df['valor'])
    except Exception as e:
        st.error(f"Erro ao converter tipos de dados: {str(e)}")
        return None
    
    return df

def adicionar_gasto(df, data, descricao, valor, categoria="Não classificado"):
    """Adiciona um novo gasto ao DataFrame e salva"""
    novo_gasto = pd.DataFrame({
        "data": [data],
        "descricao": [descricao],
        "valor": [valor],
        "categoria": [categoria]
    })
    df_atualizado = pd.concat([df, novo_gasto], ignore_index=True)
    salvar_dados(df_atualizado)
    return df_atualizado

def atualizar_categoria(df, indice, nova_categoria):
    """Atualiza a categoria de um gasto específico"""
    if indice >= 0 and indice < len(df):
        df.at[indice, "categoria"] = nova_categoria
        salvar_dados(df)
    return df

def filtrar_dados(df, categoria=None, data_inicio=None, data_fim=None):
    """Filtra os dados por categoria e/ou intervalo de datas"""
    df_filtrado = df.copy()
    
    if categoria and categoria != "Todas":
        df_filtrado = df_filtrado[df_filtrado["categoria"] == categoria]
    
    if data_inicio:
        df_filtrado = df_filtrado[df_filtrado["data"] >= data_inicio]
    
    if data_fim:
        df_filtrado = df_filtrado[df_filtrado["data"] <= data_fim]
    
    return df_filtrado

def calcular_estatisticas(df):
    """Calcula estatísticas dos gastos"""
    if df.empty:
        return {}
    
    estatisticas = {
        "total_gastos": df[df["valor"] < 0]["valor"].sum() * -1,
        "total_receitas": df[df["valor"] > 0]["valor"].sum(),
        "saldo": df["valor"].sum(),
        "media_mensal": 0,  # Será calculado abaixo
        "categoria_maior_gasto": "",
        "valor_maior_gasto": 0
    }
    
    # Calcular média mensal se houver dados suficientes
    if not df.empty:
        df_temp = df.copy()
        df_temp['ano_mes'] = pd.to_datetime(df_temp['data']).dt.to_period('M')
        meses_unicos = df_temp['ano_mes'].nunique()
        if meses_unicos > 0:
            estatisticas["media_mensal"] = estatisticas["total_gastos"] / meses_unicos
    
    # Categoria com maior gasto
    if not df.empty:
        gastos_por_categoria = df[df["valor"] < 0].groupby("categoria")["valor"].sum() * -1
        if not gastos_por_categoria.empty:
            categoria = gastos_por_categoria.idxmax()
            valor = gastos_por_categoria.max()
            estatisticas["categoria_maior_gasto"] = categoria
            estatisticas["valor_maior_gasto"] = valor
    
    return estatisticas
'''

# streamlit_app.py - Aplicação completa com todas as melhorias
streamlit_app_code = '''import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import altair as alt
import os
import io
from utils.classifier import classificar_gasto, classificar_lote
from utils.data_manager import (
    carregar_dados, salvar_dados, validar_csv, adicionar_gasto,
    atualizar_categoria, filtrar_dados, calcular_estatisticas, CATEGORIAS
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
'''

# README.md com instruções completas
readme_md = '''# 💸 Dashboard de Gastos com IA (Hugging Face)

Dashboard interativo em Streamlit com classificação automática de gastos via modelo `facebook/bart-large-mnli` da Hugging Face.

## Funcionalidades

- Upload de gastos via CSV
- Entrada manual de gastos
- Classificação automática de categorias por IA
- Edição manual de categorias
- Persistência dos dados entre sessões
- Filtros por categoria e período
- Visualizações gráficas:
  - Gráfico de pizza para distribuição de gastos
  - Gráfico de linha para tendências ao longo do tempo
- Estatísticas de gastos, receitas e saldo
- Exportação dos dados classificados

## Configuração

### 1. Configurar o token da Hugging Face

Você precisa ter uma conta na Hugging Face e criar um token de acesso:

1. Crie uma conta em [huggingface.co](https://huggingface.co)
2. Vá para Settings > Access Tokens
3. Crie um novo token com escopo "read"
4. Adicione seu token no arquivo `.streamlit/secrets.toml`:

```toml
[huggingface]
token = "hf_seu_token_aqui"
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar a aplicação

```bash
streamlit run streamlit_app.py
```

## Formato do CSV

O arquivo CSV deve conter pelo menos as seguintes colunas:
- `data`: Data da transação (formato YYYY-MM-DD)
- `descricao`: Descrição da transação
- `valor`: Valor da transação (negativo para gastos, positivo para receitas)

Exemplo:
```
data,descricao,valor
2023-04-01,Supermercado,-150.50
2023-04-03,Salário,3000.00
2023-04-05,Combustível,-100.00
```

## Melhorias futuras

- Autenticação de usuários
- Suporte a múltiplas moedas
- Previsão de gastos futuros
- Orçamento e metas
- Importação direta de extratos bancários
'''

# requirements.txt atualizado
requirements_txt = '''streamlit>=1.30.0
pandas>=2.0.0
matplotlib>=3.7.0
requests>=2.28.0
numpy>=1.24.0
altair>=5.0.0
python-dotenv>=1.0.0
'''

# Escrever todos os arquivos
with open(f"{project_name}/utils/classifier.py", "w") as f:
    f.write(classifier_code)

with open(f"{project_name}/utils/data_manager.py", "w") as f:
    f.write(data_manager_code)

with open(f"{project_name}/streamlit_app.py", "w") as f:
    f.write(streamlit_app_code)

with open(f"{project_name}/requirements.txt", "w") as f:
    f.write(requirements_txt)

with open(f"{project_name}/README.md", "w") as f:
    f.write(readme_md)

print("Projeto criado com sucesso!")
