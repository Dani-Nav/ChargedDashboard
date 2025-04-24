import requests
import json
import os
import streamlit as st
from datetime import datetime

# Categorias disponíveis para classificação
CATEGORIAS = ["Alimentação", "Transporte", "Lazer", "Moradia", "Saúde", "Educação", "Outros"]

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

# Funções de gerenciamento de dados (movidas do utils/data_manager.py)
DATA_DIR = "data"

def carregar_dados():
    """Carrega dados do CSV ou cria um DataFrame vazio se não existir"""
    try:
        import pandas as pd
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
    import pandas as pd
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
    import pandas as pd
    if df.empty:
        return {}
    
    estatisticas = {
        "total_gastos": df[df["valor"] < 0]["valor"].sum() * -1,
        "total_receitas": df[df["valor"] > 0]["valor"].sum(),
        "saldo": df["valor"].sum(),
        "media_mensal": 0,
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
