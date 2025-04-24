import requests
import streamlit as st

# URL da API do modelo de classificação zero-shot
API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

# Função auxiliar para obter o token do arquivo .streamlit/secrets.toml
def get_token():
    try:
        return st.secrets["huggingface"]["token"]
    except Exception:
        st.error("❌ Token da Hugging Face não encontrado. Verifique o arquivo `.streamlit/secrets.toml`.")
        return None

# Classificação individual com cache para performance
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
            st.warning(f"⚠️ Erro da API Hugging Face: {response.status_code}")
            
    except Exception as e:
        st.warning(f"⚠️ Erro ao classificar gasto: {str(e)}")
    
    return "Outros"

# Classificação em lote com barra de progresso (para uso em vários gastos de uma vez)
def classificar_lote(descricoes, progresso=None):
    resultados = []
    total = len(descricoes)
    
    for i, desc in enumerate(descricoes):
        cat = classificar_gasto(desc)
        resultados.append(cat)
        if progresso:
            progresso.progress((i + 1) / total)
    
    return resultados
