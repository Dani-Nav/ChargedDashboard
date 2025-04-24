import os

# Criar estrutura de diretórios
project_name = "gastos-dashboard-ia-v2"
os.makedirs(f"{project_name}/utils", exist_ok=True)
os.makedirs(f"{project_name}/data", exist_ok=True)

# Novo classifier.py com API_URL personalizada
classifier_code = '''import requests

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
headers = {
    "Authorization": "Bearer hf_SEU_TOKEN_AQUI"
}

def classificar_gasto(descricao):
    payload = {
        "inputs": descricao,
        "parameters": {
            "candidate_labels": [
                "Alimentação", "Transporte", "Lazer", 
                "Moradia", "Saúde", "Educação", "Outros"
            ]
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    if "labels" in result:
        return result["labels"][0]
    return "Outros"
'''

# Novo streamlit_app.py
streamlit_app_code = '''import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.classifier import classificar_gasto

st.set_page_config(page_title="Dashboard de Gastos com IA", layout="centered")
st.title("💸 Dashboard de Gastos com Classificação IA (Hugging Face)")

st.markdown("Envie um arquivo CSV com seus gastos ou insira manualmente:")

uploaded_file = st.file_uploader("📤 Envie seu arquivo CSV", type="csv")

with st.expander("✍️ Inserir gasto manualmente"):
    with st.form(key="form_manual"):
        data = st.date_input("Data do gasto")
        descricao = st.text_input("Descrição")
        valor = st.number_input("Valor (negativo para gasto)", step=0.01, format="%.2f")
        submit = st.form_submit_button("Adicionar gasto")

    if submit and descricao:
        df_manual = pd.DataFrame([[data, descricao, valor]], columns=["data", "descricao", "valor"])
    else:
        df_manual = pd.DataFrame(columns=["data", "descricao", "valor"])

# Leitura do CSV (se enviado)
if uploaded_file:
    try:
        df_csv = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        df_csv = pd.DataFrame(columns=["data", "descricao", "valor"])
else:
    df_csv = pd.DataFrame(columns=["data", "descricao", "valor"])

# Junta tudo
df = pd.concat([df_csv, df_manual], ignore_index=True)

if not df.empty:
    st.subheader("📄 Tabela de Gastos")
    st.write(df)

    # Classificação com IA
    st.subheader("🤖 Classificação por IA (Hugging Face)")
    with st.spinner("Classificando..."):
        df["categoria"] = df["descricao"].apply(classificar_gasto)
    st.success("Classificação concluída!")
    st.write(df)

    # Gráfico de pizza
    st.subheader("📊 Distribuição dos Gastos (%)")
    df["valor_absoluto"] = df["valor"].abs()
    resumo = df.groupby("categoria")["valor_absoluto"].sum()
    percentual = resumo / resumo.sum()

    fig, ax = plt.subplots()
    ax.pie(percentual, labels=percentual.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)
else:
    st.info("Envie um arquivo ou insira dados manualmente para começar.")
'''

# requirements.txt
requirements_txt = '''streamlit
pandas
matplotlib
requests
'''

# README.md
readme_md = '''# 💸 Dashboard de Gastos com IA (Hugging Face)

Dashboard em Streamlit com classificação de gastos via modelo `facebook/bart-large-mnli` da Hugging Face.

## Funcionalidades
- Upload de gastos via CSV
- Entrada manual de gastos
- Classificação automática de categorias por IA
- Visualização em gráfico de pizza

## Executando localmente

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
