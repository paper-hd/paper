#!/usr/bin/env python
# coding: utf-8

# In[19]:


from pre_processing import PreProcessing # Disponível em https://github.com/antonio258/npp
import requests
import pandas as pd
import os
#import nltk
from pandarallel import pandarallel
import pandas as pd
from openai import AzureOpenAI


# In[4]:


def load_data_from_api(url: str, headers: dict = None) -> list:
    """Faz requisição à API e retorna a lista de dados se bem-sucedida."""
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        json_data = response.json()
        if json_data.get("status") and "data" in json_data:
            return json_data["data"]
        else:
            raise ValueError("Resposta da API inválida: status falso ou sem campo 'data'")
    else:
        raise ConnectionError(f"Erro ao acessar API: {response.status_code} - {response.text}")


# In[5]:


def load_data_from_csv(csv_path: str) -> list:
    """Carrega os dados de um CSV no mesmo formato da API."""
    df_raw = pd.read_csv(csv_path)
    return df_raw.to_dict(orient="records")


# In[6]:


def transform_data(data: list) -> pd.DataFrame:
    """Transforma a lista de dicionários no formato esperado em um DataFrame padronizado - Exemplo ilustrativo"""
    return pd.DataFrame([{
        "ClassN1": item.get("area", ""),
        "ID": item.get("id"),
        "Texto": f"{item.get('titulo', '')} {item.get('descricao', '')}".strip(),
        "CreateTime": item.get("dataCriacao", ""),
        "TotalTimeTracked": item.get("tempoGasto", 0)
    } for item in data])


# In[7]:


def gerar_stopwords_ibge(caminho_masculino: str, caminho_feminino: str, arquivo_saida: str = "stopwords.txt"):
    """
    Lê dois arquivos CSV com nomes ('ibge-mas-10000.csv' e 'ibge-fem-10000.csv'),
    extrai os nomes únicos e salva no arquivo stopwords.txt.
    Demais stopwords devem ser inseridas diretamente neste arquivo, por exemplo, nome de clientes ou outros que podem  existir no helpdesk
    """
    try:
        # Lê os arquivos CSV
        df_mas = pd.read_csv(caminho_masculino)
        df_fem = pd.read_csv(caminho_feminino)

        # Extrai nomes das colunas 'nome'
        nomes_mas = df_mas['nome'].dropna().astype(str).str.strip().str.lower()
        nomes_fem = df_fem['nome'].dropna().astype(str).str.strip().str.lower()

        # Une e remove duplicados
        nomes_unicos = sorted(set(nomes_mas).union(set(nomes_fem)))

        # Salva no arquivo de saída
        with open(arquivo_saida, "w", encoding="utf-8") as f:
            for nome in nomes_unicos:
                f.write(f"{nome}\n")

        print(f"Arquivo '{arquivo_saida}' criado com {len(nomes_unicos)} nomes.")
    except Exception as e:
        print(f"Erro ao gerar stopwords: {e}")


# In[8]:


def remover_assinaturas_llm(df: pd.DataFrame, endpoint_url: str, api_key: str = None) -> pd.DataFrame:
    """
    Envia os textos para uma LLM via API e remove assinaturas de email.

    Parâmetros:
    - df: DataFrame com coluna 'Texto'
    - endpoint_url: URL da API da LLM (endpoint do Llama)
    - api_key: chave de autenticação se necessário (Bearer Token ou chave personalizada)

    Retorna:
    - DataFrame com nova coluna 'TextoSemAssinatura' contendo a resposta da LLM
    """
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def gerar_prompt(texto):
        prompt = (
            "Extract complete email signature from text. Return only text off output, equal of\n"
            "input.\nAll words in lowercase.\n"
            "<input> Bom dia, pessoal!\r\n\r\nMeu outlook no desktop não está funcionando. "
            "Poderiam me ajudar? Só consigo acessar pelo celular.\r\n\r\nObrigada!\r\n\r\n\r\n"
            "Obter o Outlook para iOS<https://aka.ms/o0ukef>\n"
            "<output> meu outlook no desktop não está funcionando. poderiam me ajudar? só consigo acessar pelo celular.\n"
            f"<input> {texto}\n"
            "<output>"
        )
        return prompt

    respostas = []
    for texto in df['Texto']:
        prompt = gerar_prompt(texto)

        payload = {
            "prompt": prompt,
            "temperature": 0.2,
            "max_tokens": 512,
            "stop": ["<input>"]  # evita resposta com novo input
        }

        response = requests.post(endpoint_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            resposta_texto = result.get("response") or result.get("text") or result.get("choices", [{}])[0].get("text", "").strip()
            respostas.append(resposta_texto.strip())
        else:
            print(f"Erro ao consultar API: {response.status_code} - {response.text}")
            respostas.append("")

    df['TextoSemAssinatura'] = respostas
    return df


# In[9]:


def remover_assinaturas_azure(df: pd.DataFrame, endpoint: str, subscription_key: str, deployment: str, api_version: str = "2024-12-01-preview") -> pd.DataFrame:
    """
    Remove assinaturas de e-mail usando Azure OpenAI (modelo tipo LLM).

    Parâmetros:
    - df: DataFrame com coluna 'Texto'
    - endpoint: URL do endpoint Azure OpenAI
    - subscription_key: chave de API do Azure
    - deployment: nome do deployment do modelo (ex: 'gpt-4.1')
    - api_version: versão da API Azure OpenAI

    Retorna:
    - DataFrame com nova coluna 'TextoSemAssinatura'
    """
    client = AzureOpenAI(
        api_key=subscription_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

    def gerar_prompt(texto):
       return (
            "You are an assistant that extracts only the main message from email texts.\n\n"
            "Ignore greetings, sign-offs, and email signatures (e.g., “thank you”, "
            "“best regards”, names, mobile footers, etc.).\n\n"
            "Return only the core request, issue, or message content. Keep the formatting natural. "
            "All output must be in lowercase and in the same language as the input.\n\n"
            "Example:\n"
            "<input> Bom dia, pessoal!\n\n"
            "Meu outlook no desktop não está funcionando.\n"
            "Poderiam me ajudar? Só consigo acessar pelo celular.\n\n"
            "Obrigada!\n\n"
            "Obter o Outlook para iOS<https://aka.ms/o0ukef>\n"
            "<output> meu outlook no desktop não está funcionando. poderiam me ajudar? só consigo acessar pelo celular.\n\n"
            "Now process the following:\n"
            f"<input> {texto}\n"
            "<output>"
        )

    respostas = []
    for texto in df["Texto"]:
        prompt = gerar_prompt(texto)

        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=deployment,
                temperature=0.2,
                max_tokens=512,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            resposta_texto = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Erro na chamada Azure OpenAI: {e}")
            resposta_texto = texto

        respostas.append(resposta_texto)

    df["TextoSemAssinatura"] = respostas
    return df


# In[10]:


def processing_text(text):
    text = pp.lowercase_unidecode(text)
    text = pp.remove_html_tags(text)
    text = pp.remove_urls(text)
    text = pp.remove_punctuation(text)
    text = pp.remove_numbers(text, mode="replace")
    text = pp.remove_stopwords(text)
    text = pp.remove_n(text, 2)
    return text


# In[11]:


def main(source: str, is_api: bool = True, headers: dict = None) -> pd.DataFrame:
    """Função principal que escolhe a fonte (API ou CSV) e retorna o DataFrame processado."""
    if is_api:
        raw_data = load_data_from_api(source, headers=headers)
    else:
        raw_data = load_data_from_csv(source)

    df = transform_data(raw_data)
    return df


# In[13]:


# Exemplo de uso com API
# url = "https://sua-api.com/endpoint"
# headers = {"Authorization": "Bearer seu_token"}
# df_entrada = main(url, is_api=True, headers=headers)

# Exemplo de uso com CSV
# csv_path = "dados.csv"
# df_entrada = main(csv_path, is_api=False)

# print(df_entrada)

"""
Procedimentos de limpeza da base:
1. Enviar para a LLM Llama/Azure via API para que ele retorne apenas o assunto principal do texto
2. Importar nomes brasileiros e criar o arquivo de stopwords
3. Executar, de acordo com as particularidades da base, as limpezas da biblioteca de pré-processamento
"""

#endpoint = "https://api-llm.com/endpoint"
#subscription_key ou accesss_key 
#Outros parâmetros necessários 

df_resultado = remover_assinaturas_azure(df_df_entrada, endpoint, subscription_key, deployment)
#print(df_resultado[['Texto', 'TextoSemAssinatura']])


#Gerar stop-words caso não exista o arquivo / deve-se ser adicionado, neste arquivo criado, stopwords do negócio
if not os.path.exists('stopwords.txt'):
        print(f"Arquivo stopwords.txt não encontrado. Gerando a partir dos arquivos IBGE...")
        gerar_stopwords_ibge('ibge-mas-10000.csv', 'ibge-fem-10000.csv')
else:
        print(f"Arquivo stopwords.txt encontrado. Carregando...")
        # Carregando as stopwords em uma lista
        with open('stopwords.txt', "r", encoding="utf-8") as f:
            stopwords = [linha.strip() for linha in f if linha.strip()]


#Executa o pré-processamento
pp = PreProcessing(language="pt")

pp.append_stopwords_list(stopwords)
df_resultado = pd.read_csv('semassinatura.csv')
df_resultado["TextoTratado"] = df_resultado["TextoSemAssinatura"].apply(processing_text)

print(df_resultado)

# Selecionar colunas específicas do DataFrame original para exportar para o módulo 2
df_novo = df_resultado[["ClassN1", "ID", "TextoTratado", "CreateTime", "TotalTimeTracked"]].copy()

# Exportar para CSV
df_novo.to_csv("saida_modulo1.csv", index=False, encoding="utf-8")





