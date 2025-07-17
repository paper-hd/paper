from tm_module.utils.reader import Reader
from tm_module.cluwords import CluWords
import pandas as pd


# Leitura dos dados saídos do módulo 1
reader = Reader(path="saida_modulo1.csv", id_column="ID", text_column="TextoTratado")

# Create the CluWords object
cluwords = CluWords(reader)
cluwords.generate_representation(
    embedding_file="embedding.vec",
    embedding_binary=False,
    k_neighbors=500,
    n_threads=20,
    threshold=0.4
)
resultado = cluwords.get_topics(10, 5, save_path="saidaMT")

#Agrega as saídas da MT com os dados originais
idTopic = pd.read_csv('saidaMTResumo_Topicos_Dominantes.csv', sep=',')
idTopic['top_words'] = idTopic['dominant_topic'].apply(lambda x: resultado[x])
idTopic['top_words'] = idTopic['top_words'].apply(lambda x: ', '.join(x))
idTopic

#Importar o resultado do modulo 1 para um DF
df_textos = pd.read_csv('saida_modulo1.csv')

#Renomear a coluna 'papers' do idTopic para 'ID' para facilitar o merge
idTopic_renomeado = idTopic.rename(columns={'papers': 'ID'})

#Fazer o merge com base na coluna 'ID'
df_final = pd.merge(df_textos, idTopic_renomeado, on='ID', how='left')

#Reordenar as colunas conforme solicitado
df_final = df_final[['ClassN1', 'ID', 'TextoTratado', 'CreateTime', 'TotalTimeTracked', 'dominant_topic', 'top_words']]

df_final.to_csv("saida_modulo2.csv", index=False, encoding="utf-8")
#Este arquivo será utilizado pelo PowerBI


