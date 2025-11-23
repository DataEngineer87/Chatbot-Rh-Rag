#!/usr/bin/env python
# coding: utf-8

# In[3]:


# construction d'index
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Chemin du dossier
PDF_DIR = "/home/sacko/Documents/Chatbot-Rh-Rag/Donnees"

# Chemin d'enregistrement de l’index FAISS
INDEX_DIR = "embeddings/faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# Modèle d'embedding
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def charger_documents(pdf_dir):
    print("Chargement des documents PDF…")

    docs = []  # Liste qui contiendra toutes les pages extraites des fichiers PDF
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            chemin = os.path.join(pdf_dir, file)

            # Chargeur LangChain spécialisé pour lire les PDF
            loader = PyPDFLoader(chemin)
            docs.extend(loader.load())
    print(len(docs), "pages chargées.")
    return docs

# Création d'une fonction qui coupe les textes en morceaux
def splitter_documents(docs):

    # chunk_size=500 : chaque morceau fait environ 500 caractères
    # chunk_overlap=50 : 50 caractères se chevauchent entre deux chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    textes = splitter.split_documents(docs)
    print(len(textes), "chunks créés.")
    return textes

def construire_index(textes, model_name, index_dir):
    print("Génération de l'index FAISS…")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(textes, embeddings)
    db.save_local(index_dir)
    print("Index sauvegardé dans :", index_dir)

if __name__ == "__main__":
    docs = charger_documents(PDF_DIR)
    textes = splitter_documents(docs)
    construire_index(textes, MODEL, INDEX_DIR)


# In[ ]:




