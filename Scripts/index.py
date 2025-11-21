#!/usr/bin/env python
# coding: utf-8

# In[4]:


# build_index.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Dossiers
PDF_DIR = "/home/sacko/Documents/Chatbot-Rh-Rag/Donnees"
INDEX_DIR = "embeddings/faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# Modèle d'embedding
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def charger_documents(pdf_dir):
    print("📚 Chargement des documents PDF…")
    docs = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            chemin = os.path.join(pdf_dir, file)
            loader = PyPDFLoader(chemin)
            docs.extend(loader.load())
    print("✅", len(docs), "pages chargées.")
    return docs

def splitter_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    textes = splitter.split_documents(docs)
    print("✂️", len(textes), "chunks créés.")
    return textes

def construire_index(textes, model_name, index_dir):
    print("🔢 Génération de l'index FAISS…")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(textes, embeddings)
    db.save_local(index_dir)
    print("✅ Index sauvegardé dans :", index_dir)

if __name__ == "__main__":
    docs = charger_documents(PDF_DIR)
    textes = splitter_documents(docs)
    construire_index(textes, MODEL, INDEX_DIR)


# In[ ]:




