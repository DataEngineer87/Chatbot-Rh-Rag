#!/usr/bin/env python
# coding: utf-8

# In[4]:


# app_streamlit.py
import os
from dotenv import load_dotenv

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Charger variables d'environnement (.env en local ; secrets sur Streamlit Cloud)
load_dotenv()

def recuperer_cle_openai():
    # 1) Vérifier si st.secrets existe
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass  # st.secrets n'existe pas en local

    # 2) Essayer .env
    cle = os.getenv("OPENAI_API_KEY")
    if cle and cle.strip() != "":
        return cle

    # 3) Sinon None
    return None


@st.cache_resource
def charger_index():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    index_path = "embeddings/faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return db

def creer_llm(cle_openai):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        api_key=cle_openai
    )
    return llm

def generer_reponse(question, db, llm):
    # 1. Recherche sémantique dans FAISS
    resultats = db.similarity_search(question, k=3)
    contexte = "\n\n".join([doc.page_content for doc in resultats])

    # 2. Construction du prompt
    prompt_template = ChatPromptTemplate.from_template(
        "Tu es un assistant RH. Réponds à la question suivante à partir du contexte ci-dessous.\n\n"
        "Contexte :\n{context}\n\n"
        "Question : {question}\n\n"
        "Réponse :"
    )
    prompt = prompt_template.format(context=contexte, question=question)

    # 3. Appel au LLM OpenAI
    reponse = llm.invoke(prompt)

    # 4. Extraire les sources simples
    sources = []
    for doc in resultats:
        source = doc.metadata.get("source", "inconnu")
        extrait = doc.page_content[:200].replace("\n", " ")
        sources.append((source, extrait))

    return reponse.content, sources

# === Interface Streamlit ===
def main():
    st.set_page_config(page_title="Chatbot RH RAG", page_icon="💼")
    st.title("💼 Chatbot RH avec RAG (OpenAI + FAISS)")
    st.write("Pose une question sur la politique RH (télétravail, congés, formation, etc.).")

    cle_openai = recuperer_cle_openai()
    if not cle_openai:
        st.error("❌ Clé OpenAI manquante. Ajoute OPENAI_API_KEY dans tes secrets ou ton .env.")
        return

    db = charger_index()
    llm = creer_llm(cle_openai)

    question = st.text_input("🧑‍💼 Votre question :")
    bouton = st.button("Poser la question")

    if bouton and question.strip() != "":
        with st.spinner("🔎 Recherche dans les documents RH + génération de la réponse…"):
            try:
                reponse, sources = generer_reponse(question, db, llm)
                st.success("🤖 Réponse :")
                st.write(reponse)

                st.info("📚 Sources :")
                for source, extrait in sources:
                    st.markdown(f"- **{source}** : {extrait}...")
            except Exception as e:
                st.error(f"⚠️ Erreur pendant le traitement : {e}")

if __name__ == "__main__":
    main()

