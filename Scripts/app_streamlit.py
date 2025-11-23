#!/usr/bin/env python
# coding: utf-8

# In[6]:


# applcation_streamlit.py
import os
from dotenv import load_dotenv

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Chargement des variables d'environnement (.env en local ; on met la clé dans le secrets sur Streamlit Cloud)
load_dotenv()

def recuperer_cle_openai():
    # Priorité aux secrets Streamlit (en production)
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    # Sinon .env / variable d'environnement (local)
    return os.getenv("OPENAI_API_KEY")

@st.cache_resource
def charger_index():
    # modèle d’embeddings (vectorisation avancée)
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    index_path = "embeddings/faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return db

def creer_llm(cle_openai):
    # modèle de gestion de l’intelligence linguistique (gpt-3.5)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        api_key=cle_openai
    )
    return llm

def generer_reponse(question, db, llm):
    # Recherche sémantique dans FAISS
    resultats = db.similarity_search(question, k=3)
    contexte = "\n\n".join([doc.page_content for doc in resultats])

    # Construction du prompt
    prompt_template = ChatPromptTemplate.from_template(
        "Tu es un assistant RH. Réponds à la question suivante à partir du contexte ci-dessous.\n\n"
        "Contexte :\n{context}\n\n"
        "Question : {question}\n\n"
        "Réponse :"
    )
    prompt = prompt_template.format(context=contexte, question=question)

    # Appel au LLM OpenAI
    reponse = llm.invoke(prompt)

    # Extrairaction des sources 
    sources = []
    for doc in resultats:
        source = doc.metadata.get("source", "inconnu")
        extrait = doc.page_content[:200].replace("\n", " ")
        sources.append((source, extrait))

    return reponse.content, sources

# Interface Streamlit 
def main():
    st.set_page_config(page_title="Chatbot RH RAG")
    st.title("Chatbot RH avec RAG (OpenAI + FAISS)")
    st.write("Pose une question sur la politique RH (télétravail, congés, formation, etc.).")

# Récupération de clé OpenAI depuis les secrets ou un .env
    cle_openai = recuperer_cle_openai()

    ## Si la clé API est absente, on affiche une erreur et le programme s'arrête
    if not cle_openai:
        st.error("Si la clé OpenAI est manquante. Ajoute OPENAI_API_KEY dans tes secrets ou ton .env.")
        return

    # Chargement l'index FAISS (base vectorielle)
    db = charger_index()

    # Création de l'objet LLM OpenAI (gpt-3.5-turbo)
    llm = creer_llm(cle_openai)

    # Champ de chaisie d'une requête par l’utilisateur
    question = st.text_input("🧑‍💼 Votre question, svp :")
    bouton = st.button("Envoyer")

    if bouton and question.strip() != "":
        with st.spinner("🔎 Recherche dans les documents RH + génération de la réponse…"):
            try:

                # Étape principale : recherche vectorielle + génération LLM
                reponse, sources = generer_reponse(question, db, llm)
                st.success("Réponse :")
                st.write(reponse)

                st.info("📚 Sources :")
                for source, extrait in sources:
                    st.markdown(f"- **{source}** : {extrait}...")
            except Exception as e:
                st.error(f"Erreur pendant le traitement : {e}")

if __name__ == "__main__":
    main()


# In[ ]:




