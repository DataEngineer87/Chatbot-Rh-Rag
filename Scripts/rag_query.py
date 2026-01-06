#!/usr/bin/env python
# coding: utf-8

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Chemin FAISS
INDEX_DIR = "embeddings/faiss_index"

# Modèle d'embedding
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Mots clés pour vérifier la nature RH de la question
KEYWORDS_RH = [
    "rh", "ressources humaines", "télétravail", "teletravail", "congé", "conges",
    "absence", "salaire", "contrat", "formation", "recrutement",
    "employé", "collaborateur", "procédure", "politique", "rémunération",
    "remuneration", "absence", "convention", "accord", "document interne"
]

# Prompt système renforcé
SYSTEM_PROMPT = """
Tu es un assistant RH interne.
Tu réponds strictement à partir des documents internes fournis dans le contexte.
Si les documents permettent d’apporter une réponse, tu dois répondre clairement.
Si la réponse ne se trouve pas dans les documents internes RH,
réponds exactement :
"Je ne peux répondre qu’aux questions liées aux documents internes RH."
"""

# Charge l’index FAISS
def charger_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL)
    db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    return retriever

# Détection basique domaine RH
def question_concerne_rh(question):
    q = question.lower()
    return any(keyword in q for keyword in KEYWORDS_RH)

# Pipeline RAG complet.
def repondre(question):

    # Vérifier domaine RH
    if not question_concerne_rh(question):
        return "Je ne peux répondre qu’aux questions liées aux documents internes RH."

    # Charger FAISS
    try:
        retriever = charger_retriever()
    except Exception as e:
        return "Erreur interne : impossible de charger les documents."

    # Récupération des documents pertinents
    try:
        docs = retriever.invoke(question)
    except Exception:
        return "Je ne peux répondre qu’aux questions liées aux documents internes RH."

    # Aucun document trouvé
    if not docs or len(docs) == 0:
        return "Je ne peux répondre qu’aux questions liées aux documents internes RH."

    # Construction du contexte
    context = "\n\n".join([d.page_content for d in docs])

    # Appel LLM avec prompt système + contexte
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion:\n{question}"}
    ]

    llm = ChatOpenAI(model="gpt-4o-mini")

    try:
        answer = llm.invoke(messages)
        return answer.content
    except Exception:
        return "Erreur interne : impossible de générer une réponse."
