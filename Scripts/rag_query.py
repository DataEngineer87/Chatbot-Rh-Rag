#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Chemin de l’index
INDEX_DIR = "embeddings/faiss_index"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Seuil de similarité pour éviter les hallucinations
THRESHOLD = 0.35  

# Mots-clés pour vérifier si la question est RH
KEYWORDS_RH = [
    "rh", "ressources humaines", "télétravail", "congé", "conges",
    "absence", "salaire", "contrat", "formation", "recrutement",
    "onboarding", "employé", "collaborateur", "procédure", "politique"
]

# Prompt strict RH
SYSTEM_PROMPT = """
Tu es un assistant RH interne.
Tu dois répondre uniquement à partir des documents fournis.
Si la question n'est pas liée aux documents RH ou à la politique interne,
réponds exactement :
"Je ne peux répondre qu’aux questions liées aux documents internes RH."
"""

def charger_retriever():
    """Charge l'index FAISS existant."""
    embeddings = HuggingFaceEmbeddings(model_name=MODEL)
    db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def question_concerne_rh(question):
    """Détecte si la question est dans le domaine RH."""
    q = question.lower()
    return any(keyword in q for keyword in KEYWORDS_RH)


def repondre(question):
    """Pipeline complet : vérification + RAG + LLM."""

    # 1) Vérifier domaine RH
    if not question_concerne_rh(question):
        return "Je ne peux répondre qu’aux questions liées aux documents internes RH."

    # 2) Load retriever
    retriever = charger_retriever()

    # 3) Récupérer documents de contexte
    docs = retriever.get_relevant_documents(question)

    if len(docs) == 0:
        return "Je ne peux répondre qu’aux questions liées aux documents internes RH."

    # 4) Vérifier score (si disponible)
    try:
        scores = [d.metadata.get("score", 0.0) for d in docs]
        if min(scores) > THRESHOLD:
            return "Je ne peux répondre qu’aux questions liées aux documents internes RH."
    except:
        pass

    # 5) Construire le prompt
    context = "\n\n".join([d.page_content for d in docs])

    template = PromptTemplate(
        input_variables=["system", "context", "question"],
        template="{system}\n\nContexte:\n{context}\n\nQuestion:\n{question}"
    )

    prompt = template.format(
        system=SYSTEM_PROMPT,
        context=context,
        question=question
    )

    # 6) LLM OpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
    answer = llm.invoke(prompt)

    return answer.content


# In[ ]:




