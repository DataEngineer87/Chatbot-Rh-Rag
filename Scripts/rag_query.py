#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

INDEX_DIR = "embeddings/faiss_index"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MESSAGE_HORS_RH = (
    "Je ne trouve pas d'information fiable dans les documents RH fournis "
    "pour répondre à cette question."
)

KEYWORDS_RH = [
    "rh", "ressources humaines", "télétravail", "teletravail",
    "congé", "conges", "congés", "absence", "absences", "rtt",
    "salaire", "rémunération", "remuneration", "contrat",
    "formation", "recrutement", "employé", "employe", "salarié", "salarie",
    "collaborateur", "politique", "condition", "accord",
    "maladie", "maternité", "paternité", "paie",
    "frais", "équipement", "equipement", "droit", "déconnexion"
]

SYSTEM_PROMPT = """
Tu es un assistant RH interne.

Ton rôle :
- répondre uniquement à partir des documents RH fournis ;
- expliquer les règles de façon claire, naturelle et professionnelle ;
- éviter les formulations trop robotiques ;
- ne pas inventer d’informations ;
- citer les sources utilisées quand c’est pertinent.

Style attendu :
- réponse humaine, fluide et utile ;
- paragraphes courts ;
- listes uniquement si elles améliorent la lisibilité ;
- ton professionnel mais accessible.

Si le contexte ne contient pas l'information demandée, réponds :
"Je ne trouve pas d'information fiable dans les documents RH fournis pour répondre à cette question."
"""

def question_concerne_rh(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in KEYWORDS_RH)

def charger_db():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL)
    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

def calculer_score_confiance(scores):
    """
    FAISS retourne une distance.
    Plus la distance est faible, plus le document est proche.
    """
    if not scores:
        return 0

    meilleur_score = min(scores)

    if meilleur_score < 10:
        return 90
    elif meilleur_score < 20:
        return 80
    elif meilleur_score < 35:
        return 70
    elif meilleur_score < 60:
        return 55
    else:
        return 40

def formater_sources(docs):
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "Document inconnu")
        page = doc.metadata.get("page", None)

        nom_fichier = os.path.basename(source)

        if page is not None:
            sources.append(f"- `{nom_fichier}`, page {page + 1}")
        else:
            sources.append(f"- `{nom_fichier}`")

    return "\n".join(list(dict.fromkeys(sources)))

def construire_historique(historique):
    if not historique:
        return ""

    derniers_messages = historique[-6:]

    texte = ""
    for message in derniers_messages:
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            texte += f"Utilisateur : {content}\n"
        elif role == "assistant":
            texte += f"Assistant : {content}\n"

    return texte

def repondre(question: str, historique=None):
    question = question.strip()

    if not question:
        return {
            "reponse": "Posez-moi une question RH pour commencer.",
            "sources": "",
            "score_confiance": 0
        }

    if not question_concerne_rh(question):
        return {
            "reponse": MESSAGE_HORS_RH,
            "sources": "",
            "score_confiance": 0
        }

    try:
        db = charger_db()
    except Exception as e:
        return {
            "reponse": f"Impossible de charger la base documentaire. Détail : {e}",
            "sources": "",
            "score_confiance": 0
        }

    try:
        docs_scores = db.similarity_search_with_score(question, k=8)
    except Exception as e:
        return {
            "reponse": f"Impossible d’interroger les documents. Détail : {e}",
            "sources": "",
            "score_confiance": 0
        }

    if not docs_scores:
        return {
            "reponse": MESSAGE_HORS_RH,
            "sources": "",
            "score_confiance": 0
        }

    # On garde seulement les meilleurs documents pour éviter les sources parasites
    docs_scores = docs_scores[:4]

    docs = [doc for doc, score in docs_scores]
    scores = [score for doc, score in docs_scores]

    score_confiance = calculer_score_confiance(scores)

    contexte = "\n\n".join(
        [
            f"[Source : {os.path.basename(doc.metadata.get('source', 'document'))}, "
            f"page {doc.metadata.get('page', 'N/A')}]\n"
            f"{doc.page_content}"
            for doc in docs
        ]
    )

    historique_texte = construire_historique(historique)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
Historique récent :
{historique_texte}

Documents RH disponibles :
{contexte}

Question :
{question}

Consigne :
Réponds naturellement, comme un assistant RH qui explique à un salarié.
Quand tu cites une règle importante, indique la source entre parenthèses, par exemple :
(Source : accord_teletravail.pdf, page 3).
"""
        }
    ]

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    try:
        answer = llm.invoke(messages)

        return {
            "reponse": answer.content,
            "sources": formater_sources(docs),
            "score_confiance": score_confiance
        }

    except Exception as e:
        return {
            "reponse": f"Impossible de générer une réponse. Détail : {e}",
            "sources": formater_sources(docs),
            "score_confiance": score_confiance
        }

