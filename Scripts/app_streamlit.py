#!/usr/bin/env python
# coding: utf-8

# In[8]:


# =====================================================
# Application_Web_Streamlit.py
# =====================================================

import os
from dotenv import load_dotenv
import streamlit as st

# Initialisation du module de traitement des requêtes RAG
from rag_query import repondre

# Chargement du .env en local
load_dotenv() 

def recuperer_cle_openai():
    """
    Récupère la clé OpenAI en priorité depuis les secrets Streamlit Cloud,
    puis depuis .env en local.
    """

    # Récupération sécurisée de la clé API OpenAI depuis st.secrets 
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    # Vérification de la clé API OpenAI
    cle = os.getenv("OPENAI_API_KEY")
    if cle and cle.strip():
        return cle

# Initialisation de l’état de session des messages
def initialiser_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Implémentation d’une Interface Graphique Personnalisée avec Streamlit et CSS
def appliquer_style():
    st.markdown(
        """
        <style>
        /* Fond général */
        .stApp {
            background: linear-gradient(135deg, #f4f7fb 0%, #eef3f8 100%);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
        }

        section[data-testid="stSidebar"] * {
            color: #ffffff;
        }

        /* Titre principal */
        h1 {
            color: #1e3a8a;
            font-weight: 800;
        }

        h3, h4 {
            color: #1e293b;
        }

        /* Messages */
        div[data-testid="stChatMessage"] {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 12px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
            margin-bottom: 12px;
        }

        /* Boutons */
        .stButton > button {
            background-color: #2563eb;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.6rem 1rem;
            font-weight: 600;
        }

        .stButton > button:hover {
            background-color: #1d4ed8;
            color: white;
        }

        /* Expander */
        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
        }

        /* Info box */
        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        /* Barre de progression */
        div[role="progressbar"] > div {
            background-color: #2563eb;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar de l’Assistant RH IA
def afficher_sidebar():
    with st.sidebar:
        st.title("📚 Assistant RH IA")

        st.markdown(
            """
            Cet assistant répond aux questions RH à partir de documents internes.

            Il utilise un système **RAG** :
            documents → recherche sémantique → réponse contextualisée.
            """
        )

        st.divider()

        st.subheader("📄 Documents couverts")
        st.markdown(
            """
            - Télétravail  
            - Congés & absences  
            - Rémunération  
            - Politiques internes  
            """
        )

        st.divider()

        st.subheader("🛠 Technologies")
        st.markdown(
            """
            - Python  
            - LangChain  
            - FAISS  
            - OpenAI GPT-4o mini  
            - Streamlit  
            """
        )

        st.divider()

        if st.button("🧹 Nouvelle conversation"):
            st.session_state.messages = []
            st.rerun()


# Interface d’accueil de l’Assistant RH Intelligent
def afficher_header():
    st.title("🤖 Assistant RH intelligent")

    st.markdown(
        """
        **Posez une question concernant les politiques RH internes de l’entreprise.**  
        """
    )

    st.info(
        "Exemple : Quelle est la politique de télétravail ?",
        icon="💬"
    )


# Module d’évaluation et de visualisation du score de confiance
def afficher_score_confiance(score):
    score = max(0, min(score, 100))

    st.markdown("#### 📊 Niveau de confiance")
    st.progress(score / 100)
    st.write(f"**{score}%**")

    if score >= 75:
        st.success("Réponse fortement appuyée par les documents retrouvés.")
    elif score >= 50:
        st.warning("Réponse correcte, mais à vérifier selon le contexte exact.")
    else:
        st.error("Confiance faible : les documents retrouvés sont peu proches de la question.")


# Affichage dynamique des conversations utilisateur-assistant
def afficher_historique():
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role, avatar="🧑‍💼" if role == "user" else "🤖"):
            st.markdown(content)


# Orchestration principale de l’application Assistant RH IA
def main():
    st.set_page_config(
        page_title="Assistant RH IA",
        page_icon="🤖",
        layout="wide"
    )

    appliquer_style()

    initialiser_session()
    afficher_sidebar()
    afficher_header()

    # Récupération de clé OpenAI
    cle_openai = recuperer_cle_openai()

    # Vérification de la présence de la clé API OpenAI
    if not cle_openai:
        st.error(
            """
            Clé API OpenAI introuvable.

            Veuillez configurer `OPENAI_API_KEY` :
            - dans le fichier `.env` en environnement local
            - ou dans les Secrets de Streamlit Cloud
            """
        )
        st.stop()

    # Affichage de l’historique des conversations
    afficher_historique()


    # Gestion de la saisie des questions utilisateur dans l’interface conversationnelle RH
    question = st.chat_input("✒️ Posez votre question RH ici...")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # Workflow de traitement des questions RH par intelligence artificielle
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🤖"):

                with st.spinner("Consultation des documents RH..."):

                    resultat = repondre(question)
                    reponse = resultat.get("reponse", "")
                    sources = resultat.get("sources", "")
                    score = resultat.get("score_confiance", 0)

                    st.markdown(reponse)

        with st.expander(" 💼 Sources utilisées"):
            if sources:
                st.markdown(sources)
            else:
                st.write("Aucune source pertinente trouvée.")

        with st.expander("📊 Score de confiance"):
            afficher_score_confiance(score)

            st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reponse
            }
            )

if __name__ == "__main__":
    main()

