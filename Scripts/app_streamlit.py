# =====================================================
# app_streamlit.py
# =====================================================
import os
from dotenv import load_dotenv
import streamlit as st

# On importe ta fonction RAG corrigée
from rag_query import repondre

# Chargement du .env en local
load_dotenv()

def recuperer_cle_openai():
    """
    Récupère la clé OpenAI en priorité depuis les secrets Streamlit Cloud,
    puis depuis .env en local.
    """
    # Secrets Streamlit Cloud
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    # Fichier .env en local
    cle = os.getenv("OPENAI_API_KEY")
    if cle and cle.strip() != "":
        return cle

    return None


# ====================================================
# Interface Streamlit
# =======================================================

def main():
    st.set_page_config(page_title="Chatbot RH RAG")
    st.title("Chatbot RH avec RAG (OpenAI + FAISS)")
    st.write("Pose une question sur la politique RH (télétravail, congés, formation, etc.).")

    # Récupération de clé OpenAI
    cle_openai = recuperer_cle_openai()

    if not cle_openai:
        st.error("Clé OpenAI manquante.\n\nAjoute `OPENAI_API_KEY` dans :\n- `.env` en local\n- `Secrets` sur Streamlit Cloud.")
        return

    # Saisie utilisateur
    question = st.text_input("🧑‍💼 Votre question, svp :")
    bouton = st.button("Envoyer")

    if bouton and question.strip() != "":
        with st.spinner("🔎 Recherche dans les documents RH + Génération de la réponse…"):
            try:
                # Appel du pipeline RAG complet
                reponse = repondre(question)
                st.success("Réponse :")
                st.write(reponse)

            except Exception as e:
                st.error(f"Erreur pendant le traitement : {e}")


if __name__ == "__main__":
    main()
