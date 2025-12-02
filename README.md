# 💼 Assistant RH Intelligent avec IA
## Chatbot RH RAG – OpenAI + FAISS + Streamlit
Développement d’un assistant RH intelligent exploitant les documents internes (PDF) pour répondre automatiquement aux questions des collaborateurs concernant le télétravail, les congés, la formation et autres politiques internes.

Le projet combine :
- Recherche sémantique (FAISS + embeddings)
- IA générative (GPT-4o mini)
- Interface web (Streamlit)
- Architecture propre et déployable (GitHub Actions + Streamlit Cloud)

---
## Objectif
Dans beaucoup d’entreprises, les informations RH sont enfouies dans des PDF ou intranets difficiles à naviguer.

Ce projet montre comment un data scientist / ML engineer peut :
- Transformer ces documents en base de connaissance interrogeable en langage naturel
- Construire un POC fonctionnel et déployé
- Maîtriser la chaîne complète : data → modèle → app → déploiement
---

## Architecture

1. **Indexation**
   - Les PDF RH sont placés dans `Donnees/`
   - `index.py` :
     - extraction du texte (PyPDFLoader)
     - découpage en chunks (LangChain)
     - création des embeddings (HuggingFace MiniLM)
     - construction d'index FAISS et le sauvegarde dans `embeddings/faiss_index`

2. **Application Streamlit**
   - `app_streamlit.py` :
     - chargement de l’index FAISS
     - lecture de clé OpenAI (secrets/.env)
     - envoie la question de l’utilisateur
     - fait une recherche dans l’index
     - construit un prompt et appelle gpt-4o-mini
     - affiche la réponse

3. **Déploiement**
   - CI GitHub Actions (`.github/workflows/ci.yml`)
   - Hébergement Streamlit Cloud :
     - lien public de démonstration
     - clé OpenAI dans les secrets Streamlit

---

## Stack technique
- **NLP / RAG**
  - LangChain 1.x
  - FAISS
  - Sentence-Transformers (MiniLM)
  - OpenAI gpt-4o-mini

- **Backend / App**
  - Python 3.11
  - Streamlit

- **MLOps / DevOps**
  - GitHub Actions (CI)
  - Streamlit Cloud (déploiement)
  - Gestion des secrets (Streamlit + .env)

---

## Installation locale

### Clonage du repo

```bash
git clone https://github.com/DataEngineer87/chatbot-rh-rag-openai.git
cd chatbot-rh-rag-openai

```
## Créer un environnement et installer les dépendances sous Lunix

```bash
conda create -n Projet_rag_rh python=3.11 -y
conda activate Projet_rag_rh
pip install -r requirements.txt

```
## Ajout des fichiers PDF RH
On Place tous les PDFs dans le dossier donnees/ (ex. charte_teletravail.pdf, conges_et_absences.pdf, etc.)

## Création d'un fichier .env
On se connecte à OpenAi et on génère une clé
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxx

## Génération de l’index FAISS
```
python index.py

```

## Lancement de l’app Streamlit

```
streamlit run app_streamlit.py

```

## Déploiement Streamlit Cloud
- On Pousse le projet sur GitHub
- On Crée une app sur Streamlit Cloud en pointant vers app_streamlit.py
- Dans Secrets colle la clé
  
```
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxx"

```
- Une fois déployé, on obtient un lien du type :
  
```
https://chatbot-rh-rag-scmr8r8njizt9pvbp6268f.streamlit.app/

```

## Compétences démontrées

- IA générative & RAG sur documents internes

- NLP appliqué à un cas métier (RH)

- Construction d’un pipeline complet :

- ingestion → indexation ->  recherche -> génération

Industrialisation légère : 

CI GitHub Actions

déploiement cloud

Communication technique (README, interface claire)
```
  











