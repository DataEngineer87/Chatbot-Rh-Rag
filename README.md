## Technologies utilisées
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-412991?logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 🤖 Chatbot RH Intelligent — RAG + OpenAI + FAISS
💼 Assistant conversationnel permettant de consulter automatiquement les politiques RH internes (PDF) via un système de Retrieval-Augmented Generation (RAG).

- Réponses contextualisées basées sur les documents internes
- Recherche sémantique avec FAISS + embeddings
- Interface interactive Streamlit
- Déploiement cloud avec Streamlit Cloud
- Mémoire conversationnelle + citations des sources  
---
### Démonstration interactive
![Démo Streamlit](images/DemoStreamlit.gif)

### 🚀 Application déployée sur Streamlit Cloud : 
[Tester l’application ici](https://chatbot-rh-rag-scmr8r8njizt9pvbp6268f.streamlit.app/)

💬 Exemples de questions :
- "Combien de jours de télétravail sont autorisés ?"
- "Quelle est la politique de congés ?"
- "L’employeur peut-il imposer le télétravail en cas de crise sanitaire ou de confinement ?"
- "Combien de jours de RTT un salarié à temps complet peut-il obtenir par an et quelles sont les conditions de prise ?"
- "Quels frais liés au télétravail peuvent être pris en charge par l’entreprise ?"

---
### Objectif métier

Dans de nombreuses entreprises, les informations RH sont :
- dispersées dans des PDF
- difficiles à rechercher
- peu accessibles aux employés

**Ce projet montre comment :**
- transformer des documents internes en **base de connaissance interrogeable en langage naturel**
- automatiser les réponses aux questions RH
- améliorer l’accès à l’information interne

---
### Impact business

**Ce système permet :**

- 📉 Réduction du nombre de tickets RH
- ⏰ Gain de temps pour les employés
- 📚 Accès instantané aux politiques internes
- 🤖 Automatisation des FAQ (Foire Aux Questions) RH

**👉 Impact : ce type de solution peut réduire jusqu’à 30–50% des sollicitations RH simples.**

---

### Architecture

Pipeline complet de type RAG :

1. 📄 Ingestion des documents RH (PDF)
2. ✂️ Découpage en chunks (LangChain)
3. 🔎 Création d’embeddings (MiniLM)
4. 🧠 Indexation vectorielle (FAISS)
5. 🔍 Recherche sémantique
6. 🤖 Génération de réponse (GPT-4o-mini)
7. 🌐 Interface utilisateur (Streamlit)

---
## Architecture technique

- Ingestion : PyPDFLoader
- Processing : LangChain (chunking)
- Embeddings : Sentence Transformers (MiniLM)
- Stockage vectoriel : FAISS
- LLM : OpenAI (gpt-4o-mini)
- Frontend : Streamlit
- CI/CD : GitHub Actions
- Déploiement : Streamlit Cloud
---

---

## 📊 Résultats

- ⏰ Temps de réponse rapide (< 2s)
- 🎯 Réponses contextualisées basées sur les documents internes
- 📉 Réduction potentielle de la charge RH
- 📚 Accès simplifié à l’information

---

### Stack technique

#### NLP / RAG
- LangChain
- FAISS
- Sentence-Transformers (MiniLM)
- OpenAI (gpt-4o-mini)

#### Backend / App
- Python 3.11
- Streamlit

#### MLOps / DevOps
- GitHub Actions (CI)
- Streamlit Cloud (déploiement)
- Gestion des secrets (.env + Streamlit)

### Installation locale

git clone https://github.com/DataEngineer87/chatbot-rh-rag-openai.git
cd chatbot-rh-rag-openai

conda create -n Projet_rag_rh python=3.11 -y
conda activate Projet_rag_rh

pip install -r requirements.txt

## Configuration de clé OpenAI

---
Créer un fichier sur votre machine .env :

OPENAI_API_KEY=your_key_here

---

## Technologies utilisées
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-412991?logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

### Démonstration interactive
[Application hébergée sur Streamlit Cloud](https://chatbot-rh-rag-scmr8r8njizt9pvbp6268f.streamlit.app/)

# Objectif :
Ce projet vise à créer un assistant RH intelligent capable de répondre aux questions des employés concernant :
le télétravail, les Congés & Absences, la formation et autres politiques internes.

**Le projet combine :**
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
- Maîtriser la chaîne complète : data -> modèle -> app -> déploiement
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
On Place tous les PDFs dans le dossier `Donnees/` (ex. charte_teletravail.pdf, conges_et_absences.pdf, etc.)

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

## Industrialisation légère : 

- CI GitHub Actions

- déploiement cloud

- Communication technique (README, interface claire)
  
## Auteur
**Alseny — Data Scientist confirmé orienté MLOps & GenAI**











