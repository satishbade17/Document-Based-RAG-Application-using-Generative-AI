# 📚 Document-Based RAG Application using Generative AI

A **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions based on their content.

## 🚀 Project Workflow

```text
PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Embeddings
     ↓
ChromaDB
     ↓
Document Retrieval
     ↓
LLM
     ↓
AI Generated Answer
```

## 🛠️ Technologies Used

* Python
* Streamlit
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Transformers
* Large Language Models
* Retrieval-Augmented Generation (RAG)

## 🔍 Retrieval Techniques

This project also explores different retrieval approaches:

* **Similarity Search** – retrieves documents based on semantic similarity.
* **MMR (Maximal Marginal Relevance)** – retrieves relevant and diverse documents.
* **MultiQuery Retriever** – generates multiple query variations to improve retrieval.
* **Arxiv Retriever** – retrieves research papers from arXiv.

## 🎯 Objective

The objective of this project is to build a document-based question-answering system where relevant information is retrieved from uploaded documents and provided to an AI model for generating answers.

## 💡 Key Learning

Through this project, I learned how:

* PDF documents can be processed for RAG.
* Text can be divided into meaningful chunks.
* Embeddings represent text as vectors.
* ChromaDB can be used as a vector database.
* Retrievers can find relevant information from documents.
* LLMs can generate answers using retrieved context.
* Different retrieval techniques can improve document search.

## 📂 Project Structure

```text
RAG/
│
├── retrievers/
│   ├── arxiv.py
│   ├── mmr.py
│   └── multiquery.py
│
├── app.py
├── create_database.py
├── main.py
├── requirements.txt
├── README.md
│
├── chroma_db/        # Local vector database - do not upload
└── .venv/            # Virtual environment - do not upload
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd RAG
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL in your browser.

## 📌 Future Improvements

* Support multiple PDF documents
* Improve retrieval accuracy
* Add conversation memory
* Add source/document references
* Improve embedding and retrieval performance
* Add a better user interface

## 👩‍💻 Project

**Document-Based RAG Application using Generative AI**

Built using Python, LangChain, ChromaDB, Streamlit, embeddings, and LLM-based retrieval.
