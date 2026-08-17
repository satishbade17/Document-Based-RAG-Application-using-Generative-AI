from pathlib import Path
import os

from dotenv import load_dotenv

from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI
)

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = BASE_DIR.parent / ".env"

CHROMA_DIR = BASE_DIR / "chroma_db"


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv(ENV_FILE)

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError(
        "MISTRAL_API_KEY not found.\n"
        "Please add it to C:\\Generative AI\\.env"
    )


# ==========================================
# MISTRAL EMBEDDING MODEL
# ==========================================

embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=api_key
)


# ==========================================
# CHROMA VECTOR DATABASE
# ==========================================

if not CHROMA_DIR.exists():
    raise FileNotFoundError(
        "chroma_db does not exist.\n"
        "Create the database first using create_database.py."
    )


vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embedding_model
)


# ==========================================
# RETRIEVER
# ==========================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# ==========================================
# MISTRAL LLM
# ==========================================

llm = ChatMistralAI(
    model="mistral-small-2506",
    api_key=api_key,
    temperature=0
)


# ==========================================
# PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not make up information.
3. If the answer is not present in the context, say:
   "I could not find the answer in the document."
4. Give a clear and concise answer.
"""
        ),
        (
            "human",
            """
Context:

{context}


Question:

{question}
"""
        )
    ]
)


# ==========================================
# START RAG
# ==========================================

print()
print("=" * 50)
print("       MISTRAL RAG CHATBOT")
print("=" * 50)
print("Model: mistral-small-2506")
print("Embeddings: mistral-embed")
print("Vector DB: Chroma")
print("Retriever: MMR")
print("Type 0 to exit")
print("=" * 50)


# ==========================================
# CHAT LOOP
# ==========================================

while True:

    query = input("\nYou: ").strip()

    if query == "0":
        print("\nGoodbye!")
        break

    if not query:
        print("Please enter a question.")
        continue


    # --------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------

    try:

        docs = retriever.invoke(query)

    except Exception as e:

        print(f"\nRetrieval error: {e}")
        continue


    if not docs:

        print(
            "\nAI: I could not find the answer in the document."
        )

        continue


    # --------------------------------------
    # CREATE CONTEXT
    # --------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # --------------------------------------
    # CREATE PROMPT
    # --------------------------------------

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )


    # --------------------------------------
    # GENERATE ANSWER
    # --------------------------------------

    try:

        response = llm.invoke(final_prompt)

        print("\nAI:", response.content)

    except Exception as e:

        print(f"\nLLM error: {e}")