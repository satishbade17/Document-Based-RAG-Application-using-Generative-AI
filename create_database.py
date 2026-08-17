# ==========================================
# LOAD PDF
# SPLIT INTO CHUNKS
# CREATE MISTRAL EMBEDDINGS
# STORE INTO CHROMA
# ==========================================

from pathlib import Path
import os
import shutil

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

PDF_FILE = BASE_DIR / "document loaders" / "deeplearning.pdf"

CHROMA_DIR = BASE_DIR / "chroma_db"

ENV_FILE = BASE_DIR.parent / ".env"


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
# CHECK PDF
# ==========================================

if not PDF_FILE.exists():
    raise FileNotFoundError(
        f"PDF not found:\n{PDF_FILE}"
    )


print()
print("=" * 60)
print("       CREATING MISTRAL CHROMA DATABASE")
print("=" * 60)


# ==========================================
# DELETE OLD CHROMA DATABASE
# ==========================================

if CHROMA_DIR.exists():

    print("\nDeleting old Chroma database...")

    shutil.rmtree(CHROMA_DIR)

    print("Old Chroma database deleted.")


# ==========================================
# LOAD PDF
# ==========================================

print("\nLoading PDF...")

loader = PyPDFLoader(str(PDF_FILE))

docs = loader.load()

print(f"Pages loaded: {len(docs)}")


# ==========================================
# SPLIT INTO CHUNKS
# ==========================================

print("\nSplitting document into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print(f"Chunks created: {len(chunks)}")


# ==========================================
# CREATE MISTRAL EMBEDDINGS
# ==========================================

print("\nCreating Mistral embedding model...")

embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=api_key
)


# ==========================================
# STORE INTO CHROMA
# ==========================================

print("\nCreating Chroma vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(CHROMA_DIR)
)


# ==========================================
# DONE
# ==========================================

print("\n" + "=" * 60)
print("       DATABASE CREATED SUCCESSFULLY")
print("=" * 60)

print(f"\nPDF:")
print(PDF_FILE)

print("\nEmbedding model:")
print("mistral-embed")

print("\nEmbedding dimension:")
print("1024")

print("\nTotal chunks:")
print(len(chunks))

print("\nChroma database:")
print(CHROMA_DIR)

print("\nNow run:")
print("python main.py")

print("=" * 60)