import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI
)

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv(
    r"C:\Generative AI\.env"
)

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error(
        "MISTRAL_API_KEY not found in C:\\Generative AI\\.env"
    )
    st.stop()


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚"
)


# ==========================================
# TITLE
# ==========================================

st.title("📚 RAG Book Assistant")

st.write(
    "Upload a PDF and ask questions from the document"
)


# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a PDF book",
    type="pdf"
)


# ==========================================
# CREATE MISTRAL EMBEDDING MODEL
# ==========================================

embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=api_key
)


# ==========================================
# UPLOAD PDF
# ==========================================

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.read()
        )

        file_path = tmp_file.name


    st.success(
        "PDF uploaded successfully!"
    )


    # ======================================
    # CREATE VECTOR DATABASE BUTTON
    # ======================================

    if st.button("Create Vector Database"):

        with st.spinner(
            "Processing document..."
        ):

            try:

                # ------------------------------
                # DELETE OLD DATABASE
                # ------------------------------

                if os.path.exists("chroma_db"):

                    shutil.rmtree(
                        "chroma_db"
                    )


                # ------------------------------
                # LOAD PDF
                # ------------------------------

                loader = PyPDFLoader(
                    file_path
                )

                docs = loader.load()


                # ------------------------------
                # SPLIT DOCUMENT
                # ------------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(
                    docs
                )


                # ------------------------------
                # CREATE CHROMA DATABASE
                # ------------------------------

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory="chroma_db"
                )


                st.success(
                    "Vector database created successfully!"
                )

                st.info(
                    f"Created {len(chunks)} document chunks."
                )


            except Exception as e:

                st.error(
                    f"Error creating vector database: {e}"
                )


# ==========================================
# LOAD VECTOR DATABASE
# ==========================================

if os.path.exists("chroma_db"):

    try:

        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embedding_model
        )


        # ==================================
        # RETRIEVER
        # ==================================

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )


        # ==================================
        # MISTRAL LLM
        # ==================================

        llm = ChatMistralAI(
            model="mistral-small-2506",
            api_key=api_key,
            temperature=0
        )


        # ==================================
        # PROMPT
        # ==================================

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

Rules:

1. Do not use outside knowledge.
2. Do not make up information.
3. If the answer is not present in the context,
say:

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


        # ==================================
        # QUESTION SECTION
        # ==================================

        st.divider()

        st.subheader(
            "Ask Questions From the Book"
        )


        query = st.text_input(
            "Enter your question"
        )


        # ==================================
        # RAG QUERY
        # ==================================

        if query:

            with st.spinner(
                "Searching the document..."
            ):

                try:

                    # --------------------------
                    # RETRIEVE DOCUMENTS
                    # --------------------------

                    docs = retriever.invoke(
                        query
                    )


                    if not docs:

                        st.warning(
                            "I could not find the answer in the document."
                        )

                    else:

                        # ----------------------
                        # CREATE CONTEXT
                        # ----------------------

                        context = "\n\n".join(
                            doc.page_content
                            for doc in docs
                        )


                        # ----------------------
                        # CREATE PROMPT
                        # ----------------------

                        final_prompt = prompt.invoke(
                            {
                                "context": context,
                                "question": query
                            }
                        )


                        # ----------------------
                        # GENERATE ANSWER
                        # ----------------------

                        response = llm.invoke(
                            final_prompt
                        )


                        st.write(
                            "### 🤖 AI Answer"
                        )

                        st.write(
                            response.content
                        )


                except Exception as e:

                    st.error(
                        f"Error while answering: {e}"
                    )


    except Exception as e:

        st.error(
            f"Error loading Chroma database: {e}"
        )