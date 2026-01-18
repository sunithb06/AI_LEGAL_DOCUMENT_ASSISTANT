import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

VECTOR_DB_PATH = "vectorstore"


def load_rag_pipeline():
    # 1️⃣ Load embeddings (same as ingest)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2️⃣ Load FAISS vector DB
    vector_db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 3️⃣ UPDATED FREE GROQ MODEL (FIXED HERE)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",   # ✅ UPDATED MODEL
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 4️⃣ Prompt template (legal-safe)
    prompt = ChatPromptTemplate.from_template(
        """
        You are a legal document assistant.
        Answer the question ONLY using the context below.
        Do not give legal advice.

        Context:
        {context}

        Question:
        {question}
        """
    )

    # 5️⃣ Modern RAG chain (LCEL)
    rag_chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
