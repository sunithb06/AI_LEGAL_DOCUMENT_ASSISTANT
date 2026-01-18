#  AI Legal Document Assistant

An AI-powered application that enables users to ask natural language questions about legal documents and receive accurate, document-grounded answers using **Retrieval-Augmented Generation (RAG)**.

---

## 📌 Project Overview

Legal documents such as rental agreements and contracts are often lengthy and difficult to understand.  
Using a general chatbot can be risky because it may hallucinate or provide incorrect legal information.

This project solves that problem by ensuring that **all answers are generated strictly from the uploaded legal document**.

---

##  Key Features

- Document-based legal question answering  
- Retrieval-Augmented Generation (RAG) architecture  
- Semantic search using vector embeddings  
- Hallucination-controlled responses  
- Streamlit-based user interface  

---

##  Architecture (How It Works)

1. Legal PDF documents are loaded and split into smaller text chunks  
2. Each chunk is converted into embeddings using a transformer model  
3. Embeddings are stored in a FAISS vector database  
4. User questions are converted into embeddings  
5. Relevant document chunks are retrieved using semantic similarity  
6. The retrieved context is passed to an LLM to generate an answer  

---

##  Technology Stack

- **Python**
- **LangChain**
- **FAISS** (Vector Database)
- **HuggingFace Sentence Transformers**
- **Groq LLaMA 3.1 (Free LLM API)**
- **Streamlit**

---



