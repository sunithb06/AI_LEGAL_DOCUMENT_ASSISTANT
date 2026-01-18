import os

# PDF loader (reads legal PDF page by page)
from langchain_community.document_loaders import PyPDFLoader

# Text splitter (NEW correct import)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings (FREE – runs locally)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector database (FREE – local)
from langchain_community.vectorstores import FAISS


# Folder where your PDF is stored
DATA_PATH = "data/legal_docs"

# Folder where FAISS vector DB will be saved
VECTOR_DB_PATH = "vectorstore"


def ingest_documents():
    """
    This function:
    1. Loads all PDF files from DATA_PATH
    2. Splits text into chunks
    3. Converts text to embeddings
    4. Stores them in FAISS vector database
    """

    documents = []

    # 1️⃣ Load all PDF files
    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
            documents.extend(pdf_docs)

    print(f"📄 Loaded {len(documents)} pages from PDF(s)")

    # 2️⃣ Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # good for legal text
        chunk_overlap=200     # preserves context between clauses
    )

    chunks = splitter.split_documents(documents)
    print(f"✂️ Created {len(chunks)} text chunks")

    # 3️⃣ Create embeddings (FREE model)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4️⃣ Create and save FAISS vector store
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)

    print("✅ Vector database created and saved successfully!")


# Run ingestion
if __name__ == "__main__":
    ingest_documents()
