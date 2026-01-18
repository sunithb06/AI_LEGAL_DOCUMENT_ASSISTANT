import os
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------
# 1. Load environment variables FIRST (very important)
# ---------------------------------------------------
load_dotenv()

# DEBUG (temporary – you can remove later)
st.write("DEBUG GROQ KEY LOADED:", os.getenv("GROQ_API_KEY") is not None)

# ---------------------------------------------------
# 2. Import RAG pipeline AFTER loading env
# ---------------------------------------------------
from rag_pipeline import load_rag_pipeline

# ---------------------------------------------------
# 3. Streamlit Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Legal Document Assistant",
    page_icon="",
    layout="centered"
)

st.markdown("##  AI Legal Document Assistant")
st.caption(
    "An intelligent legal document analysis system powered by Retrieval-Augmented Generation"
)

st.divider()

st.warning(
    "Disclaimer: This application provides document-based explanations only and should not be "
    "considered legal advice."
)


# ---------------------------------------------------
# 4. Load RAG Pipeline (NO caching to avoid key issues)
# ---------------------------------------------------
def get_rag_pipeline():
    return load_rag_pipeline()

rag_chain = get_rag_pipeline()

# ---------------------------------------------------
# 5. User Input
# ---------------------------------------------------
user_question = st.text_input(
    "Ask a question about the legal document:",
    placeholder="e.g. What is a lease deed?"
)

# ---------------------------------------------------
# 6. Handle User Question
# ---------------------------------------------------
if user_question:
    with st.spinner("Analyzing the document..."):
        try:
            answer = rag_chain.invoke(user_question)

            st.subheader("📄 Answer")
            st.write(answer)

        except Exception as e:
            st.error("❌ An error occurred")
            st.error(str(e))
