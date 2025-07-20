import streamlit as st
import json
import os
import hashlib
import pickle
from app.doc_ingestor import ingest_documents
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
from datetime import datetime
# Load environment variable
import dotenv
dotenv.load_dotenv()

# Paths
DB_FAISS_PATH = "data/vector_store"
INDEX_METADATA_PATH = "data/vector_store/index.pkl"
FEEDBACK_LOG_PATH = "data/vector_store/feedback_log.json"

# Ingest docs if not already ingested
if not os.path.exists(DB_FAISS_PATH) or not os.path.exists(INDEX_METADATA_PATH):
    ingest_documents()

# Load vector store and retriever
vector_store = FAISS.load_local(DB_FAISS_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Load metadata
with open(INDEX_METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# Utility: Get unique widget key
def get_unique_key(query, doc_id):
    return hashlib.md5(f"{query}_{doc_id}".encode()).hexdigest()

# Streamlit UI
st.title("Document Q&A Assistant")

query = st.text_input("Ask your question:")

if query:
    llm = ChatOpenAI(temperature=0.7)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.run(query)
    st.subheader("Answer:")
    st.write(result)

    # Get retrieved documents
    docs = retriever.get_relevant_documents(query)

    st.subheader("Sources and Feedback")
    for doc in docs:
        doc_metadata = metadata.get(doc.metadata.get("source"), {})
        doc_id = doc.metadata.get("source", "unknown") + "::" + str(doc.metadata.get("page", 0))
        display_name = os.path.basename(doc.metadata.get("source", "Unknown File"))
        st.markdown(f"**{display_name} (Page {doc.metadata.get('page', 0)})**")
        st.markdown(f"> {doc.page_content[:300]}...")  # Truncate preview

        # Generate unique widget key
        key = get_unique_key(query, doc_id)

        feedback = st.radio(
            "Is this relevant?",
            ["Not Rated", "Yes", "No"],
            key=key,
            horizontal=True
        )

        # Save feedback
        if feedback != "Not Rated":
            feedback_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "document_id": doc_id,
                "feedback": feedback,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page", 0)
            }

            # Read and append
            if os.path.exists(FEEDBACK_LOG_PATH):
                with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
                    all_feedback = json.load(f)
            else:
                all_feedback = []

            all_feedback.append(feedback_data)

            # Save back
            with open(FEEDBACK_LOG_PATH, "w") as f:
                json.dump(all_feedback, f, indent=2)

            st.success("Feedback saved!")

