import streamlit as st
from app.doc_ingestor import ingest_documents
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import pickle
import os
from app.utils import load_feedback, save_feedback, filter_no_feedback

# Load environment variable
import dotenv
dotenv.load_dotenv()

os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/vector_store", exist_ok=True)

# Constants
DB_FAISS_PATH = "data/vector_store"
INDEX_METADATA_PATH = "data/vector_store/index.pkl"
FEEDBACK_LOG_PATH = "data/feedback_log.json"

# Ingest docs if not already ingested
if not os.path.exists(DB_FAISS_PATH) or not os.path.exists(INDEX_METADATA_PATH):
    ingest_documents()

# Load index and vector store
with open(INDEX_METADATA_PATH, "rb") as f:
    index_metadata = pickle.load(f)

vector_store = FAISS.load_local(DB_FAISS_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(temperature=0.3)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Load feedback
feedback_log = load_feedback(FEEDBACK_LOG_PATH)

st.title("Semantic Search App with Feedback")

# Sidebar actions
with st.sidebar:
    st.header("Admin Controls")
    if st.button("Re-Ingest Documents"):
        ingest_documents()
        st.success("Re-ingestion completed.")

    if st.button("Retrain Model"):
        st.info("Training model with feedback... (Mock action)")
        # Here you can hook into a fine-tuning or filtering mechanism

# User query input
query = st.text_input("Enter your question:")
if query:
    # Run search
    results = retriever.get_relevant_documents(query)
    docs = []

    if not results:
        st.warning("No results found for this query.")
    else:
        for r in results:
            r.metadata["id"] = f"{r.metadata.get('source')}::{r.metadata.get('page', 0)}"
            docs.append({
                "id": r.metadata["id"],
                "content": r.page_content,
                "source": r.metadata.get("source"),
                "page": r.metadata.get("page", 0)
            })


    # Filter using prior feedback
    docs = filter_no_feedback(docs, query, feedback_log)

    st.subheader("Answer")
    answer = qa_chain.run(query)
    st.write(answer)

    st.subheader("Supporting Sources")
    feedback_collected = {"yes": [], "no": []}

    for doc in docs:
        with st.expander(f"Source: {doc['source']} - Page {doc['page']}"):
            st.write(doc["content"])
            feedback = st.radio(
                f"Is this relevant?", ["Not Rated", "Yes", "No"],
                key=f"feedback_{doc['id']}", horizontal=True
            )
            if feedback == "Yes":
                feedback_collected["yes"].append(doc["id"])
            elif feedback == "No":
                feedback_collected["no"].append(doc["id"])

    if st.button("Submit Feedback"):
        feedback_log.setdefault(query, {"yes": [], "no": []})
        for fb_type in ["yes", "no"]:
            for doc_id in feedback_collected[fb_type]:
                if doc_id not in feedback_log[query][fb_type]:
                    feedback_log[query][fb_type].append(doc_id)
        save_feedback(feedback_log, FEEDBACK_LOG_PATH)
        st.success("Feedback submitted!")
