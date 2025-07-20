import streamlit as st
import requests
import os

API_URL = "http://localhost:8000"

st.title("📄 AI Document Semantic Search")

query = st.text_input("Enter your engineering query:")

if st.button("Search"):
    with st.spinner("Searching..."):
        try:
            response = requests.get(f"{API_URL}/search/", params={"query": query})
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                st.warning("No results found.")
            for r in results:
                st.write(f"**Result**: {r['text']}  \n**Score**: {r['score']:.2f}")
                feedback = st.radio(f"Is this result relevant?", ["Yes", "No"], key=r['text'])
                if st.button("Submit Feedback", key="fb_" + r['text']):
                    relevance = feedback == "Yes"
                    try:
                        fb_response = requests.post(f"{API_URL}/feedback/", params={
                            "query": query,
                            "selected_result": r['text'],
                            "relevance": relevance
                        })
                        fb_response.raise_for_status()
                        st.success("Feedback submitted.")
                    except Exception as fb_error:
                        st.error(f"Failed to submit feedback: {fb_error}")
        except Exception as e:
            st.error(f"Search failed: {e}")

if st.button("🔁 Retrain Model with Feedback"):
    with st.spinner("Retraining..."):
        try:
            retrain_response = requests.post(f"{API_URL}/retrain/")
            retrain_response.raise_for_status()
            st.success("Model retrained.")
        except Exception as retrain_error:
            st.error(f"Retraining failed: {retrain_error}")
