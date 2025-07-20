import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')
feedback_file = "data/feedback_log.csv"
index_path = "data/vector_store.faiss"

def retrain_model_from_feedback():
    try:
        df = pd.read_csv(feedback_file, header=None)
        df.columns = ["timestamp", "query", "result", "relevance"]
        df = df[df["relevance"] == True]

        queries = df["query"].tolist()
        responses = df["result"].tolist()

        new_docs = queries + responses
        new_embeddings = model.encode(new_docs)

        index = faiss.read_index(index_path)
        index.add(np.array(new_embeddings).astype("float32"))
        faiss.write_index(index, index_path)

        # Append metadata
        with open("data/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        metadata.extend(new_docs)
        with open("data/metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
    except Exception as e:
        print("Retraining failed:", e)
