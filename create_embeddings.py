import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from common import load_faqs, MODEL_DIR

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_faq_embeddings():
    faqs = load_faqs()
    print("Loading FAQ dataset...")
    print("FAQ records:", len(faqs))
    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        faqs["question"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    np.save(MODEL_DIR / "faq_embeddings.npy", embeddings)
    joblib.dump(faqs, MODEL_DIR / "faqs.pkl")
    joblib.dump(MODEL_NAME, MODEL_DIR / "embedding_model_name.pkl")
    print("FAQ embeddings created and saved in models/")
    print("Embedding shape:", embeddings.shape)

if __name__ == "__main__":
    create_faq_embeddings()
