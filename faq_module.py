import numpy as np
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from common import MODEL_DIR, ollama_generate

class FAQModule:
    def __init__(self, threshold=0.45):
        emb_path = MODEL_DIR / "faq_embeddings.npy"
        faq_path = MODEL_DIR / "faqs.pkl"
        if not emb_path.exists() or not faq_path.exists():
            raise FileNotFoundError("FAQ embeddings are missing. Run: py create_embeddings.py")
        self.faqs = joblib.load(faq_path)
        self.embeddings = np.load(emb_path)
        model_name = joblib.load(MODEL_DIR / "embedding_model_name.pkl")
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def answer(self, query):
        query_vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = self.embeddings @ query_vector
        best_index = int(np.argmax(scores))
        score = float(scores[best_index])
        row = self.faqs.iloc[best_index]
        print("\nBest Matching FAQ:", row["question"])
        print(f"Similarity Score: {score:.2f}")
        if score < self.threshold:
            print("FAQ similarity is below threshold. Passing to next module...")
            return None
        prompt = f"""
You are an e-commerce customer support assistant.
Answer the customer's question using only the FAQ answer below.
Do not invent order details or policies. If the answer is incomplete, say so.

Customer question: {query}
Retrieved FAQ question: {row['question']}
Retrieved FAQ answer: {row['answer']}

Write a clear, friendly answer:
"""
        return ollama_generate(prompt)

if __name__ == "__main__":
    print("Loading FAQ Module...")
    module = FAQModule()
    q = input("Ask Question: ")
    result = module.answer(q)
    print("\nChatbot:")
    print(result if result else "No suitable FAQ found.")
