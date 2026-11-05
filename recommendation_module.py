from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from common import load_products, ollama_generate

class RecommendationModule:
    def __init__(self):
        self.products = load_products()
        self.products["combined"] = (
            self.products["product_name"].astype(str) + " " +
            self.products["brand"].astype(str) + " " +
            self.products["description"].astype(str)
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.product_vectors = self.vectorizer.fit_transform(self.products["combined"])

    def recommend(self, query, top_k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.product_vectors).flatten()
        indices = scores.argsort()[::-1][:min(top_k, len(scores))]
        results = self.products.iloc[indices].copy()
        results["similarity"] = scores[indices]
        results = results[results["similarity"] > 0].reset_index(drop=True)
        if results.empty:
            return "No matching products were found."
        print("\nRetrieved Products:")
        for _, p in results.iterrows():
            print(f"- {p['product_name']} | Similarity: {p['similarity']:.2f}")
        context = results[["product_name", "brand", "price", "description", "similarity"]].to_string(index=False)
        prompt = f"""
You are an e-commerce product recommendation assistant.
Recommend the single best match from the retrieved products only.
Do not invent features, prices, or specifications.
Explain briefly why it matches the user's request. If the data is insufficient,
say that clearly.

User request: {query}
Retrieved products:
{context}

Give product name, brand, price, and reason:
"""
        return ollama_generate(prompt)

if __name__ == "__main__":
    module = RecommendationModule()
    query = input("Search Product: ")
    print("\nGPT Response:\n", module.recommend(query))
