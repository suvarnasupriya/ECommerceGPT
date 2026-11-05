from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from common import load_products, ollama_generate

class ProductModule:
    def __init__(self):
        self.products = load_products()
        self.products["combined"] = (
            self.products["product_name"].astype(str) + " " +
            self.products["brand"].astype(str) + " " +
            self.products["description"].astype(str)
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectors = self.vectorizer.fit_transform(self.products["combined"])
        self.last_product = None

    def answer(self, query):
        # Follow-up questions such as "What is its price?" use the last product.
        follow_up_words = ["its price", "what is the price", "what brand", "how much", "tell me more"]
        if self.last_product is not None and any(w in query.lower() for w in follow_up_words):
            product = self.last_product
            score = 1.0
        else:
            qv = self.vectorizer.transform([query])
            scores = cosine_similarity(qv, self.vectors).flatten()
            idx = int(scores.argmax())
            score = float(scores[idx])
            if score <= 0:
                return "I could not find that product in the product dataset."
            product = self.products.iloc[idx]
            self.last_product = product

        context = (
            f"Product name: {product['product_name']}\n"
            f"Brand: {product['brand']}\n"
            f"Price: {product['price']}\n"
            f"Description: {product['description']}"
        )
        print("\nRetrieved Product:", product["product_name"])
        print(f"Similarity: {score:.2f}")
        prompt = f"""
You are an e-commerce product information assistant.
Answer only using the product context below. Never invent missing details.
If the user asks a follow-up, use the same product context.
If a requested detail is absent, say it is not available in the dataset.

Product context:
{context}

Conversation question: {query}
Answer clearly and briefly:
"""
        return ollama_generate(prompt)

if __name__ == "__main__":
    module = ProductModule()
    while True:
        q = input("\nAsk about a product (or Exit): ")
        if q.lower().strip() == "exit":
            break
        print("\nAssistant:", module.answer(q))
