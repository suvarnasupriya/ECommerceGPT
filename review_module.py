from transformers import pipeline
from common import load_reviews, ollama_generate

class ReviewModule:
    def __init__(self):
        self.reviews = load_reviews()
        print("Loading pre-trained sentiment analysis model...")
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

    def analyze(self, product_name):
        matched = self.reviews[
            self.reviews["product_name"].str.contains(product_name, case=False, na=False)
        ]
        if matched.empty:
            return f"No reviews found for {product_name}."

        texts = matched["review"].tolist()
        predictions = self.sentiment_model(texts, truncation=True)
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        detail_lines = []
        for text, pred in zip(texts, predictions):
            label = pred["label"].lower()
            if label not in counts:
                label = "neutral"
            counts[label] += 1
            detail_lines.append(
                f"Review: {text}\nSentiment: {label.title()}\nConfidence: {pred['score']:.3f}"
            )

        print("\nSentiment Counts:", counts)
        print("\nDetailed Reviews:")
        print("\n------------------\n".join(detail_lines))
        prompt = f"""
You are an e-commerce review analyst.
Summarize only the evidence in these customer reviews. Mention common strengths,
weaknesses, and a balanced buying suggestion. Do not claim that reviews prove
product quality. Do not invent comments.

Product: {product_name}
Sentiment counts: {counts}
Reviews:
{chr(10).join(detail_lines)}

Write:
Overall Opinion
Strengths
Weaknesses
Buying Suggestion
"""
        return ollama_generate(prompt)

if __name__ == "__main__":
    module = ReviewModule()
    name = input("Enter Product: ")
    print("\nGPT Summary:\n", module.analyze(name))
