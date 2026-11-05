import json
import re
from common import ollama_generate

class IntentModule:
    VALID = {"FAQ", "RECOMMENDATION", "PRODUCT", "REVIEW"}

    def detect(self, query):
        prompt = f"""
Classify the e-commerce user message into exactly one intent:
FAQ, RECOMMENDATION, PRODUCT, REVIEW.
FAQ: delivery, tracking, returns, payment, account, policies.
RECOMMENDATION: asks to suggest, recommend, compare or find products by need.
PRODUCT: asks details, price, brand, features or specifications of a named product.
REVIEW: asks customer opinions, ratings, sentiment or review summary.
Return only the intent word, with no explanation.

Message: {query}
Intent:
"""
        result = ollama_generate(prompt).upper()
        match = re.search(r"\b(FAQ|RECOMMENDATION|PRODUCT|REVIEW)\b", result)
        if match:
            return match.group(1)

        # Safe fallback if model output is not a valid label.
        q = query.lower()
        if any(w in q for w in ["review", "customer opinion", "sentiment", "buying suggestion"]):
            return "REVIEW"
        if any(w in q for w in ["recommend", "suggest", "best laptop", "looking for"]):
            return "RECOMMENDATION"
        if any(w in q for w in ["track", "parcel", "delivery", "return", "refund", "payment", "order"]):
            return "FAQ"
        return "PRODUCT"

if __name__ == "__main__":
    module = IntentModule()
    while True:
        q = input("Enter query (Exit to stop): ")
        if q.lower() == "exit":
            break
        print("Detected Intent:", module.detect(q))
