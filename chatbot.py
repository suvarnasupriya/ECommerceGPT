from common import ollama_generate
from intent_module import IntentModule
from faq_module import FAQModule
from recommendation_module import RecommendationModule
from product_module import ProductModule
from review_module import ReviewModule

def main():
    print("=" * 60)
    print("       AI-POWERED E-COMMERCE CHATBOT")
    print("       Powered by Ollama + RAG")
    print("=" * 60)
    print("Initializing modules...")

    intent_module = IntentModule()
    faq_module = FAQModule(threshold=0.45)
    recommendation_module = RecommendationModule()
    product_module = ProductModule()
    review_module = ReviewModule()

    print("\nAll modules loaded successfully!")
    print("Ask your questions. Type 'exit' to stop.")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ("exit", "quit", "bye"):
            print("Assistant: Thank you for using the E-Commerce Chatbot. Goodbye!")
            break
        if not query:
            continue

        intent = intent_module.detect(query)
        print("\nDetected Intent:", intent)

        if intent == "FAQ":
            answer = faq_module.answer(query)
            if answer is None:
                print("FAQ did not meet the similarity threshold.")
                print("Assistant:", "I could not find a matching FAQ. Please rephrase your question.")
            else:
                print("\nAssistant:", answer)

        elif intent == "RECOMMENDATION":
            print("\nAssistant:", recommendation_module.recommend(query))

        elif intent == "PRODUCT":
            print("\nAssistant:", product_module.answer(query))

        elif intent == "REVIEW":
            # Extract product phrase after common review request wording.
            product_query = query
            for prefix in ["reviews for", "review for", "reviews of", "review of",
                           "summarize reviews for", "customer reviews for"]:
                if prefix in query.lower():
                    pos = query.lower().find(prefix) + len(prefix)
                    product_query = query[pos:].strip(" ?.")
                    break
            print("\nAssistant:", review_module.analyze(product_query))

if __name__ == "__main__":
    main()
