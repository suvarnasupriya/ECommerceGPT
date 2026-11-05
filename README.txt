AI-POWERED E-COMMERCE CHATBOT USING OLLAMA

1. Copy your three dataset files into the data folder:
   products.xlsx
   faqs.xlsx
   reviews.xlsx

2. Expected columns (the code also recognizes common alternatives):
   products: Product_Name, Brand, Price, Description
   faqs: Question, Answer
   reviews: Product_Name, Review

3. Open PowerShell in the ECommerceGPT folder.

4. Install dependencies:
   py -m pip install -r requirements.txt

5. Install Ollama from https://ollama.com/download
   Then run:
   ollama pull llama3.2:3b

6. Check datasets:
   py dataset_check.py

7. Create FAQ embeddings:
   py create_embeddings.py

8. Run the complete chatbot:
   py chatbot.py

9. Type exit to stop.

IMPORTANT:
- Keep Ollama running while using the chatbot.
- First run may download embedding and sentiment models.
- Similarity threshold is set to 0.45 in chatbot.py. Adjust after testing.
- Outputs depend on your actual dataset records.
- If your dataset columns differ, common.py displays the columns it found
  and reports which expected field could not be identified.
