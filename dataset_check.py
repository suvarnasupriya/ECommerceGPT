from common import load_dataset, load_products, load_faqs, load_reviews

print("\n========== PART A: DATASET PREPARATION ==========")
for title, loader in [
    ("PRODUCTS", load_products),
    ("FAQS", load_faqs),
    ("REVIEWS", load_reviews)
]:
    print(f"\n--- {title} DATASET ---")
    df = loader()
    print("\nFirst five records:")
    print(df.head())
    print("\nRows and columns:", df.shape)
    print("\nAttributes:", list(df.columns))
    print("\nData types:")
    print(df.dtypes)
print("\nDataset loading completed successfully.")
