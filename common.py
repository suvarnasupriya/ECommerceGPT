from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def clean_col(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

def load_dataset(stem):
    """Load stem.xlsx or stem.csv from data/. Also detects a single matching file."""
    candidates = [DATA_DIR / f"{stem}.xlsx", DATA_DIR / f"{stem}.csv",
                  DATA_DIR / f"{stem}.xls"]
    for path in candidates:
        if path.exists():
            if path.suffix == ".csv":
                return pd.read_csv(path)
            return pd.read_excel(path, engine="openpyxl" if path.suffix == ".xlsx" else None)

    matches = list(DATA_DIR.glob(f"{stem}.*"))
    if matches:
        path = matches[0]
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path)
    raise FileNotFoundError(
        f"Could not find {stem}.xlsx or {stem}.csv inside {DATA_DIR}. "
        f"Copy your {stem} dataset into the data folder."
    )

def normalize_columns(df):
    df = df.copy()
    df.columns = [clean_col(c) for c in df.columns]
    return df

def find_column(df, aliases, required=True, label="column"):
    cols = list(df.columns)
    for alias in aliases:
        key = clean_col(alias)
        if key in cols:
            return key
    # allow partial match
    for col in cols:
        if any(clean_col(alias) in col or col in clean_col(alias) for alias in aliases):
            return col
    if required:
        raise ValueError(
            f"Cannot identify {label}. Found columns: {cols}. "
            f"Expected one of: {aliases}"
        )
    return None

def load_products():
    df = normalize_columns(load_dataset("products"))
    name = find_column(df, ["product_name", "name", "product", "title"], label="product name")
    brand = find_column(df, ["brand", "manufacturer"], required=False)
    price = find_column(df, ["price", "cost", "amount"], required=False)
    desc = find_column(df, ["description", "product_description", "details", "features"], required=False)
    out = pd.DataFrame()
    out["product_name"] = df[name].fillna("").astype(str)
    out["brand"] = df[brand].fillna("").astype(str) if brand else ""
    out["price"] = df[price] if price else ""
    out["description"] = df[desc].fillna("").astype(str) if desc else ""
    out = out[out["product_name"].str.strip() != ""].reset_index(drop=True)
    return out

def load_faqs():
    df = normalize_columns(load_dataset("faqs"))
    q = find_column(df, ["question", "faq_question", "query"], label="FAQ question")
    a = find_column(df, ["answer", "faq_answer", "response"], label="FAQ answer")
    out = pd.DataFrame({
        "question": df[q].fillna("").astype(str),
        "answer": df[a].fillna("").astype(str)
    })
    return out[(out.question.str.strip() != "") & (out.answer.str.strip() != "")].reset_index(drop=True)

def load_reviews():
    df = normalize_columns(load_dataset("reviews"))
    p = find_column(df, ["product_name", "product", "name", "product_title"], label="review product name")
    r = find_column(df, ["review", "review_text", "comment", "text", "customer_review"], label="review text")
    out = pd.DataFrame({
        "product_name": df[p].fillna("").astype(str),
        "review": df[r].fillna("").astype(str)
    })
    return out[(out.product_name.str.strip() != "") & (out.review.str.strip() != "")].reset_index(drop=True)

def ollama_generate(prompt, model="llama3.2:3b"):
    import requests
    url = "http://localhost:11434/api/generate"
    try:
        response = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as exc:
        return f"[Ollama error: {exc}. Check that Ollama is running and model '{model}' is installed.]"
