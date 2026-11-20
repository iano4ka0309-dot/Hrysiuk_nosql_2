# scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract

# TODO: твій код
# 2. Функція кодування запиту
# 3. Чистий семантичний пошук (топ-5: назва, категорія, рік, частина abstract)
# 4. Пошук з фільтрацією: A) RL, останні 5 років, cs.LG; B) до 2015 року
# 5. Порівняння метрик локально: cosine, dot product, L2 (топ-5 кожної)
# 2. Кодирование запроса
def encode_query(text: str):
    # формат "title [SEP] abstract" для запроса НЕ нужен — это обычный текст
    return model.encode(text, normalize_embeddings=True).tolist()


# 3. Семантический поиск (топ-5)
def search(query: str, top_k: int = TOP_K, filter: dict | None = None):
    vec = encode_query(query)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        filter=filter,
    )
    for m in res["matches"]:
        md = m["metadata"]
        print(f"{m['score']:.4f} | {md['title']} | {md['category']} | {int(md['year'])}")
        print("    ", md["abstract"][:200])
    return res


search("teaching machines to recognize objects in pictures")