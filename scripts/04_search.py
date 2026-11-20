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
    if not res["matches"]:
        print("    (порожній результат)")
    for m in res["matches"]:
        md = m["metadata"]
        print(f"{m['score']:.4f} | {md['title']} | {md['category']} | {int(md['year'])}")
        print("    ", md["abstract"][:200])
    return res

print("\n=== Без фільтра: reinforcement learning ===")
search("reinforcement learning")

print("\n=== A: RL, з 2021 року, cs.LG ===")
search("reinforcement learning", filter={"year": {"$gte": 2021}, "category": {"$eq": "cs.LG"}})

print("\n=== B: до 2015 року ===")
search("reinforcement learning", filter={"year": {"$lt": 2015}})

# 5. Порівняння метрик локально
embeddings = np.load("embeddings/embeddings.npy")
q = np.array(encode_query("teaching machines to recognize objects in pictures"))

cos = (embeddings @ q) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q))
dot = embeddings @ q
l2 = np.linalg.norm(embeddings - q, axis=1)

# Для схожості (cosine, dot) найкращі найбільші значення, для відстані (L2) найменші
top_cos = np.argsort(-cos)[:5]
top_dot = np.argsort(-dot)[:5]
top_l2 = np.argsort(l2)[:5]

for name, top, scores in [("cosine", top_cos, cos), ("dot product", top_dot, dot), ("L2", top_l2, l2)]:
    print(f"\n=== {name} ===")
    for i in top:
        print(f"{scores[i]:.4f} | {df.iloc[i]['title']}")

print("\nТоп-5 збігаються (cosine vs dot):", list(top_cos) == list(top_dot))
print("Топ-5 збігаються (cosine vs L2):", list(top_cos) == list(top_l2))