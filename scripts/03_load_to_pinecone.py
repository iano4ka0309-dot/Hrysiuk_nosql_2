# scripts/03_load_to_pinecone.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# TODO: твій код
# 1. Створити індекс arxiv-papers, якщо не існує (яку метрику обрати?)
# 2. Прочитати parquet і embeddings.npy
# 3. Батчі по 200: id "paper_<номер>", values, metadata:
#    arxiv_id, title, abstract[:500], authors[:200], year, category
# 4. upsert з прогресом (tqdm)
# 5. Вивести загальну кількість векторів в індексі
# Створюємо індекс, якщо його ще немає
# scripts/03_load_to_pinecone.py


 

# 1. Створюємо індекс, якщо його ще немає
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Індекс '{INDEX_NAME}' створено.")
else:
    print(f"Індекс '{INDEX_NAME}' уже існує.")

index = pc.Index(INDEX_NAME)

# 2. Читаємо дані та ембеддинги
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)
print("Статей:", len(df), "| Ембеддинги:", embeddings.shape)

# 3-4. Завантажуємо батчами з прогресом
for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження в Pinecone"):
    end = min(start + BATCH_SIZE, len(df))
    batch = []
    for i in range(start, end):
        row = df.iloc[i]
        batch.append({
            "id": f"paper_{i}",
            "values": embeddings[i].tolist(),
            "metadata": {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"])[:500],
                "authors": str(row["authors"])[:200],
                "year": int(row["year"]),
                "category": str(row["category"]),
            },
        })
    index.upsert(vectors=batch)

# 5. Підсумок
time.sleep(5)  # статистика оновлюється із невеликою затримкою
stats = index.describe_index_stats()
print("Усього векторів в індексі:", stats["total_vector_count"])