# scripts/07_add_recent.py
# Додає до індексу arxiv-papers свіжі статті cs.LG (2021+),
# щоб показати роботу фільтрів у Частині 3.
import json
import os
import random

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

INPUT_FILE = "arxiv-metadata-oai-snapshot.json"
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
START_ID = 10_000      # нумерація продовжує paper_0 ... paper_9999
N_SAMPLE = 600
MIN_YEAR = 2021
BATCH_SIZE = 200
random.seed(42)

os.makedirs("data", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)


def extract_year(paper: dict) -> int:
    versions = paper.get("versions", [])
    if versions:
        try:
            return int(versions[0]["created"].split()[3])
        except (IndexError, ValueError, KeyError):
            pass
    return int(paper.get("update_date", "2000-01-01")[:4])


def format_authors(paper: dict) -> str:
    parsed = paper.get("authors_parsed", [])
    parts = []
    for entry in parsed[:10]:
        last = entry[0].strip() if len(entry) > 0 else ""
        first = entry[1].strip() if len(entry) > 1 else ""
        if last:
            parts.append(f"{last} {first}".strip())
    return ", ".join(parts) if parts else paper.get("authors", "").replace("\n", " ")


# 1. Резервуарна вибірка: рівномірно випадкові N_SAMPLE статей cs.LG за 2021+
sample = []
seen = 0
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Читаємо датасет"):
        if "cs.LG" not in line:      # швидка попередня перевірка:      # швидка попередня перевірка
            continue
        paper = json.loads(line)
        if paper.get("categories", "").split()[0] != "cs.LG":
            continue
        year = extract_year(paper)
        if year < MIN_YEAR:
            continue
        abstract = paper.get("abstract", "").replace("\n", " ").strip()
        title = paper.get("title", "").replace("\n", " ").strip()
        if not abstract or not title:
            continue
        record = {
            "id": paper["id"],
            "title": title,
            "abstract": abstract,
            "authors": format_authors(paper),
            "year": year,
            "category": "cs.LG",
        }
        seen += 1
        if len(sample) < N_SAMPLE:
            sample.append(record)
        else:
            j = random.randint(0, seen - 1)
            if j < N_SAMPLE:
                sample[j] = record

df = pd.DataFrame(sample)
if not sample:
    raise SystemExit("Не знайдено жодної статті cs.LG за 2021+. Перевір формат файлу.")
print(f"Знайдено кандидатів: {seen}, відібрано: {len(df)}")
print(df["year"].value_counts().sort_index())
df.to_parquet("data/arxiv_recent.parquet", index=False)

# 2. Ембеддинги (той самий формат і нормалізація, що й у скрипті 02)
model = SentenceTransformer(MODEL_NAME)
texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()
embeddings = model.encode(
    texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
)
np.save("embeddings/embeddings_recent.npy", embeddings)
print("Ембеддинги:", embeddings.shape)

# 3. Завантаження в той самий індекс з новими id
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження в Pinecone"):
    end = min(start + BATCH_SIZE, len(df))
    batch = []
    for i in range(start, end):
        row = df.iloc[i]
        batch.append({
            "id": f"paper_{START_ID + i}",
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

print("Готово. Статистика індексу:", index.describe_index_stats())