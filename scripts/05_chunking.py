# scripts/05_chunking.py
import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
MAIN_INDEX = "arxiv-papers"
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
CHUNK_WORDS = 80      # розмір чанка в словах
OVERLAP = 20          # перекриття для fixed-size
BATCH_SIZE = 100

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


# 1. 30 статей з найдовшими анотаціями
df["n_words"] = df["abstract"].str.split().str.len()
top = df.sort_values("n_words", ascending=False).head(30).reset_index(drop=True)
print("Статей для чанкінгу:", len(top), "| слів в анотаціях:", top["n_words"].min(), "-", top["n_words"].max())


# 2. Дві стратегії чанкінгу
def fixed_chunks(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return chunks


def semantic_chunks(text: str, max_words: int = CHUNK_WORDS):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, count = [], [], 0
    for s in sentences:
        n = len(s.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(s)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_records(strategy: str, chunker):
    records = []
    for _, row in top.iterrows():
        for i, chunk in enumerate(chunker(row["abstract"])):
            records.append({
                "id": f"{row['id']}_{strategy}_{i}",
                "text": chunk,
                "meta": {
                    "arxiv_id": str(row["id"]),
                    "title": str(row["title"]),
                    "text": chunk[:500],
                    "chunk_no": i,
                    "year": int(row["year"]),
                    "category": str(row["category"]),
                },
            })
    return records


rec_fixed = build_records("fixed", fixed_chunks)
rec_sem = build_records("semantic", semantic_chunks)
print(f"Чанків fixed-size: {len(rec_fixed)} | semantic: {len(rec_sem)}")

# Для відповідей у README: скільки fixed-чанків закінчується посеред речення
cut = sum(1 for r in rec_fixed if not r["text"].rstrip().endswith((".", "!", "?")))
cut_sem = sum(1 for r in rec_sem if not r["text"].rstrip().endswith((".", "!", "?")))
print(f"Чанків, що закінчуються не на межі речення: fixed {cut}/{len(rec_fixed)}, semantic {cut_sem}/{len(rec_sem)}")

# Для відповіді про overlap: кількість чанків при різному перекритті
for ov in (0, 10, 20, 40):
    n = sum(len(fixed_chunks(a, CHUNK_WORDS, ov)) for a in top["abstract"])
    print(f"overlap={ov:>2} слів -> {n} чанків")


# 3. Індекси (якщо ліміт Starter не дозволяє, використовуємо namespace в основному індексі)
def get_target(name: str):
    try:
        if name not in pc.list_indexes().names():
            pc.create_index(
                name=name,
                dimension=VECTOR_DIM,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(name).status["ready"]:
                time.sleep(1)
            print(f"Індекс '{name}' створено.")
        return pc.Index(name), ""
    except Exception as e:
        print(f"Не вдалося створити індекс '{name}': {str(e)[:150]}")
        print(f"Використовуємо namespace '{name}' в індексі '{MAIN_INDEX}'.")
        return pc.Index(MAIN_INDEX), name


idx_fixed, ns_fixed = get_target(INDEX_FIXED)
idx_sem, ns_sem = get_target(INDEX_SEMANTIC)


# 4-5. Ембеддинги чанків і батчеве завантаження
def upload(records, index, namespace, label):
    texts = [r["text"] for r in records]
    vectors = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Завантаження {label}"):
        batch = [
            {"id": r["id"], "values": vectors[i].tolist(), "metadata": r["meta"]}
            for i, r in enumerate(records[start:start + BATCH_SIZE], start=start)
        ]
        index.upsert(vectors=batch, namespace=namespace)


upload(rec_fixed, idx_fixed, ns_fixed, "fixed")
upload(rec_sem, idx_sem, ns_sem, "semantic")
time.sleep(5)  # дати Pinecone час оновити статистику


# 6. Пошук по чанках
def search_chunks(query: str, index, namespace, label: str, top_k: int = 5):
    vec = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=vec, top_k=top_k, include_metadata=True, namespace=namespace)
    print(f"\n--- {label} ---")
    for m in res["matches"]:
        md = m["metadata"]
        print(f"{m['score']:.4f} | {md['title'][:60]} | чанк {int(md['chunk_no'])}")
        print("    ", md["text"][:150])


QUERIES = [
    "dark matter halo density profile",
    "quantum entanglement in many-body systems",
    "next-to-leading order QCD calculation",
]
for q in QUERIES:
    print(f"\n=========== Запит: {q} ===========")
    search_chunks(q, idx_fixed, ns_fixed, "fixed-size")
    search_chunks(q, idx_sem, ns_sem, "semantic")