# scripts/02_embed.py
# Кроки з завдання (1.3):
# 1. Завантажити data/arxiv_subset.parquet (pandas)
# 2. Тексти: title + " [SEP] " + abstract
# 3-4. SentenceTransformer("allenai/specter2_base"), batch_size=64,
#      show_progress_bar=True, normalize_embeddings=True
# 5. Вивести: кількість текстів, розмірність (768), норму першого вектора (~1.0)
# 6-7. Створити папку embeddings/ і зберегти embeddings/embeddings.npy

# TODO: твій код
# scripts/02_embed.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PARQUET = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"

# 1. Читаємо дані
df = pd.read_parquet(INPUT_PARQUET)

# 2. Тексти у форматі: title + " [SEP] " + abstract
texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

# 3. Модель
model = SentenceTransformer(MODEL_NAME)

# 4. Кодування
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

# 5. Вивід
print("Кількість текстів:", len(embeddings))
print("Розмірність:", embeddings.shape[1])
print("Норма першого вектора:", np.linalg.norm(embeddings[0]))

# 6-7. Зберігаємо
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(OUTPUT_FILE, embeddings)
print("Збережено в", OUTPUT_FILE)