# scripts/02_embed.py
# Кроки з завдання (1.3):
# 1. Завантажити data/arxiv_subset.parquet (pandas)
# 2. Тексти: title + " [SEP] " + abstract
# 3-4. SentenceTransformer("allenai/specter2_base"), batch_size=64,
#      show_progress_bar=True, normalize_embeddings=True
# 5. Вивести: кількість текстів, розмірність (768), норму першого вектора (~1.0)
# 6-7. Створити папку embeddings/ і зберегти embeddings/embeddings.npy

# TODO: твій код
