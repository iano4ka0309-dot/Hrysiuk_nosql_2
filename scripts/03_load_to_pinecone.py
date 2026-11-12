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
