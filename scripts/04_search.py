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
