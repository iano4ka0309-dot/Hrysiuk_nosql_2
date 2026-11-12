# scripts/05_chunking.py
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# TODO: твій код
# 1. 30 статей з найдовшими анотаціями
# 2. Fixed-size chunking (слова + overlap) і semantic chunking (речення до ліміту слів)
# 3. Індекси arxiv-chunks-fixed і arxiv-chunks-semantic
#    (увага: Starter-тір може обмежувати кількість індексів!)
# 4. Ембеддинги чанків + метадані: arxiv_id, title, текст чанка, номер чанка, рік, категорія
# 5. Батчеве завантаження з прогресом
# 6. Пошук по чанках: кілька запитів, топ-5 для кожного типу
