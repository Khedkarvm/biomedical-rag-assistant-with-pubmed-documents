# build_chromadb.py
import json
from pathlib import Path
import numpy as np
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_db"
PERSIST_DIR.mkdir(exist_ok=True)

EMBEDDING_FILE = BASE_DIR / "embedding" / "embedding_save.json"
if not EMBEDDING_FILE.exists():
    raise FileNotFoundError(f"Precomputed embeddings not found: {EMBEDDING_FILE}")

# load precomputed embedding
with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["content"] for item in data]
embeddings = [np.array(item["embedding"], dtype=float) for item in data]
metadatas = [
    {
        "pdf": item["metadata"].get("filename", f"doc_{i}.pdf"),  # <-- use 'filename'
        "page": item["metadata"].get("page_number", i + 1),       # <-- use 'page_number'
        "chunk_id": i
    }
    for i, item in enumerate(data)
]


ids = [f"chunk_{i}" for i in range(len(data))]

print(f"Total chunks to index: {len(texts)}")


# cretae load choroma
vectordb = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=None  # Use precomputed embeddings
)

# avoid duplication 
existing = vectordb._collection.get()
existing_ids = set(existing["ids"]) if existing["ids"] else set()

new_texts, new_embeddings, new_metadatas, new_ids = [], [], [], []

for t, e, m, i in zip(texts, embeddings, metadatas, ids):
    if i not in existing_ids:
        new_texts.append(t)
        new_embeddings.append(e)
        new_metadatas.append(m)
        new_ids.append(i)

# add to chroma
if new_texts:
    vectordb.add_texts(
        texts=new_texts,
        embeddings=new_embeddings,
        metadatas=new_metadatas,
        ids=new_ids
    )
    print(f"Indexed {len(new_texts)} new chunks into ChromaDB")
else:
    print("No new chunks to index")

print(f"ChromaDB stored at: {PERSIST_DIR.resolve()}")
print("Sample metadata:")
print(vectordb._collection.get()["metadatas"][:3])
