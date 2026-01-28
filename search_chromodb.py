import json
from pathlib import Path
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_db"
OUTPUT_FILE = BASE_DIR / "searching_results.json"

# load chromadb
vectordb = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=None  # already precomputed
)


# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def search_and_save(query_text, top_k=5):
    # Convert query text to embedding
    query_embedding = model.encode(query_text).tolist()

    # Search in ChromaDB
    results = vectordb.similarity_search_by_vector(
        embedding=query_embedding,
        k=top_k
    )

    # Prepare output
    output = []
    for doc in results:
        meta = doc.metadata
        output.append({
            "pdf": meta.get("pdf", "unknown"),
            "page": meta.get("page", "unknown"),
            "chunk_id": meta.get("chunk_id", "unknown"),
            "text": doc.page_content
        })

    # Save results to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Top {top_k} results saved to {OUTPUT_FILE}")


# example 
if __name__ == "__main__":
    query = "What are the effects of oxygen on preterm infants?"  # <-- Your search text here
    search_and_save(query, top_k=3)
