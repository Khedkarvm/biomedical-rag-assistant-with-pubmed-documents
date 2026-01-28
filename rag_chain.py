# rag_chain.py
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

# Base directories
BASE_DIR = Path(__file__).parent
PERSIST_DIR = BASE_DIR / "chroma_db"

# Load ChromaDB
vectordb = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=None  # Using precomputed embeddings
)

# Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# LLM (replace with a valid model you have quota for)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-lite-latest",  
    temperature=0,
    google_api_key=api_key
)

# Prompt template with source attribution
biomed_prompt = ChatPromptTemplate.from_template(
"""
You are a biomedical expert assistant.

Rules:
- Answer the question using ONLY the information provided in the context.
- Group statements from the same PDF and page together.
- Place the citation at the end of the group on its own line.
- Citation format must be:
  **[Source: <PDF> | Page: <PAGE>]**
- Include a blank line after each citation.
- If the answer is not present, respond:
  "The provided documents do not contain the answer."
- Use clear, concise, professional language.

Context:
{context}

Question:
{question}
"""
)

# Format retrieved documents with grouped sources
def format_docs_grouped_by_source(docs):
    if not docs:
        return "No context available from documents."

    # Ensure docs are Document objects
    documents = [
        doc if isinstance(doc, Document)
        else Document(page_content=doc["text"], metadata=doc.get("metadata", {}))
        for doc in docs
    ]

    # Group by PDF and page
    grouped = {}
    for doc in documents:
        meta = doc.metadata
        key = (meta.get('pdf', 'unknown'), meta.get('page', 'unknown'))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(doc.page_content.strip())

    # Combine statements and add citation
    formatted = []
    for (pdf, page), texts in grouped.items():
        combined_text = " ".join(texts)
        formatted.append(f"{combined_text}\n**[Source: {pdf} | Page: {page}]**\n\n")

    return "\n".join(formatted)

# RAG Chain
rag_chain = (
    {
        "context": (
            RunnableLambda(lambda x: x["question"])
            | retriever
            | RunnableLambda(format_docs_grouped_by_source)
        ),
        "question": RunnableLambda(lambda x: x["question"])
    }
    | biomed_prompt
    | llm
    | StrOutputParser()
)

# Example query for testing
if __name__ == "__main__":
    query = "Who were the participants and what interventions did they receive in the 3-phase crossover trial?"
    response = rag_chain.invoke({"question": query})
    print("\n--- RAG Output with Grouped Source Attribution ---\n")
    print(response)
