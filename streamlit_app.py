import streamlit as st
from rag_chain import rag_chain  # your RAG chain
from pathlib import Path
from langchain_chroma import Chroma
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Biomedical RAG Chat")
st.title("Biomedical Question-Answering")


uploaded_files = st.file_uploader(
    "Upload PDF(s) to add to knowledge base",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("Processing uploaded PDFs...")

    # ChromaDB path
    BASE_DIR = Path(__file__).parent
    PERSIST_DIR = BASE_DIR / "chroma_db"
    vectordb = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=None)

    # Embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for uploaded_file in uploaded_files:
        pdf_name = uploaded_file.name
        reader = PdfReader(uploaded_file)
        all_text = []

        # Extract text from pages
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                all_text.append((i + 1, text.strip()))

        # Chunk text and embed
        new_texts, new_embeddings, new_metadatas, new_ids = [], [], [], []
        for page_num, text in all_text:
            chunks = [text[j:j+1000] for j in range(0, len(text), 1000)]
            for chunk_id, chunk in enumerate(chunks):
                emb = model.encode(chunk).tolist()
                new_texts.append(chunk)
                new_embeddings.append(emb)
                new_metadatas.append({"pdf": pdf_name, "page": page_num, "chunk_id": chunk_id})
                new_ids.append(f"{pdf_name}_p{page_num}_c{chunk_id}")

        # Add to ChromaDB
        if new_texts:
            vectordb.add_texts(
                texts=new_texts,
                embeddings=new_embeddings,
                metadatas=new_metadatas,
                ids=new_ids
            )

    st.success("Uploaded PDFs have been processed and added to the knowledge base.")


# user query
user_question = st.text_input("Enter a biomedical question:")

if st.button("Ask"):
    if user_question.strip():
        with st.spinner("Fetching answer..."):
            # check RAG chain returns source documents
            response = rag_chain.invoke({
                "question": user_question,
                "return_source_documents": True  # Important
            })

        # Display answer
        st.subheader("Answer:")
        st.write(response['result'] if 'result' in response else response)

        # Display sources
        st.subheader("Sources:")
        if 'source_documents' in response:
            for doc in response['source_documents']:
                pdf_name = doc.metadata.get("pdf", "Unknown PDF")
                page_num = doc.metadata.get("page", "?")
                chunk_id = doc.metadata.get("chunk_id", "?")
                st.markdown(f"- **{pdf_name}**, page {page_num}, chunk {chunk_id}")
                st.text(doc.page_content[:300] + "...")  
        else:
            st.write("No sources returned.")
