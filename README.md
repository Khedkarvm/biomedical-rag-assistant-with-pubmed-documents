# Biomedical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions from biomedical research documents and clinical trial PDFs using document retrieval, semantic search, and a large language model.

## Overview

The Biomedical RAG Chatbot allows users to ask questions about biomedical documents without manually searching through lengthy research papers and clinical trial PDFs.

The system processes documents, extracts their text, creates semantic embeddings, stores the embeddings in a vector database, retrieves relevant document chunks for a user query, and generates an answer using an LLM.

The chatbot also provides source references to improve the traceability of generated answers.

---

## Key Features

* Question answering over biomedical research documents
* Retrieval-Augmented Generation (RAG)
* PDF document processing
* OCR support for scanned documents
* Text chunking for document retrieval
* Semantic search using embeddings
* ChromaDB vector database
* LLM-based answer generation
* Source/citation support
* Context-aware document retrieval

---

## RAG Architecture

```text
Biomedical PDFs
      ↓
PDF Text Extraction
      ↓
OCR for Scanned Pages
      ↓
Text Cleaning
      ↓
Text Chunking
      ↓
Embedding Model
      ↓
ChromaDB
      ↓
User Question
      ↓
Query Embedding
      ↓
Semantic Retrieval
      ↓
Relevant Document Chunks
      ↓
LLM
      ↓
Answer + Sources
```

---

## How It Works

### 1. Document Ingestion

Biomedical research papers and clinical trial documents in PDF format are provided as the knowledge source.

### 2. Text Extraction

Text is extracted from the PDFs.

For scanned or image-based documents, OCR is used to extract readable text.

The project uses:

* `pdf2image`
* `PyTesseract`

for processing scanned PDF content.

### 3. Text Chunking

Extracted text is divided into smaller chunks so that relevant sections can be retrieved efficiently.

The project uses approximately:

```text
Chunk Size: 500
Chunk Overlap: 100
```

This helps maintain context between neighboring sections.

### 4. Embeddings

The document chunks are converted into vector representations using a sentence-transformer embedding model.

Embedding model:

```text
all-MiniLM-L6-v2
```

The generated embeddings have a dimension of 384.

### 5. Vector Database

The embeddings and corresponding document chunks are stored in **ChromaDB**.

When a user asks a question, the query is converted into an embedding and compared with stored document embeddings to retrieve the most relevant information.

### 6. Retrieval-Augmented Generation

The retrieved document chunks are provided as context to the LLM.

The model generates an answer based on the retrieved biomedical information rather than relying only on its pretrained knowledge.

### 7. Source References

Relevant document sources are included with the generated response to make the answer easier to trace back to the original documents.

---

## Technology Stack

| Category        | Technologies              |
| --------------- | ------------------------- |
| Programming     | Python                    |
| RAG Framework   | LangChain                 |
| Embeddings      | all-MiniLM-L6-v2          |
| Vector Database | ChromaDB                  |
| LLM             | Gemini                    |
| PDF Processing  | pdf2image, PyTesseract    |
| Data Processing | NumPy, Pandas             |
| Development     | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub               |

---

## Project Structure

```text
Biomedical-RAG-Chatbot/
│
├── data/
│   └── documents/
│
├── src/
│   ├── document_loader.py
│   ├── text_processing.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   └── ...
│
├── chroma_db/
│
├── notebooks/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

> Update the file names above if your actual project structure uses different names.

---

## Example Questions

The chatbot can be used to ask questions such as:

```text
What is the purpose of this clinical trial?

What are the eligibility criteria?

What treatment is being studied?

What are the reported outcomes?

What are the study conditions?

What is the study methodology?
```

The actual questions depend on the biomedical documents included in the knowledge base.

---

## Example RAG Workflow

```text
User:
"What are the eligibility criteria for this clinical trial?"

              ↓

Convert question into embedding

              ↓

Search ChromaDB

              ↓

Retrieve relevant document chunks

              ↓

Send retrieved context + question to LLM

              ↓

Generate answer

              ↓

Return answer with source reference
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Biomedical-RAG-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file and add the required API key:

```env
GEMINI_API_KEY=your_api_key
```

Do not commit API keys or other sensitive information to GitHub.

### 6. Run the application

If the project uses Streamlit:

```bash
streamlit run app.py
```

---

## Key Learning

Through this project, I worked with:

* Retrieval-Augmented Generation
* Biomedical document processing
* PDF text extraction
* OCR for scanned documents
* Text chunking and preprocessing
* Semantic embeddings
* Vector databases
* ChromaDB
* LangChain retrieval pipelines
* LLM-based question answering
* Source-aware response generation

---

## Future Improvements

* Improve retrieval accuracy using hybrid search
* Add metadata-based filtering
* Support larger biomedical document collections
* Improve citation and source tracking
* Add document-level summarization
* Evaluate retrieval and generation quality using RAG evaluation metrics
* Add a more advanced user interface

---

## Disclaimer

This project is intended for **educational and research purposes**.

The chatbot provides information based on the documents included in its knowledge base and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## Author

**Vaishnavi Khedkar**

M.Sc. Industrial Mathematics with Computer Applications

Interested in **AI/ML, Generative AI, LLMs, RAG, and Data Science**.
