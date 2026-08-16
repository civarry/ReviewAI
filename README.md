# ReviewAI

A Streamlit app that turns uploaded documents into study questions and grades your answers, built as a study-review tool for educators and students.

## What it does

1. Upload a `.txt`, `.csv`, or `.docx` file.
2. The app splits the document into chunks (`RecursiveCharacterTextSplitter`, 1000 chars with 100 char overlap), embeds them with a local `sentence-transformers/all-MiniLM-L6-v2` model, and stores them in a Chroma vector store.
3. Pick how many questions to generate (5–20) and their difficulty (Easy/Medium/Hard).
4. A retrieval-augmented generation chain queries the vector store and asks an LLM (Groq-hosted) to generate that many unique, open-ended questions grounded strictly in the uploaded document — no outside knowledge.
5. Answer the questions inline, submit, and the same RAG chain evaluates your answers against the source document and gives feedback on which are correct or incorrect.

Uploaded files are written to disk only long enough to build embeddings, then deleted.

## Stack

- **Streamlit** for the UI
- **LangChain** (`RetrievalQA`) to wire retrieval and generation together
- **ChromaDB** as the vector store
- **HuggingFace `sentence-transformers`** for local embeddings
- **Groq** (`ChatGroq`) as the LLM backend, configured via a `groq_api_key` environment variable

## Running it

```
pip install -r requirements.txt
```

Set `groq_api_key` in a `.env` file, then:

```
streamlit run main.py
```

This is the original Streamlit version of the project; a Flask-based rewrite lives at [ReviewAIFlask](https://github.com/civarry/ReviewAIFlask), built to get more control over layout and multi-user support than Streamlit allows.
