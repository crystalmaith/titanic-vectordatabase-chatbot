# ⚓ Titanic Vector Explorer — Coast & Calm

A semantic RAG chatbot powered by OpenAI embeddings + GPT-4o, themed with the **Coast & Calm** palette.

## Architecture

```
titanic.csv
    │
    ▼
Document Builder  →  418 passenger rows  +  ~20 aggregate stat chunks
    │
    ▼
OpenAI text-embedding-3-small  →  NumPy matrix (in-memory vector store)
    │
    ▼ (at query time)
Cosine Similarity Retrieval (top-8 chunks)
    │
    ▼
GPT-4o  →  Grounded natural-language answer
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. Enter your OpenAI API key in the sidebar
2. Click **Build Vector DB** (one-time embedding, ~30s)
3. Ask anything!

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to share.streamlit.io → New app → select `app.py`
3. Done! Users enter their own API key in the UI

## Example Questions

- "What was the survival rate for women vs men?"
- "Who paid the most expensive ticket?"
- "Show a histogram of passenger ages"
- "How many passengers embarked from Southampton?"
- "Tell me about the youngest passenger"
- "What % survived in first class vs third class?"
- "Show survival rate by gender as a bar chart"
- "What was the average fare for each class?"
