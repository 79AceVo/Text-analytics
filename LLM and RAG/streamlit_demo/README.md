# NovaTech RAG Chatbot -- Streamlit Demo

BSAN 6200 | Spring 2026 | Week 15 Companion Demo

## Quick Start

### 1. Install dependencies

```
python setup.py
```

Or manually:

```
pip install -r requirements.txt
```

### 2. Set up your HuggingFace token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token
3. Check **"Make calls to Inference Providers"**
4. Copy `.env.example` to `.env` and paste your token:

```
cp .env.example .env
```

Edit `.env`:

```
HF_TOKEN=hf_your_actual_token_here
```

### 3. Run the app

```
python -m streamlit run streamlit_app.py
```

The app opens at http://localhost:8501

## Project Structure

```
streamlit_demo/
├── streamlit_app.py     # Main app (RAG chatbot)
├── requirements.txt     # Dependencies
├── setup.py             # One-click installer
├── .env.example         # Token template (copy to .env)
├── .env                 # Your actual token (DO NOT COMMIT)
├── .gitignore           # Keeps .env out of git
└── README.md            # This file
```

## How It Works

1. **Documents** are loaded into ChromaDB (in-memory vector store)
2. **User asks a question** via the chat interface
3. **Retrieval:** ChromaDB finds the 3 most relevant chunks
4. **Generation:** HuggingFace LLM answers using only the retrieved context
5. **Transparency:** Retrieved chunks are shown in expandable sections

## Troubleshooting

| Error | Fix |
|---|---|
| `HF_TOKEN not found` | Create `.env` file with your token |
| `401 Unauthorized` | Token missing "Inference Providers" permission |
| `provider unexpected keyword` | Run `pip install --upgrade huggingface-hub` |
| `streamlit not recognized` | Use `python -m streamlit run streamlit_app.py` |
