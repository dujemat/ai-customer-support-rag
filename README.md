# 🤖 WhatsApp AI Customer Support (RAG)

AI-powered WhatsApp customer support bot built with **FastAPI + Gemini +
RAG (PDF knowledge base)**.

------------------------------------------------------------------------

## 🚀 Features

-   📱 WhatsApp integration via Wasender API
-   🤖 AI responses powered by Gemini (Google)
-   📚 RAG (Retrieval-Augmented Generation) using PDF documents
-   🧠 Context-aware answers based on uploaded docs
-   👋 First-time user greeting + memory
-   ⚡ Rate-limit friendly responses (1 message/min handling)

------------------------------------------------------------------------

## 🧱 Tech Stack

-   FastAPI
-   Google Gemini API
-   NumPy (cosine similarity)
-   PyPDF (document parsing)
-   Railway (deployment)
-   Wasender (WhatsApp API)

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── app.py
    ├── requirements.txt
    └── runtime.txt
    ├── .env
    ├── docs/           # PDF knowledge base
    └── README.md
    └── prompts/
    └── support_prompt.py

------------------------------------------------------------------------

## ⚙️ Setup (Local)

### 1. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

### 2. Create `.env`

``` env
GEMINI_API_KEY=your_key_here
WASENDER_TOKEN=your_token_here
```

------------------------------------------------------------------------

### 3. Run server

``` bash
uvicorn app:app --reload
```

------------------------------------------------------------------------

## 🌍 Deployment (Railway)

1.  Push project to GitHub\
2.  Deploy via Railway\
3.  Add environment variables:
    -   `GEMINI_API_KEY`
    -   `WASENDER_TOKEN`
4.  Set start command:

``` bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

------------------------------------------------------------------------

## 📚 How RAG Works

1.  PDFs are loaded from `/docs`
2.  Text is chunked
3.  Embeddings are generated via Gemini
4.  User question is embedded
5.  Top relevant chunks are retrieved
6.  AI generates answer based on context

------------------------------------------------------------------------

## ⚠️ Limitations

-   Free plan: \~1 message per minute (rate limit)
-   Depends on Wasender API availability
-   No persistent database (memory is in-process only)

------------------------------------------------------------------------

## 💡 Future Improvements

-   Dashboard for uploading documents
-   Multi-user support
-   Conversation history (persistent)
-   Better flow-based responses (menu system)
-   Admin panel

------------------------------------------------------------------------

## 🧑‍💻 Author

Built as a practical AI + WhatsApp integration project.

------------------------------------------------------------------------