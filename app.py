from fastapi import FastAPI, Request
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np
import re
import os
import requests
from dotenv import load_dotenv

# ----------------------------
# LOAD ENV
# ----------------------------
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
WASENDER_TOKEN = os.getenv("WASENDER_TOKEN")

if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY nije postavljen")

if not WASENDER_TOKEN:
    raise ValueError("❌ WASENDER_TOKEN nije postavljen")

genai.configure(api_key=API_KEY)

app = FastAPI()

chunks = []
embeddings = []
sources = []
user_memory = {}

# ----------------------------
# UTILS
# ----------------------------

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0

    return float(np.dot(a, b) / denom)


def chunk_text(text, chunk_size=2000, overlap=200):
    text = re.sub(r"\s+", " ", text).strip()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def embed_texts(texts):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts
        )

        return [e["embedding"] for e in result["embedding"]]

    except Exception as e:
        print("❌ EMBEDDING ERROR:", e)
        return []


# ----------------------------
# GENERATE ANSWER
# ----------------------------

def generate_answer(context, question, user_id):

    first_time = user_id not in user_memory

    prompt = f"""
Ti si prijateljski i opušten customer support agent za webshop "TechStore".

Odgovaraj prirodno, kratko i jasno na hrvatskom, kao stvarna osoba.

PRAVILA:
- Ako odgovor postoji u kontekstu → koristi ga
- Ako NE postoji → reci da nemaš točnu info i ponudi pomoć
- Budi opušten (npr. "možeš", "nema problema", "slobodno pitaj")

KONTEKST:
{context}

PITANJE:
{question}

ODGOVOR:
"""

    try:
        response = genai.generate_content(
            model="models/gemini-1.5-flash",
            contents=prompt
        )

        answer = response.text.strip()

    except Exception as e:
        print("❌ GENERATION ERROR:", e)
        answer = "Došlo je do greške pri odgovoru, pokušaj ponovno 👍"

    # welcome poruka
    if first_time:
        answer = (
            "👋 Dobrodošli na TechStore webshop podršku!\n\n"
            + answer
        )
        user_memory[user_id] = True

    # rate limit
    answer += (
        "\n\n⚠️ Napomena: Možemo poslati jednu poruku po minuti. "
        "Ako ne dobiješ odgovor odmah, slobodno pošalji ponovno 👍"
    )

    return answer


# ----------------------------
# LOAD PDFS
# ----------------------------

def load_pdfs(folder="docs"):
    global chunks, embeddings, sources

    chunks = []
    embeddings = []
    sources = []

    if not os.path.exists(folder):
        print("⚠️ docs folder ne postoji")
        return

    try:
        for filename in os.listdir(folder):
            if filename.endswith(".pdf"):
                path = os.path.join(folder, filename)
                reader = PdfReader(path)

                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""

                c = chunk_text(text)
                e = embed_texts(c)

                for chunk, emb in zip(c, e):
                    if emb:  # sigurnost
                        chunks.append(chunk)
                        embeddings.append(emb)
                        sources.append(filename)

        print(f"✅ Loaded {len(chunks)} chunks.")

    except Exception as e:
        print("❌ ERROR LOADING PDFS:", e)


@app.on_event("startup")
def startup():
    load_pdfs()


# ----------------------------
# RAG
# ----------------------------

def run_rag(question: str, user_id: str):

    if not embeddings:
        return "⚠️ Trenutno nemam učitane informacije, ali slobodno pitaj pa ću pokušati pomoći 👍"

    q_embedding = embed_texts([question])

    if not q_embedding:
        return "⚠️ Trenutno imam problem s obradom upita, pokušaj ponovno 👍"

    q_embedding = q_embedding[0]

    scores = []
    for idx, emb in enumerate(embeddings):
        score = cosine_similarity(q_embedding, emb)
        scores.append((score, idx))

    scores.sort(reverse=True)
    top = scores[:3]

    context = "\n\n".join([chunks[idx] for _, idx in top])

    return generate_answer(context, question, user_id)


# ----------------------------
# SEND MESSAGE
# ----------------------------

def send_whatsapp_message(to, text):

    url = "https://api.wasenderapi.com/api/send-message"

    headers = {
        "Authorization": f"Bearer {WASENDER_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "to": to,
        "text": text
    }

    try:
        res = requests.post(url, json=payload, headers=headers)
        print("📤 SEND STATUS:", res.status_code, res.text)

    except Exception as e:
        print("❌ SEND ERROR:", e)


# ----------------------------
# WEBHOOK
# ----------------------------

@app.post("/webhook")
async def webhook(req: Request):

    data = await req.json()
    print("📩 RAW:", data)

    try:
        msg_obj = data.get("data", {}).get("messages", {})

        message = (
            msg_obj.get("message", {}).get("conversation")
            or msg_obj.get("messageBody")
        )

        sender = msg_obj.get("key", {}).get("remoteJid")

    except Exception as e:
        print("❌ PARSE ERROR:", e)
        return {"status": "error"}

    if not message or not sender:
        return {"status": "ignored"}

    print(f"📩 Poruka od {sender}: {message}")

    answer = run_rag(message, sender)

    send_whatsapp_message(sender, answer)

    return {"status": "ok"}