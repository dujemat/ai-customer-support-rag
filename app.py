from fastapi import FastAPI, Request
import google.generativeai as genai
from pypdf import PdfReader
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
user_memory = {}

# ----------------------------
# UTILS
# ----------------------------

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


# ----------------------------
# GENERATE ANSWER
# ----------------------------

def generate_answer(context, question, user_id):

    first_time = user_id not in user_memory

    prompt = f"""
Ti si prijateljski i opušten customer support agent za webshop "TechStore".

Odgovaraj prirodno i kratko na hrvatskom.

Ako nemaš info iz konteksta, pokušaj pomoći općenito.

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
        answer = "Ups, nešto je zapelo 😅 Aj probaj ponovno."

    if first_time:
        answer = "👋 Dobrodošli na TechStore podršku!\n\n" + answer
        user_memory[user_id] = True

    answer += "\n\n⚠️ Možemo poslati jednu poruku po minuti. Ako ne dođe odmah, probaj opet 👍"

    return answer


# ----------------------------
# LOAD PDFS
# ----------------------------

def load_pdfs(folder="docs"):
    global chunks

    chunks = []

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

                chunks.extend(chunk_text(text))

        print(f"✅ Loaded {len(chunks)} chunks.")

    except Exception as e:
        print("❌ ERROR LOADING PDFS:", e)


@app.on_event("startup")
def startup():
    load_pdfs()


# ----------------------------
# SIMPLE RAG
# ----------------------------

def run_rag(question: str, user_id: str):

    if not chunks:
        return generate_answer("", question, user_id)

    words = question.lower().split()

    scored = []
    for chunk in chunks:
        score = sum(word in chunk.lower() for word in words)
        scored.append((score, chunk))

    scored.sort(reverse=True)

    top_chunks = [c for s, c in scored if s > 0][:3]
    context = "\n\n".join(top_chunks)

    return generate_answer(context, question, user_id)


# ----------------------------
# SEND MESSAGE
# ----------------------------

def send_whatsapp_message(to, text):

    print("📤 SENDING TO:", to)

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
        print("📤 STATUS:", res.status_code, res.text)

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
        messages = data.get("data", {}).get("messages", [])

        # 🔥 FIX: može biti lista
        if isinstance(messages, list) and len(messages) > 0:
            msg_obj = messages[0]
        else:
            msg_obj = messages

        message = (
            msg_obj.get("message", {}).get("conversation")
            or msg_obj.get("messageBody")
        )

        sender = msg_obj.get("key", {}).get("remoteJid")

    except Exception as e:
        print("❌ PARSE ERROR:", e)
        return {"status": "error"}

    print("👉 MESSAGE:", message)
    print("👉 SENDER:", sender)

    if not message or not sender:
        print("⚠️ EMPTY MESSAGE")
        return {"status": "ignored"}

    try:
        answer = run_rag(message, sender)
    except Exception as e:
        print("❌ RAG ERROR:", e)
        answer = "Ups, nešto je zapelo 😅"

    send_whatsapp_message(sender, answer)

    return {"status": "ok"}


# ----------------------------
# ROOT (da nema 404)
# ----------------------------

@app.get("/")
def home():
    return {"status": "alive"}