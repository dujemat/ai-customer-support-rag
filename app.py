from fastapi import FastAPI, Request
from google import genai
from pypdf import PdfReader
import numpy as np
import re
import os
import requests
from dotenv import load_dotenv
from prompts.support_prompt import build_prompt

# ----------------------------
# MEMORY
# ----------------------------
user_memory = {}
chat_history = {}

def get_history(user_id):
    return chat_history.get(user_id, [])

def save_history(user_id, question, answer):
    if user_id not in chat_history:
        chat_history[user_id] = []

    chat_history[user_id].append({
        "q": question,
        "a": answer
    })

    chat_history[user_id] = chat_history[user_id][-5:]


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

client = genai.Client(api_key=API_KEY)

app = FastAPI()

chunks = []
embeddings = []
sources = []

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
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    return [e.values for e in result.embeddings]


# ----------------------------
# GENERATE ANSWER
# ----------------------------
def generate_answer(context, question, user_id):

    first_time = user_id not in user_memory

    history = get_history(user_id)

    history_text = "\n".join([
        f"Korisnik: {h['q']}\nBot: {h['a']}"
        for h in history
    ])

    prompt = build_prompt(context, question, history_text)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = response.text.strip()

    except Exception as e:
        print("❌ GENERATION ERROR:", e)
        answer = "Ups, nešto je zapelo 😅 Aj probaj ponovno."

    if first_time:
        answer = "👋 Dobrodošli na TechStore webshop podršku!\n\n" + answer
        user_memory[user_id] = True

    answer += (
        "\n\n⚠️ *Napomena*: Možemo poslati jednu poruku po minuti. "
        "Ako ne dobiješ odgovor odmah, slobodno pošalji ponovno."
    )

    save_history(user_id, question, answer)

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
                chunks.append(chunk)
                embeddings.append(emb)
                sources.append(filename)

    print(f"✅ Loaded {len(chunks)} chunks.")


@app.on_event("startup")
def startup():
    load_pdfs()


# ----------------------------
# RAG
# ----------------------------
def run_rag(question: str, user_id: str):

    if not embeddings:
        return "Trenutno nemam učitane podatke 😅"

    history = get_history(user_id)
    history_text = " ".join([h["q"] for h in history])

    q_embedding = embed_texts([history_text + " " + question])[0]

    scores = []
    for idx, emb in enumerate(embeddings):
        score = cosine_similarity(q_embedding, emb)
        scores.append((score, idx))

    scores.sort(reverse=True)

    # filtriraj loše matchove
    top = [item for item in scores if item[0] > 0.5][:3]

    if not top:
        return generate_answer("", question, user_id)

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

    res = requests.post(url, json=payload, headers=headers)

    print("📤 SEND STATUS:", res.status_code, res.text)


# ----------------------------
# WEBHOOK
# ----------------------------
@app.post("/webhook")
async def webhook(req: Request):

    data = await req.json()
    print("📩 RAW JSON:", data)

    try:
        msg_obj = data.get("data", {}).get("messages", {})

        message = (
            msg_obj.get("message", {}).get("conversation")
            or msg_obj.get("messageBody")
        )

        user_id = msg_obj.get("key", {}).get("remoteJid")

    except Exception as e:
        print("❌ ERROR PARSING:", e)
        return {"status": "error"}

    if not message or not user_id:
        return {"status": "ignored"}

    print(f"📩 Poruka od {user_id}: {message}")

    answer = run_rag(message, user_id)

    send_whatsapp_message(user_id, answer)

    return {"status": "ok"}