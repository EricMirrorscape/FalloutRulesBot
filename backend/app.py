import os
import sqlite3
import uuid
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

# ============ CRITICAL FIX: FORCE OPENAI KEY FROM RAILWAY ============
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set! Add it in Railway Variables.")

client = OpenAI(api_key=openai_api_key)

# ============ FASTAPI APP SETUP ============
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index and SQLite DB
DB_PATH = "data/rulebook.db"
FAISS_PATH = "data/rulebook.index"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
index = faiss.read_index(FAISS_PATH)

# ============ PYDANTIC MODELS ============
class Source(BaseModel):
    page: int
    image_url: str | None = None

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sources: List[Source]

# ============ EMBEDDING & RETRIEVAL ============
def embed(text: str) -> np.ndarray:
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(res.data[0].embedding, dtype="float32")

def rewrite_query(user_message: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Rewrite player questions about Fallout: Factions – Nuka-World into short, precise rulebook search queries using official terminology. Output ONLY the query."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()

def retrieve_chunks(user_message: str):
    rewritten = rewrite_query(user_message)
    emb1 = embed(user_message)
    emb2 = embed(rewritten)
    emb = (emb1 + emb2) / 2.0
    emb = np.expand_dims(emb, axis=0).astype("float32")

    D, I = index.search(emb, k=6)
    chunks = []
    pages = []

    for idx in I[0]:
        if idx == -1:
            continue
        cur.execute("SELECT page, content FROM chunks WHERE id = ?", (int(idx),))
        row = cur.fetchone()
        if row:
            pages.append(row[0])
            chunks.append(row[1][:3000])  # limit chunk size

    return chunks, pages, rewritten

# ============ CHAT ENDPOINT ============
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    chunks, pages, rewritten = retrieve_chunks(req.message)

    if not chunks:
        return ChatResponse(
            session_id=session_id,
            reply="SHORT ANSWER: No matching rule found.\n\nREASONING: I couldn't locate that in the Fallout: Factions – Nuka-World rulebook. Try rephrasing!",
            sources=[]
        )

    context = "\n\n-----\n\n".join(chunks)

    system_prompt = (
        "You are the official Fallout: Factions Rules Oracle (Pip-Boy edition). "
        "Answer ONLY using the provided rulebook excerpts. "
        "Format:\n"
        "SHORT ANSWER: [one clear ruling]\n\n"
        "REASONING: [natural explanation, no page numbers here]\n"
        "Be friendly and helpful like a vault dweller."
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Player question: {req.message}\nSearch terms: {rewritten}\n\nExcerpts:\n{context}"}
        ],
    )

    reply = completion.choices[0].message.content.strip()

    unique_pages = sorted(set(pages))
    sources = [Source(page=p, image_url=f"/pages/page_{p:03d}.webp") for p in unique_pages]

    return ChatResponse(session_id=session_id, reply=reply, sources=sources)

# ============ SERVE PIP-BOY FRONTEND ============
app.mount("/pages", StaticFiles(directory="data/pages"), name="pages")  # optional for page images
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

# Root route so the Railway URL opens the Pip-Boy instantly
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

print("Pip-Boy Rules Oracle backend loaded – ready for the wasteland!")
