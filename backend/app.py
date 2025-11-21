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

# ============ OPENAI KEY – FAILS FAST IF MISSING ============
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set in Railway Variables!")

client = OpenAI(api_key=openai_api_key)

# ============ APP SETUP ============
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ DATABASE & INDEX ============
DB_PATH = "data/rulebook.db"
FAISS_PATH = "data/rulebook.index"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
index = faiss.read_index(FAISS_PATH)

# ============ MODELS ============
class Source(BaseModel):
    page: int

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
        model="gpt-4-turbo",        # exact replacement for the old gpt-4.1
        temperature=0,
        messages=[
            {"role": "system", "content": "Rewrite the player's question into precise Fallout: Factions rulebook search terms. Output ONLY the search query."},
            {"role": "user", "content": user_message}
        ],
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
            chunks.append(row[1])        # full chunk – no truncation

    return chunks, pages, rewritten

# ============ CHAT ENDPOINT ============
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    chunks, pages, rewritten = retrieve_chunks(req.message)

    if not chunks:
        return ChatResponse(
            session_id=session_id,
            reply="SHORT ANSWER: No matching rule found.\n\nREASONING: I couldn't locate that in the official Fallout: Factions – Nuka-World rulebook. Try rephrasing your question!",
            sources=[]
        )

    context = "\n\n-----\n\n".join(chunks)

    system_prompt = (
        "You are the official Fallout: Factions Rules Oracle (Pip-Boy edition).\n"
        "Answer using ONLY the provided rulebook excerpts.\n"
        "Format exactly:\n"
        "SHORT ANSWER: [clear ruling in 1-2 sentences]\n\n"
        "REASONING: [natural, detailed explanation]\n"
        "Never mention page numbers – the UI shows them."
    )

    completion = client.chat.completions.create(
        model="gpt-4-turbo",          # ← this is your perfect offline model
        temperature=0.2,
        max_tokens=800,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Player question: {req.message}\nSearch terms: {rewritten}\n\nExcerpts:\n{context}"}
        ],
    )

    reply = completion.choices[0].message.content.strip()

    unique_pages = sorted(set(pages))
    sources = [Source(page=p) for p in unique_pages]

    return ChatResponse(session_id=session_id, reply=reply, sources=sources)

# ============ SERVE THE PIP-BOY FRONTEND ============
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

print("Pip-Boy Rules Oracle fully loaded – precision restored!")
