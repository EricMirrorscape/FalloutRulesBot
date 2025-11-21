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

# ============ OPENAI KEY – RAILWAY GUARANTEED ============
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set! Go to Railway → Variables and add it as a plain service variable.")

client = OpenAI(api_key=openai_api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to your data files
DB_PATH = "data/rulebook.db"
FAISS_PATH = "data/rulebook.index"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
index = faiss.read_index(FAISS_PATH)

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

def embed(text: str) -> np.ndarray:
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(res.data[0].embedding, dtype="float32")

def rewrite_query(user_message: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Rewrite Fallout: Factions player questions into precise rulebook search terms. Output ONLY the query."},
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
            chunks.append(row[1][:3000])

    return chunks, pages, rewritten

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    chunks, pages, rewritten = retrieve_chunks(req.message)

    if not chunks:
        return ChatResponse(
            session_id=session_id,
            reply="SHORT ANSWER: No rule found.\n\nREASONING: I couldn't locate that in the official Fallout: Factions – Nuka-World rulebook. Try rephrasing!",
            sources=[]
        )

    context = "\n\n-----\n\n".join(chunks)

    system_prompt = (
        "You are the official Pip-Boy Rules Oracle for Fallout: Factions – Nuka-World.\n"
        "Answer clearly using ONLY the provided excerpts.\n"
        "Format exactly:\n"
        "SHORT ANSWER: [one clear ruling]\n\n"
        "REASONING: [natural explanation]\n"
        "Never mention page numbers in text – the UI shows them."
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {req.message}\nSearch: {rewritten}\n\nExcerpts:\n{context}"}
        ],
    )

    reply = completion.choices[0].message.content.strip()

    unique_pages = sorted(set(pages))
    sources = [Source(page=p) for p in unique_pages]  # no image_url since no /pages folder

    return ChatResponse(session_id=session_id, reply=reply, sources=sources)

# ============ SERVE FRONTEND – THIS IS ALL YOU NEED ============
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

print("Pip-Boy Rules Oracle ONLINE – Wasteland ready!")
