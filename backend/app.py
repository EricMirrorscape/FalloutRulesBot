import os
import sqlite3
import uuid
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        messages=[
            {"role": "system", "content": "Rewrite casual Fallout: Factions questions into rulebook search terms. Output ONLY the query."},
            {"role": "user", "content": user_message},
        ],
    )
    return completion.choices[0].message.content.strip()

def retrieve_chunks(user_message: str):
    rewritten = rewrite_query(user_message)
    emb_orig = embed(user_message)
    emb_rewritten = embed(rewritten)
    emb = (emb_orig + emb_rewritten) / 2
    emb = np.expand_dims(emb, axis=0)

    D, I = index.search(emb, k=6)
    results = []
    pages = []
    for idx in I[0]:
        if idx == -1:
            continue
        cur.execute("SELECT page, content FROM chunks WHERE id = ?", (int(idx),))
        row = cur.fetchone()
        if row:
            pages.append(row[0])
            results.append(row[1][:2000])
    return results, pages, rewritten

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    chunks, pages, rewritten_query = retrieve_chunks(req.message)

    if not chunks:
        return ChatResponse(
            session_id=session_id,
            reply="SHORT ANSWER: I couldn't find that rule.\n\nREASONING: No matching sections in the rulebook – try rephrasing?",
            sources=[]
        )

    context = "\n\n-----\n\n".join(chunks)

    system_prompt = "You are the official Fallout: Factions Rules Oracle. Answer clearly using only the provided excerpts. Start with SHORT ANSWER: then blank line then REASONING:. Never mention page numbers in text."
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {req.message}\nRewritten: {rewritten_query}\n\nExcerpts:\n{context}"}
        ],
    )

    reply = completion.choices[0].message.content.strip()

    unique_pages = sorted(set(pages))
    sources = [Source(page=p, image_url=f"/pages/page_{p:03d}.webp") for p in unique_pages]

    return ChatResponse(session_id=session_id, reply=reply, sources=sources)

# Serve the Pip-Boy frontend
app.mount("/pages", StaticFiles(directory="data/pages"), name="pages")  # if you have page images later
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
