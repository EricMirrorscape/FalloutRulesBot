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

# ============================================================
#  OPENAI KEY (Railway)
# ============================================================
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set in Railway variables.")

client = OpenAI(api_key=openai_api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  SESSION MEMORY (OPTION B FEATURE)
# ============================================================
last_questions = {}   # stores previous user question per session ID

# ============================================================
#  DATA FILES
# ============================================================
DB_PATH = "data/rulebook.db"
FAISS_PATH = "data/rulebook.index"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
index = faiss.read_index(FAISS_PATH)

# ============================================================
#  MODELS
# ============================================================
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

# ============================================================
#  EMBEDDINGS
# ============================================================
def embed(text: str) -> np.ndarray:
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(res.data[0].embedding, dtype="float32")

# ============================================================
#  FOLLOW-UP AWARE QUERY REWRITE (OPTION B)
# ============================================================
def rewrite_query(user_message: str, session_id: str | None):
    """
    Rewrite casual questions into rulebook-style queries.
    If the question is short (e.g., 'what about his hat?'), 
    prepend the previous question for context.
    """
    short = (
        len(user_message.split()) <= 6
        or user_message.lower().startswith(("what about", "what if", "and if", "and what"))
    )

    if session_id and short and session_id in last_questions:
        combined = f"{last_questions[session_id]}\nFollow-up: {user_message}"
    else:
        combined = user_message

    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite Fallout: Factions player questions into precise rulebook search terms. "
                    "Output ONLY the rewritten search query."
                )
            },
            {"role": "user", "content": combined}
        ],
    )

    return completion.choices[0].message.content.strip()

# ============================================================
#  FAISS RETRIEVAL
# ============================================================
def retrieve_chunks(user_message: str, session_id: str | None):
    rewritten = rewrite_query(user_message, session_id)

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
            chunks.append(row[1])  # full chunk, no truncation

    return chunks, pages, rewritten

# ============================================================
#  CHAT ENDPOINT
# ============================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    chunks, pages, rewritten = retrieve_chunks(req.message, session_id)

    if not chunks:
        return ChatResponse(
            session_id=session_id,
            reply=(
                "SHORT ANSWER: No rule found.\n\n"
                "REASONING: I couldn't find anything related to that in the official "
                "Fallout: Factions – Nuka-World rulebook. Try rephrasing!"
            ),
            sources=[]
        )

    context = "\n\n-----\n\n".join(chunks)

    system_prompt = (
        "You are the Pip-Boy Rules Oracle for Fallout: Factions – Nuka-World.\n"
        "You must answer ONLY using the provided rulebook excerpts.\n\n"
        "Format:\n"
        "SHORT ANSWER: [one-sentence ruling]\n\n"
        "REASONING: [rule explanation]\n\n"
        "Do NOT mention page numbers in the text (UI will show them)."
    )

    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {req.message}\nSearch: {rewritten}\n\nExcerpts:\n{context}"}
        ]
    )

    reply = completion.choices[0].message.content.strip()

    # Save the current message for follow-up context next time
    last_questions[session_id] = req.message

    unique_pages = sorted(set(pages))
    sources = [Source(page=p) for p in unique_pages]

    return ChatResponse(session_id=session_id, reply=reply, sources=sources)

# ============================================================
#  SERVE FRONTEND
# ============================================================
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

print("Pip-Boy Rules Oracle ONLINE — Follow-up Context Enabled!")
