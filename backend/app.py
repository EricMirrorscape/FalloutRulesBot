import os
import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

# -------------------------------------------------
# Paths (works both locally and on Railway)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../backend
ROOT_DIR = BASE_DIR.parent                          # repo root in Docker
DATA_DIR = ROOT_DIR / "data"
FRONTEND_DIR = ROOT_DIR / "frontend"

DB_PATH = str(DATA_DIR / "rulebook.db")
FAISS_PATH = str(DATA_DIR / "rulebook.index")

# -------------------------------------------------
# OpenAI client
# -------------------------------------------------
load_dotenv()  # lets you use .env locally; Railway uses service vars

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY not set. "
        "Locally: put it in a .env file. "
        "On Railway: add it as a service variable."
    )

client = OpenAI(api_key=openai_api_key)

# -------------------------------------------------
# FastAPI app + CORS
# -------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # UI is served from same host, but this keeps life simple
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# DB + FAISS index
# -------------------------------------------------
# Single shared connection; Railway is just one process here
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

index = faiss.read_index(FAISS_PATH)

# -------------------------------------------------
# Models
# -------------------------------------------------
class Source(BaseModel):
    page: int
    image_url: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sources: List[Source]


# -------------------------------------------------
# Embeddings + retrieval
# -------------------------------------------------
def embed(text: str) -> np.ndarray:
    """Return a float32 embedding suitable for FAISS search."""
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return np.array(res.data[0].embedding, dtype="float32")


def rewrite_query(user_message: str) -> str:
    """
    Rewrite casual questions into tight Fallout: Factions search terms.
    """
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rewrite Fallout: Factions – Battle for Nuka-World player questions "
                    "into short, precise rulebook search queries.\n"
                    "Use official terms like: Visibility, Line of sight, Ranged Attack, Brawl, "
                    "Engagement, Proximity, Movement, Terrain, Cover, Damage, Weapons, etc.\n"
                    "Output ONLY the rewritten query, no explanation."
                ),
            },
            {"role": "user", "content": user_message},
        ],
    )
    return completion.choices[0].message.content.strip()


def retrieve_chunks(user_message: str):
    """
    1) Rewrite question into rulebook language
    2) Embed original + rewritten and average
    3) Search FAISS
    4) Pull matching chunks + pages from SQLite
    """
    rewritten = rewrite_query(user_message)

    emb_orig = embed(user_message)
    emb_rew = embed(rewritten)
    q_emb = (emb_orig + emb_rew) / 2.0
    q_emb = np.expand_dims(q_emb, axis=0).astype("float32")

    top_k = 8
    distances, indices = index.search(q_emb, top_k)

    chunks: List[str] = []
    pages: List[int] = []

    for idx in indices[0]:
        if idx < 0:
            continue

        cur.execute("SELECT page, content FROM chunks WHERE id = ?", (int(idx),))
        row = cur.fetchone()
        if row:
            page, content = row
            pages.append(page)
            chunks.append(content)

    return chunks, pages, rewritten


# -------------------------------------------------
# /chat API
# -------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    chunks, pages, rewritten_query = retrieve_chunks(req.message)

    if not chunks:
        reply_text = (
            "SHORT ANSWER: No specific rule found.\n\n"
            "REASONING: I couldn't find a matching rule in the indexed pages of the "
            "official Fallout: Factions – Battle for Nuka-World rulebook. "
            "You can try rephrasing the question or making a table-consensus ruling."
        )
        return ChatResponse(session_id=session_id, reply=reply_text, sources=[])

    context = "\n\n-----\n\n".join(chunks)

    system_prompt = (
        "You are the Pip-Boy Rules Oracle for Fallout: Factions – Battle for Nuka-World.\n"
        "You answer rules questions ONLY using the provided rulebook excerpts.\n\n"
        "Answer format (follow exactly):\n"
        "SHORT ANSWER: [one clear ruling in one or two sentences]\n"
        "\n"
        "REASONING: [natural explanation of how the ruling follows from the rules]\n\n"
        "Interpretation rules:\n"
        "• Treat casual language as rules language (e.g. 'punch' -> Brawl, 'gun' -> Ranged Attack).\n"
        "• For Visibility / Line of Sight: bases, weapons, equipment, and protruding clothing "
        "(such as hats, coats, or capes) are IGNORED when checking visibility. "
        "If only an ignored part is visible, the model is not Visible.\n"
        "• However, any non-ignored part of the model is enough to satisfy visibility for a ranged attack; "
        "the attacker does NOT need to see the whole model.\n"
        "• Do NOT invent new mechanics; stay inside the logic of the provided excerpts.\n"
        "• If the rule truly isn’t covered, say so clearly and suggest a fair house-rule, "
        "prefacing it with: 'This part is a suggested house-rule, not from the book.'\n"
        "• Never mention page numbers in your text; the UI will display them separately."
    )

    user_prompt = (
        f"Original player question:\n{req.message}\n\n"
        f"Rewritten rulebook-style query:\n{rewritten_query}\n\n"
        f"Relevant rulebook excerpts:\n{context}"
    )

    completion = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    reply_text = completion.choices[0].message.content.strip()

    # Unique sorted pages for UI
    unique_pages = sorted(set(pages))
    sources = [Source(page=p) for p in unique_pages]

    return ChatResponse(session_id=session_id, reply=reply_text, sources=sources)


# -------------------------------------------------
# Frontend serving (index.html, script.js, styles.css)
# -------------------------------------------------
# This makes / serve index.html and all the static assets from /frontend.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# Optional: simple health check for Railway logs
@app.get("/health")
async def health():
    return {"status": "Fallout Rules Bot API Online"}


print("Pip-Boy Rules Oracle ONLINE – Wasteland ready!")
