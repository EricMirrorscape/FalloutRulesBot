import os
import sqlite3
import uuid
from typing import List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# -------------------------------------------------
# Setup
# -------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_PATH = "../data/rulebook.db"
FAISS_PATH = "../data/rulebook.index"

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
# Embeddings + query rewrite
# -------------------------------------------------

def embed(text: str) -> np.ndarray:
  res = client.embeddings.create(
    model="text-embedding-3-small",
    input=text,
  )
  return np.array(res.data[0].embedding, dtype="float32")


def rewrite_query(user_message: str) -> str:
  """
  Rewrite casual questions into rulebook-style search queries.
  """
  completion = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
      {
        "role": "system",
        "content": (
          "You rewrite player questions about Fallout: Factions – Battle for Nuka-World "
          "into short search queries using the rulebook's terminology. "
          "Use terms like: Actions, Attack, Brawl, Ranged attack, Engagement, Proximity, "
          "Line of sight, Visibility, Cover, Movement, Terrain, Weapons, Damage, etc. "
          "Output ONLY the rewritten query, no explanation."
        ),
      },
      {"role": "user", "content": user_message},
    ],
  )
  return completion.choices[0].message.content.strip()


def retrieve_chunks(user_message: str):
  # 1) Rewrite into rulebook language
  rewritten = rewrite_query(user_message)

  # 2) Embed original + rewritten, average for robustness
  emb_orig = embed(user_message)
  emb_rew = embed(rewritten)
  q_emb = (emb_orig + emb_rew) / 2.0
  q_emb = np.expand_dims(q_emb, axis=0)  # shape (1, d)

  # 3) Search FAISS
  top_k = 8
  distances, indices = index.search(q_emb, top_k)
  ids = indices[0]

  chunks: List[str] = []
  pages: List[int] = []

  for idx in ids:
    if idx < 0:
      continue

    cur.execute("SELECT page, content FROM chunks WHERE id = ?", (int(idx),))
    row = cur.fetchone()
    if row:
      page, content = row
      chunks.append(content)
      pages.append(page)

  return chunks, pages, rewritten


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------

app = FastAPI()

# Allow local file usage & future hosted frontends
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
  session_id = req.session_id or str(uuid.uuid4())

  chunks, pages, rewritten_query = retrieve_chunks(req.message)

  if not chunks:
    reply_text = (
      "SHORT ANSWER:\n"
      "I couldn't find any relevant rules in the indexed pages for that question.\n\n"
      "REASONING:\n"
      "The local rulebook index didn't return any matching excerpts. "
      "Make sure you're using the correct Fallout: Factions rulebook version, "
      "or try rephrasing your question."
    )
    return ChatResponse(session_id=session_id, reply=reply_text, sources=[])

  context = "\n\n-----\n\n".join(chunks)

  system_prompt = (
    "You are the Fallout Factions Rules Bot.\n"
    "You answer questions ONLY using the provided excerpts from the official "
    "Fallout: Factions – Battle for Nuka-World rulebook.\n\n"
    "Answer style:\n"
    "1. Start with a line that begins with 'SHORT ANSWER:' followed by a single clear ruling "
    "in one or two sentences.\n"
    "2. Then write a blank line.\n"
    "3. Then write a line that begins with 'REASONING:' and explain why that ruling follows "
    "from the rules, referring to the rules in natural language.\n"
    "4. Be interpretive: if the player uses casual language, map it to the closest rules "
    "(e.g., 'toe' -> visibility / line of sight; 'punch' -> Brawl / melee attack).\n"
    "5. If the rule truly isn't covered, say that clearly and suggest a fair house-rule, but "
    'preface it with something like: "This part is a suggested house-rule, not from the book.".\n'
    "6. Do NOT mention page numbers in your text; the UI will show them separately.\n"
  )

  user_prompt = (
    f"Original player question:\n{req.message}\n\n"
    f"Rewritten rulebook-style query:\n{rewritten_query}\n\n"
    f"Relevant rulebook excerpts:\n{context}"
  )

  completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
    ],
  )

  reply_text = completion.choices[0].message.content.strip()

  # Build sources list with unique pages
  unique_pages = sorted(set(pages))
  sources = [
    Source(page=p, image_url=f"https://YOUR-CDN.com/pages/page_{p:03d}.webp")
    for p in unique_pages
  ]

  return ChatResponse(session_id=session_id, reply=reply_text, sources=sources)
