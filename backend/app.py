import base64
import os
import time
from collections import defaultdict, deque
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from generator import INSTRUMENTS, KEYS, MOODS, generate, get_model

BASE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE), "frontend")

app = FastAPI(title="Melodia - AI Music Generator", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

get_model()   # load the AI once at startup so the first request is fast


class GenerateRequest(BaseModel):
    mood: str = "happy"
    key: str = "C"
    instrument: str = "piano"
    tempo: Optional[int] = Field(default=None, ge=50, le=200)
    bars: int = Field(default=8, ge=2, le=32)
    temperature: Optional[float] = Field(default=None, ge=0.3, le=1.5)
    seed: Optional[int] = Field(default=None, ge=0, le=999999)


# ---- Simple rate limiter: 20 requests per minute per visitor ----------
HITS = defaultdict(deque)
LIMIT, WINDOW = 20, 60


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(ip: str):
    now = time.time()
    if len(HITS) > 5000:          # keep memory small on a long-running server
        HITS.clear()
    q = HITS[ip]
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= LIMIT:
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
    q.append(now)
    
# ---- API routes --------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": get_model() is not None}


@app.get("/api/options")
def options():
    return {
        "moods": {name: {"tempo": cfg["tempo"]} for name, cfg in MOODS.items()},
        "keys": KEYS,
        "instruments": list(INSTRUMENTS.keys()),
    }


@app.post("/api/generate")
def api_generate(req: GenerateRequest, request: Request):
    check_rate_limit(client_ip(request))
    if req.mood not in MOODS:
        raise HTTPException(400, f"Unknown mood. Choose from: {', '.join(MOODS)}")
    if req.key not in KEYS:
        raise HTTPException(400, f"Unknown key. Choose from: {', '.join(KEYS)}")
    if req.instrument not in INSTRUMENTS:
        raise HTTPException(400, f"Unknown instrument. Choose from: {', '.join(INSTRUMENTS)}")

    try:
        result = generate(
            mood=req.mood, key=req.key, tempo=req.tempo, bars=req.bars,
            temperature=req.temperature, seed=req.seed, instrument=req.instrument,
        )
    except Exception as e:
        print("[api] generation error:", e)
        raise HTTPException(500, "Music generation failed. Please try again.")

    midi = result.pop("midi")
    result["midi_base64"] = base64.b64encode(midi).decode("ascii")
    return result


# ---- Website (must be mounted LAST) ------------------------------------
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"\nMelodia is starting. Open http://127.0.0.1:{port} in your browser.\n")
    uvicorn.run("app:app", host=host, port=port)