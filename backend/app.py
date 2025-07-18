import os, re, asyncio, json
from typing import AsyncGenerator
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, HttpUrl
from readability import Document
from sentence_transformers import CrossEncoder

# ────── Config ──────
load_dotenv()

BRAVE_BASE_URL   = "https://api.search.brave.com/res/v1"
BRAVE_WEB_SEARCH = f"{BRAVE_BASE_URL}/web/search"
BRAVE_API_KEY    = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

if not BRAVE_API_KEY:
    raise RuntimeError("Missing BRAVE_API_KEY")

HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
    "X-Subscription-Token": BRAVE_API_KEY,
}

MAX_CHARS_PER_PAGE = 1_500
REQUEST_TIMEOUT    = 8.0

# Training data storage
TRAINING_DATA_FILE = Path("training_data.json")

def load_training_data():
    """Load existing training data from file"""
    if TRAINING_DATA_FILE.exists():
        try:
            with open(TRAINING_DATA_FILE, 'r') as f:
                data = json.load(f)
                return data.get("training_data", [])
        except Exception as e:
            print(f"Error loading training data: {e}")
            return []
    return []

def save_training_data(training_data):
    """Save training data to file"""
    try:
        # Ensure directory exists
        TRAINING_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data and append new entry
        existing_data = load_training_data()
        existing_data.append(training_data)
        
        # Save back to file
        with open(TRAINING_DATA_FILE, 'w') as f:
            json.dump({"training_data": existing_data}, f, indent=2)
        
        print(f"Training data saved. Total entries: {len(existing_data)}")
        return True
    except Exception as e:
        print(f"Error saving training data: {e}")
        return False

# ────── FastAPI app ──────
app = FastAPI(title="Better‑Perplexity – Brave Search Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────── Models ──────
class BraveSearchResult(BaseModel):
    title: str
    url:   HttpUrl
    snippet: str
    source: str

class WebSearchResponse(BaseModel):
    query: str
    results: list[BraveSearchResult]

class TrainingResponse(BaseModel):
    text: str
    temperature: float
    latency: float
    tokens_per_minute: float

class SourceRanking(BaseModel):
    title: str
    url: str
    ranked: bool = False

class TrainingDataPoint(BaseModel):
    query: str
    response1: TrainingResponse
    response2: TrainingResponse
    selected_response: int
    sources: list[SourceRanking] = []
    timestamp: str

class SaveTrainingDataRequest(BaseModel):
    query: str
    response1: TrainingResponse
    response2: TrainingResponse
    selected_response: int
    sources: list[SourceRanking]

class SaveSourceRankingRequest(BaseModel):
    training_id: str
    sources: list[SourceRanking]

# ────── Reranker ──────
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v1-turbo-en")
try:
    print("⏳ Loading reranker…")
    reranker = CrossEncoder(RERANKER_MODEL_ID, trust_remote_code=True)
    print("✅ Reranker ready")
except Exception as e:
    print(f"⚠️  Reranker unavailable: {e}")
    reranker = None

# ────── OpenAI ──────
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ────── Helper funcs ──────
async def fetch_clean_text(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        html = Document(r.text).summary()
        text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
        return text[:MAX_CHARS_PER_PAGE]
    except Exception:
        return ""

def split_into_chunks(text: str, max_chars: int = 800, overlap: int = 120):
    chunks, start = [], 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ────── Routes ──────
@app.get("/search", response_model=WebSearchResponse)
async def web_search(
    q: str = Query(...), count: int = Query(10, ge=1, le=50), offset: int = 0
):
    params = {"q": q, "count": count, "offset": offset}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(BRAVE_WEB_SEARCH, params=params, headers=HEADERS)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Brave API error: {e}")

    data  = resp.json()
    hits  = []
    for item in data.get("web", {}).get("results", []):
        profile = item.get("profile") or {}
        meta    = item.get("meta_url") or {}
        hits.append(
            BraveSearchResult(
                title   = item.get("title", ""),
                url     = item.get("url", ""),
                snippet = item.get("description", ""),
                source  = profile.get("long_name") or meta.get("hostname") or "",
            )
        )
    return WebSearchResponse(query=q, results=hits)

@app.post("/save_training_data")
async def save_training_data_endpoint(request: SaveTrainingDataRequest):
    """Save training data when user selects better response"""
    try:
        training_data = {
            "id": f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(load_training_data())}",
            "query": request.query,
            "response1": request.response1.dict(),
            "response2": request.response2.dict(),
            "selected_response": request.selected_response,
            "sources": [source.dict() for source in request.sources],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        success = save_training_data(training_data)
        if success:
            return {"status": "success", "message": "Training data saved successfully", "training_id": training_data["id"]}
        else:
            raise HTTPException(500, "Failed to save training data")
    except Exception as e:
        raise HTTPException(500, f"Error saving training data: {str(e)}")

@app.post("/save_source_ranking")
async def save_source_ranking_endpoint(request: SaveSourceRankingRequest):
    """Save source rankings for reranker training"""
    try:
        # Load existing training data
        existing_data = load_training_data()
        
        # Find the training entry to update
        training_entry = None
        for entry in existing_data:
            if entry.get("id") == request.training_id:
                training_entry = entry
                break
        
        if not training_entry:
            raise HTTPException(404, "Training data not found")
        
        # Update the sources with rankings
        training_entry["sources"] = [source.dict() for source in request.sources]
        
        # Save back to file
        with open(TRAINING_DATA_FILE, 'w') as f:
            json.dump({"training_data": existing_data}, f, indent=2)
        
        print(f"Source rankings saved for training ID: {request.training_id}")
        return {"status": "success", "message": "Source rankings saved successfully"}
    except Exception as e:
        raise HTTPException(500, f"Error saving source rankings: {str(e)}")

@app.get("/training_data")
async def get_training_data():
    """Get all training data (for debugging/admin purposes)"""
    try:
        data = load_training_data()
        return {"training_data": data, "count": len(data)}
    except Exception as e:
        raise HTTPException(500, f"Error loading training data: {str(e)}")


# ─── Add near the top, after other imports ─────────────────────────────
import boto3, datetime, math

SM_REGION        = "us-east-2"
SM_ENDPOINT_NAME = "Endpoint-20250717-165210"   # Llama‑80B endpoint

sm_client = boto3.client("sagemaker-runtime", region_name=SM_REGION)
# ───────────────────────────────────────────────────────────────────────



# ────── /answer  (sync OR stream) ──────
@app.get("/answer")
async def answer_with_citations(
    q: str = Query(...),
    count: int = 5,
    mode: str = Query("vanilla", pattern="^(vanilla|rerank)$"),
    model: str = Query("gpt4o", pattern="^(gpt4o|llama)$"),
    training: bool = Query(False, description="Enable training mode for model comparison"),
    stream: bool = False,
):
    if stream:
        return StreamingResponse(
            _stream_answer(q, count, mode, model, training),
            media_type="text/plain",  # newline‑delimited JSON
        )
    else:
        full = [chunk async for chunk in _stream_answer(q, count, mode, model, training)]
        final_json = json.loads(full[-1])
        return {
            "query": q,
            "answer": final_json["answer"],
            "sources": final_json["sources"],
            "training": training,
        }

# ─── Core generator, now model‑aware and training‑aware ───────────────────────────────────
async def _stream_answer(
    q: str, count: int, mode: str, model: str, training: bool = False
) -> AsyncGenerator[str, None]:
    
    # 0) Sanity check for API keys / permissions
    if model == "gpt4o" and not openai_client:
        yield json.dumps({"type": "error", "msg": "OpenAI key missing"}) + "\n"
        return

    # 1) Search Brave
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                BRAVE_WEB_SEARCH, params={"q": q, "count": count}, headers=HEADERS
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            yield json.dumps({"type": "error", "msg": f"Brave error: {e}"}) + "\n"
            return
    raw_results = resp.json().get("web", {}).get("results", [])[:count]
    if not raw_results:
        yield json.dumps({"type": "end", "answer": "No web results.", "sources": []}) + "\n"
        return

    # 2) Grab pages
    async with httpx.AsyncClient() as client:
        bodies = await asyncio.gather(
            *[fetch_clean_text(client, item["url"]) for item in raw_results]
        )

    # 3) Build chunks
    candidates, source_map = [], []
    for i, (item, body) in enumerate(zip(raw_results, bodies), start=1):
        source_map.append({"title": item.get("title", ""), "url": item.get("url", "")})
        payload = body if body else item.get("description", "")
        for chunk in split_into_chunks(payload):
            candidates.append((chunk, i))

    if not candidates:
        yield json.dumps({"type": "end", "answer": "No content.", "sources": source_map}) + "\n"
        return

    # 4) Select chunks
    if mode == "rerank" and reranker:
        pairs   = [[q, c] for c, _ in candidates]
        scores  = reranker.predict(pairs, convert_to_numpy=True)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:8]
        selected = [candidates[i] for i in top_idx]
    else:
        seen, selected = set(), []
        for chunk, idx in candidates:
            if idx not in seen:
                selected.append((chunk, idx))
                seen.add(idx)
            if len(selected) >= 8:
                break

    context_text = "\n".join(f"[{idx}] {chunk}" for chunk, idx in selected)

    # 5) Stream LLM
    sys_prompt = (
        "You are a helpful research assistant. Use the web context below to answer "
        "the user's question. Cite sources like [1], [2], …"
    )
    user_prompt = f"Question: {q}\n\nWeb Results:\n{context_text}\n\nAnswer (with citations):\n"

    # send sources immediately so UI can reserve citation bubbles
    yield json.dumps({"type": "sources", "data": source_map}) + "\n"

    # Training mode: Generate two responses for comparison
    if training:
        yield json.dumps({"type": "training_start"}) + "\n"
        
        start_time = asyncio.get_event_loop().time()
        answer1_buf = ""
        answer2_buf = ""
        
        if model == "gpt4o":
            # Use real streaming for GPT-4o mini - simple approach
            # Create both streams simultaneously
            chat_stream1 = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user",   "content": user_prompt}],
                temperature=0.3,
                max_tokens=512,
                stream=True,
            )
            
            chat_stream2 = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user",   "content": user_prompt}],
                temperature=0.7,
                max_tokens=512,
                stream=True,
            )

            # Stream both responses directly
            async def stream_response1():
                nonlocal answer1_buf
                async for event in chat_stream1:
                    delta = event.choices[0].delta.content or ""
                    if delta:
                        answer1_buf += delta
                        yield json.dumps({"type": "token", "value": delta, "response": 1}) + "\n"

            async def stream_response2():
                nonlocal answer2_buf
                async for event in chat_stream2:
                    delta = event.choices[0].delta.content or ""
                    if delta:
                        answer2_buf += delta
                        yield json.dumps({"type": "token", "value": delta, "response": 2}) + "\n"

            # Simple approach: stream both responses one after another
            # First stream response 1 completely
            async for token in stream_response1():
                yield token
            
            # Then stream response 2 completely
            async for token in stream_response2():
                yield token
            
        else:
            # Llama model - generate both responses completely (no streaming support)
            payload1 = {
                "inputs": (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{sys_prompt}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_prompt}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                ),
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stop": "<|eot_id|>",
                },
            }
            
            payload2 = {
                "inputs": (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{sys_prompt}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_prompt}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                ),
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": "<|eot_id|>",
                },
            }
            
            # Generate both responses completely
            resp1 = sm_client.invoke_endpoint(
                EndpointName=SM_ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(payload1),
            )
            resp2 = sm_client.invoke_endpoint(
                EndpointName=SM_ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(payload2),
            )
            
            out1 = json.loads(resp1["Body"].read())
            out2 = json.loads(resp2["Body"].read())
            answer1_buf = out1.get("generated_text", "")
            answer2_buf = out2.get("generated_text", "")
            
            # For Llama, stream both responses interleaved (since no real streaming)
            # Use character-based chunking to avoid word splitting issues
            chunk_size = 50  # characters per chunk
            max_chunks = max(len(answer1_buf), len(answer2_buf))
            
            for i in range(0, max_chunks, chunk_size):
                # Stream chunk from response 1
                if i < len(answer1_buf):
                    chunk1 = answer1_buf[i : i + chunk_size]
                    yield json.dumps({"type": "token", "value": chunk1, "response": 1}) + "\n"
                
                # Stream chunk from response 2
                if i < len(answer2_buf):
                    chunk2 = answer2_buf[i : i + chunk_size]
                    yield json.dumps({"type": "token", "value": chunk2, "response": 2}) + "\n"
        
        end_time = asyncio.get_event_loop().time()
        latency = end_time - start_time
        
        # Calculate metrics
        tokens_per_minute1 = len(answer1_buf.split()) / (latency / 60) if latency > 0 else 0
        tokens_per_minute2 = len(answer2_buf.split()) / (latency / 60) if latency > 0 else 0
        
        # Send training comparison data
        yield json.dumps({
            "type": "training_comparison",
            "query": q,
            "response1": {
                "answer": answer1_buf.strip(),
                "latency": round(latency, 3),
                "tokens_per_minute": round(tokens_per_minute1, 1),
                "temperature": 0.3
            },
            "response2": {
                "answer": answer2_buf.strip(),
                "latency": round(latency, 3),
                "tokens_per_minute": round(tokens_per_minute2, 1),
                "temperature": 0.7
            },
            "sources": source_map
        }) + "\n"
        
        # End training mode
        yield json.dumps({"type": "training_end"}) + "\n"
        
    else:
        # Production mode: Single response
    if model == "gpt4o":
        chat_stream = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user",   "content": user_prompt}],
            temperature=0.3,
            max_tokens=512,
            stream=True,
        )
        answer_buf = ""
        async for event in chat_stream:
            delta = event.choices[0].delta.content or ""
            if delta:
                answer_buf += delta
                    print(f"Streaming token: '{delta}'")  # Debug logging
                yield json.dumps({"type": "token", "value": delta}) + "\n"

        # SageMaker Llama‑80B (single call, then chunk tokens locally)
    else:
        print(f"Here calling Llama model instead of gpt 4o !!!!!")
        payload = {
            "inputs": (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{sys_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.9,
                "stop": "<|eot_id|>",
            },
        }
        resp = sm_client.invoke_endpoint(
            EndpointName=SM_ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        out = json.loads(resp["Body"].read())
        text = out.get("generated_text", "")

        # stream it in ~30‑token chunks so the front‑end behaves the same
        TOKENS_PER_CHUNK = 30
        words = text.split()
        answer_buf = ""
        for i in range(0, len(words), TOKENS_PER_CHUNK):
            chunk = " ".join(words[i : i + TOKENS_PER_CHUNK])
            answer_buf += chunk + " "
            yield json.dumps({"type": "token", "value": chunk + " "}) + "\n"

    # 5) Done
    yield json.dumps(
        {"type": "end", "answer": answer_buf.strip(), "sources": source_map}
    ) + "\n"


