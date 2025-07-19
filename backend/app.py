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
TRAINING_DATA_FILE = Path("training_data/training_data.json")

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


class QueryRewritingTrainingData(BaseModel):
    original_query: str
    selected_sources: list[str]  # URLs of top 3 selected sources
    query_source_mapping: dict
    query_generator: str  # "gpt4o" or "t5"
    timestamp: str



def save_query_rewriting_training_data(training_data: QueryRewritingTrainingData):
    """Save query rewriting training data to file"""
    try:
        # Create query rewriting training data file
        query_training_file = Path("training_data/query_rewriting_training_data.json")
        
        # Load existing data
        existing_data = []
        if query_training_file.exists():
            try:
                with open(query_training_file, 'r') as f:
                    data = json.load(f)
                    existing_data = data.get("query_rewriting_data", [])
            except Exception as e:
                print(f"Error loading query rewriting training data: {e}")
        
        # Add new entry
        existing_data.append(training_data.dict())
        
        # Save back to file
        with open(query_training_file, 'w') as f:
            json.dump({"query_rewriting_data": existing_data}, f, indent=2)
        
        print(f"Query rewriting training data saved. Total entries: {len(existing_data)}")
        return True
    except Exception as e:
        print(f"Error saving query rewriting training data: {e}")
        return False

# ────── Reranker Training Data ──────
class RerankerTrainingData(BaseModel):
    query: str
    all_sources: list[dict]  # All sources from all queries
    positive_sources: list[str]  # URLs of user-selected top 3
    negative_sources: list[str]  # URLs of remaining sources
    timestamp: str

def save_reranker_training_data(training_data: RerankerTrainingData):
    """Save reranker training data to file"""
    try:
        # Create reranker training data file
        reranker_training_file = Path("training_data/reranker_training_data.json")
        
        # Load existing data
        existing_data = []
        if reranker_training_file.exists():
            try:
                with open(reranker_training_file, 'r') as f:
                    data = json.load(f)
                    existing_data = data.get("reranker_data", [])
            except Exception as e:
                print(f"Error loading reranker training data: {e}")
        
        # Add new entry
        existing_data.append(training_data.dict())
        
        # Save back to file
        with open(reranker_training_file, 'w') as f:
            json.dump({"reranker_data": existing_data}, f, indent=2)
        
        print(f"Reranker training data saved. Total entries: {len(existing_data)}")
        return True
    except Exception as e:
        print(f"Error saving reranker training data: {e}")
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

class QueryRewritingTrainingData(BaseModel):
    original_query: str
    selected_sources: list[str]  # URLs of top 3 selected sources
    query_source_mapping: dict
    query_generator: str  # "gpt4o" or "t5"
    timestamp: str

# ────── Query Rewriting Models ──────
class SearchQuery(BaseModel):
    query: str
    reasoning: str

class QueryRewritingResponse(BaseModel):
    original_query: str
    rewritten_queries: list[SearchQuery]

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

# ────── T5 Query Reformulation Model ──────
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    T5_MODEL_ID = "prhegde/t5-query-reformulation-RL"
    print("⏳ Loading T5 query reformulation model...")
    t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_ID)
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_ID)
    t5_model.eval()
    if torch.cuda.is_available():
        t5_model.cuda()
    print("✅ T5 query reformulation model ready")
except Exception as e:
    print(f"⚠️  T5 model unavailable: {e}")
    t5_tokenizer = None
    t5_model = None

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

# ────── Query Rewriting Functions ──────
async def generate_search_queries_t5(user_query: str, num_queries: int = 4) -> QueryRewritingResponse:
    """Generate multiple search queries using T5 query reformulation model"""
    
    if not t5_model or not t5_tokenizer:
        # Fallback to original query if T5 model is not available
        return QueryRewritingResponse(
            original_query=user_query,
            rewritten_queries=[SearchQuery(query=user_query, reasoning="Original query (T5 model not available)")]
        )
    
    try:
        print(f"Generating {num_queries} T5 queries for: {user_query}")
        
        input_ids = t5_tokenizer(user_query, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        rewritten_queries = []
        
        with torch.no_grad():
            for i in range(num_queries):
                output = t5_model.generate(
                    input_ids, 
                    max_length=35, 
                    num_beams=1, 
                    do_sample=True, 
                    repetition_penalty=1.8
                )
                target_sequence = t5_tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Clean up the generated query
                cleaned_query = target_sequence.strip()
                if cleaned_query and cleaned_query != user_query:
                    rewritten_queries.append(SearchQuery(
                        query=cleaned_query,
                        reasoning=f"T5 reformulation {i+1}"
                    ))
        
        # If we didn't get enough unique queries, add the original
        if len(rewritten_queries) < num_queries:
            rewritten_queries.append(SearchQuery(
                query=user_query,
                reasoning="Original query (fallback)"
            ))
        
        result = QueryRewritingResponse(
            original_query=user_query,
            rewritten_queries=rewritten_queries
        )
        return result
        
    except Exception as e:
        print(f"Error generating T5 search queries: {e}")
        # Fallback to original query
        return QueryRewritingResponse(
            original_query=user_query,
            rewritten_queries=[SearchQuery(query=user_query, reasoning="Original query (T5 fallback)")]
        )

async def generate_search_queries(user_query: str, query_generator: str = "gpt4o") -> QueryRewritingResponse:
    """Generate multiple search queries using specified model (GPT-4o-mini or T5)"""
    
    if query_generator == "t5":
        return await generate_search_queries_t5(user_query)
    
    # Default to GPT-4o-mini
    if not openai_client:
        # Fallback to original query if OpenAI is not available
        return QueryRewritingResponse(
            original_query=user_query,
            rewritten_queries=[SearchQuery(query=user_query, reasoning="Original query (OpenAI not available)")]
        )
    
    try:
        print(f"Generating queries for: {user_query}")
        print(f"OpenAI client available: {openai_client is not None}")
        
        # First, let's test with a simple response to make sure OpenAI is working
        test_response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Generate 3 search queries for: {user_query}. Return as JSON array."}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        print(f"Test response: {test_response.choices[0].message.content}")
        
        # Now try with structured output
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Generate 3-5 search queries for web search. Return as JSON with 'queries' array."
                },
                {
                    "role": "user",
                    "content": f"Generate search queries for: {user_query}"
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.7
        )
        
        print(f"OpenAI response received: {response.choices[0].message.content}")
        
        # Parse the structured response
        content = response.choices[0].message.content
        parsed_data = json.loads(content)
        
        # Convert to our expected format
        queries = parsed_data.get("queries", [])
        rewritten_queries = []
        for query in queries:
            if isinstance(query, str):
                rewritten_queries.append(SearchQuery(query=query, reasoning="Generated query"))
            elif isinstance(query, dict):
                rewritten_queries.append(SearchQuery(
                    query=query.get("query", ""), 
                    reasoning=query.get("reasoning", "Generated query")
                ))
        
        result = QueryRewritingResponse(
            original_query=user_query,
            rewritten_queries=rewritten_queries
        )
        return result
        
    except Exception as e:
        print(f"Error generating search queries: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        # Fallback to original query
        return QueryRewritingResponse(
            original_query=user_query,
            rewritten_queries=[SearchQuery(query=user_query, reasoning="Original query (fallback)")]
        )

async def search_with_query(query: str, count: int = 5) -> list:
    """Search Brave API with a specific query"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                BRAVE_WEB_SEARCH, 
                params={"q": query, "count": count}, 
                headers=HEADERS
            )
            resp.raise_for_status()
            raw_results = resp.json().get("web", {}).get("results", [])[:count]
            return raw_results
        except Exception as e:
            print(f"Error searching for query '{query}': {e}")
            return []

async def aggregate_search_results(all_queries: list[SearchQuery], count_per_query: int = 3) -> tuple[list, list, dict]:
    """Aggregate search results from multiple queries and track query-source mappings"""
    
    # Create tasks for all queries
    search_tasks = []
    for query_info in all_queries:
        task = search_with_query(query_info.query, count_per_query)
        search_tasks.append((query_info, task))
    
    # Execute all searches concurrently
    all_results = []
    source_map = []
    query_source_mapping = {}  # Track which query led to which sources
    
    for query_info, task in search_tasks:
        try:
            results = await task
            all_results.extend(results)
            
            # Track sources for this query
            query_sources = []
            
            # Add source mapping with query info
            for result in results:
                source_info = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "query_used": query_info.query,
                    "reasoning": query_info.reasoning
                }
                source_map.append(source_info)
                query_sources.append(source_info)
            
            # Store mapping for this query
            query_source_mapping[query_info.query] = {
                "query": query_info.query,
                "reasoning": query_info.reasoning,
                "sources": query_sources
            }
                
        except Exception as e:
            print(f"Error in search task for query '{query_info.query}': {e}")
    
    return all_results, source_map, query_source_mapping

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

@app.post("/save_query_rewriting_training")
async def save_query_rewriting_training_endpoint(request: QueryRewritingTrainingData):
    """Save query rewriting training data"""
    try:
        success = save_query_rewriting_training_data(request)
        if success:
            return {"status": "success", "message": "Query rewriting training data saved successfully"}
        else:
            raise HTTPException(500, "Failed to save query rewriting training data")
    except Exception as e:
        raise HTTPException(500, f"Error saving query rewriting training data: {str(e)}")

@app.post("/save_reranker_training")
async def save_reranker_training_endpoint(request: RerankerTrainingData):
    """Save reranker training data"""
    try:
        success = save_reranker_training_data(request)
        if success:
            return {"status": "success", "message": "Reranker training data saved successfully"}
        else:
            raise HTTPException(500, "Failed to save reranker training data")
    except Exception as e:
        raise HTTPException(500, f"Error saving reranker training data: {str(e)}")

@app.get("/training_data")
async def get_training_data():
    """Get all training data (for debugging/admin purposes)"""
    try:
        data = load_training_data()
        return {"training_data": data, "count": len(data)}
    except Exception as e:
        raise HTTPException(500, f"Error loading training data: {str(e)}")

@app.get("/query_rewriting_training_data")
async def get_query_rewriting_training_data():
    """Get all query rewriting training data (for debugging/admin purposes)"""
    try:
        query_training_file = Path("training_data/query_rewriting_training_data.json")
        if query_training_file.exists():
            with open(query_training_file, 'r') as f:
                data = json.load(f)
                return {"query_rewriting_data": data.get("query_rewriting_data", []), "count": len(data.get("query_rewriting_data", []))}
        else:
            return {"query_rewriting_data": [], "count": 0}
    except Exception as e:
        raise HTTPException(500, f"Error loading query rewriting training data: {str(e)}")

@app.get("/generate_queries")
async def generate_queries_endpoint(
    q: str = Query(...),
    query_generator: str = Query("gpt4o", pattern="^(gpt4o|t5)$", description="Query generation model to use")
):
    """Test endpoint for query generation"""
    try:
        query_response = await generate_search_queries(q, query_generator)
        return {
            "original_query": q,
            "generated_queries": [
                {"query": sq.query, "reasoning": sq.reasoning} 
                for sq in query_response.rewritten_queries
            ],
            "query_generator": query_generator
        }
    except Exception as e:
        raise HTTPException(500, f"Error generating queries: {str(e)}")


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
    query_generator: str = Query("gpt4o", pattern="^(gpt4o|t5)$", description="Query generation model to use"),
):
    if stream:
        return StreamingResponse(
            _stream_answer(q, count, mode, model, training, query_generator),
            media_type="text/plain",  # newline‑delimited JSON
        )
    else:
        full = [chunk async for chunk in _stream_answer(q, count, mode, model, training, query_generator)]
        final_json = json.loads(full[-1])
        return {
            "query": q,
            "answer": final_json["answer"],
            "sources": final_json["sources"],
            "training": training,
        }

# ─── Core generator, now model‑aware and training‑aware ───────────────────────────────────
async def _stream_answer(
    q: str, count: int, mode: str, model: str, training: bool = False, query_generator: str = "gpt4o"
) -> AsyncGenerator[str, None]:
    
    # 0) Sanity check for API keys / permissions
    if model == "gpt4o" and not openai_client:
        yield json.dumps({"type": "error", "msg": "OpenAI key missing"}) + "\n"
        return
    
    # 1) Generate multiple search queries using specified model
    query_response = await generate_search_queries(q, query_generator)
    
    # Include original query in the list
    all_queries = [SearchQuery(query=q, reasoning="Original user query")]
    all_queries.extend(query_response.rewritten_queries)
    
    print(f"Generated {len(all_queries)} search queries:")
    for i, query_info in enumerate(all_queries):
        print(f"  {i+1}. {query_info.query} (Reasoning: {query_info.reasoning})")
    
    # 2) Search with all queries concurrently
    raw_results, source_map, query_source_mapping = await aggregate_search_results(all_queries, count_per_query=3)
    
    if not raw_results:
        yield json.dumps({"type": "end", "answer": "No web results found with any search query.", "sources": []}) + "\n"
        return

    # 3) Grab pages
    async with httpx.AsyncClient() as client:
        bodies = await asyncio.gather(
            *[fetch_clean_text(client, item["url"]) for item in raw_results]
        )

    # 4) Build chunks
    candidates = []
    for i, (item, body) in enumerate(zip(raw_results, bodies), start=1):
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

    # Send query generation info and mapping for training
    yield json.dumps({
        "type": "query_generation", 
        "original_query": q,
        "generated_queries": [{"query": sq.query, "reasoning": sq.reasoning} for sq in all_queries],
        "query_source_mapping": query_source_mapping
    }) + "\n"

    # Training mode: Generate two responses for comparison
    if training:
        yield json.dumps({"type": "training_start"}) + "\n"
        
        start_time = asyncio.get_event_loop().time()
        answer1_buf = ""
        answer2_buf = ""
        first_token_time1 = None
        first_token_time2 = None
        token_count1 = 0
        token_count2 = 0
        
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
                nonlocal answer1_buf, first_token_time1, token_count1
                async for event in chat_stream1:
                    delta = event.choices[0].delta.content or ""
                    if delta:
                        if first_token_time1 is None:
                            first_token_time1 = asyncio.get_event_loop().time()
                        answer1_buf += delta
                        token_count1 += 1
                        yield json.dumps({"type": "token", "value": delta, "response": 1}) + "\n"

            async def stream_response2():
                nonlocal answer2_buf, first_token_time2, token_count2
                async for event in chat_stream2:
                    delta = event.choices[0].delta.content or ""
                    if delta:
                        if first_token_time2 is None:
                            first_token_time2 = asyncio.get_event_loop().time()
                        answer2_buf += delta
                        token_count2 += 1
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
            
            # For Llama, we can't measure real first token time since it's not streaming
            # But we can simulate it by timing the first chunk
            first_token_time1 = asyncio.get_event_loop().time()
            first_token_time2 = asyncio.get_event_loop().time()
            token_count1 = len(answer1_buf.split())
            token_count2 = len(answer2_buf.split())
            
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
        
        # Calculate comprehensive metrics
        ttft1 = first_token_time1 - start_time if first_token_time1 else 0
        ttft2 = first_token_time2 - start_time if first_token_time2 else 0
        tokens_per_minute1 = token_count1 / (latency / 60) if latency > 0 else 0
        tokens_per_minute2 = token_count2 / (latency / 60) if latency > 0 else 0
        
        # Send enhanced training comparison data
        yield json.dumps({
            "type": "training_comparison",
            "query": q,
            "response1": {
                "answer": answer1_buf.strip(),
                "latency": round(latency, 3),
                "ttft": round(ttft1, 3),
                "tokens_per_minute": round(tokens_per_minute1, 1),
                "total_tokens": token_count1,
                "temperature": 0.3
            },
            "response2": {
                "answer": answer2_buf.strip(),
                "latency": round(latency, 3),
                "ttft": round(ttft2, 3),
                "tokens_per_minute": round(tokens_per_minute2, 1),
                "total_tokens": token_count2,
                "temperature": 0.7
            },
            "sources": source_map
        }) + "\n"
        
        # End training mode
        yield json.dumps({"type": "training_end"}) + "\n"
        
    else:
        # Production mode: Single response
        start_time = asyncio.get_event_loop().time()
        first_token_time = None
        token_count = 0
        
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
                    if first_token_time is None:
                        first_token_time = asyncio.get_event_loop().time()
                    answer_buf += delta
                    token_count += 1
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

            # For Llama, we can't measure real first token time since it's not streaming
            first_token_time = asyncio.get_event_loop().time()
            token_count = len(text.split())

            # stream it in ~30‑token chunks so the front‑end behaves the same
            TOKENS_PER_CHUNK = 30
            words = text.split()
            answer_buf = ""
            for i in range(0, len(words), TOKENS_PER_CHUNK):
                chunk = " ".join(words[i : i + TOKENS_PER_CHUNK])
                answer_buf += chunk + " "
                yield json.dumps({"type": "token", "value": chunk + " "}) + "\n"

        # 5) Done
        end_time = asyncio.get_event_loop().time()
        latency = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else 0
        tokens_per_minute = token_count / (latency / 60) if latency > 0 else 0
        
        yield json.dumps({
            "type": "end", 
            "answer": answer_buf.strip(), 
            "sources": source_map,
            "metrics": {
                "latency": round(latency, 3),
                "ttft": round(ttft, 3),
                "tokens_per_minute": round(tokens_per_minute, 1),
                "total_tokens": token_count
            }
        }) + "\n"


