import json, re, random
from pathlib import Path
from datetime import datetime
from collections import Counter

RAW_FILE = Path("query_rewriting_training_data.json")
OUT_TRAIN = Path("qr_train.jsonl")
OUT_VALID = Path("qr_valid.jsonl")

MIN_POS_OVERLAP = 1          # ≥1 selected source retrieved
ENTITY_MIN_KEEP_FRAC = 0.5   # keep at least 50% of core entities
VALID_SPLIT = 0.1
SEED = 42
random.seed(SEED)

def core_entities(q):
    # crude token selection; replace w/ real NER later
    toks = re.findall(r"[A-Za-z][A-Za-z'-]+", q)
    toks = [t.lower() for t in toks if len(t) >= 4]
    return set(toks)

def entity_overlap(orig_set, rewrite):
    rw = set([w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]+", rewrite) if len(w) >= 4])
    if not orig_set: return 1.0
    return len(orig_set & rw) / len(orig_set)

def looks_bad(rw, orig_set):
    # Heuristic drift: if no overlap with core tokens
    return entity_overlap(orig_set, rw) == 0

def normalize(q):
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    return q[:120]

records = json.loads(RAW_FILE.read_text()).get("query_rewriting_data", [])
rows = []

for rec in records:
    original = rec["original_query"]
    selected = set(rec.get("selected_sources", []))
    mapping = rec.get("query_source_mapping", {})
    orig_entities = core_entities(original)
    
    # Always consider original query as baseline positive
    rows.append({
        "input": f"Rewrite the user query into a concise standalone search query.\nQuery: {original}",
        "target": normalize(original),
        "meta": {"label": "original", "overlap_score": None}
    })
    
    for rewrite_text, info in mapping.items():
        if rewrite_text == original:
            continue
        sources = info.get("sources", [])
        urls = {s["url"] for s in sources}
        overlap = len(urls & selected)
        ov_score = overlap  # integer (count of selected)
        ent_ov = entity_overlap(orig_entities, rewrite_text)
        
        if overlap >= MIN_POS_OVERLAP and ent_ov >= ENTITY_MIN_KEEP_FRAC and not looks_bad(rewrite_text, orig_entities):
            rows.append({
                "input": f"Rewrite the user query into a concise standalone search query.\nQuery: {original}",
                "target": normalize(rewrite_text),
                "meta": {
                    "label": "positive",
                    "overlap_selected": overlap,
                    "entity_overlap": round(ent_ov, 3)
                }
            })
        else:
            # optionally store negatives for later contrastive / RL
            pass

print(f"Total raw rows (including original duplicates): {len(rows)}")

# Deduplicate (input,target) pairs
seen = set()
deduped = []
for r in rows:
    key = (r["input"], r["target"].lower())
    if key not in seen:
        seen.add(key)
        deduped.append(r)

print(f"After dedup: {len(deduped)}")

# Split
random.shuffle(deduped)
n_valid = int(len(deduped) * VALID_SPLIT)
valid = deduped[:n_valid]
train = deduped[n_valid:]

def write_jsonl(path, data):
    with path.open("w") as f:
        for d in data:
            f.write(json.dumps({"input": d["input"], "target": d["target"]}, ensure_ascii=False) + "\n")

write_jsonl(OUT_TRAIN, train)
write_jsonl(OUT_VALID, valid)

print(f"Train: {len(train)}  Valid: {len(valid)}")
