import os
import re
import csv
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

# ---------- Env ----------
load_dotenv()  # OpenAI key is injected via st.secrets in app.py

# ---------- Paths ----------
if os.path.exists("manuals"):
    DOCS_DIR = "manuals"
elif os.path.exists("Manuals"):
    DOCS_DIR = "Manuals"
else:
    DOCS_DIR = "manuals"  # default

PERSIST_DIR = "storage_bge_small"
ENGINEERS_FILE = "data/engineers.csv"

# ---------- Models ----------
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = OpenAI(model="gpt-4o-mini")

# ---------- Templates ----------
QA_TEMPLATE = PromptTemplate(
    "You are Boiler Mate, assisting qualified Gas Safe engineers.\n"
    "Always explain step-by-step, with practical, on-site diagnostics:\n"
    "1) Understand the question.\n"
    "2) Break into diagnostic steps (most likely first).\n"
    "3) For each step: what to test, how to test (e.g., multimeter), expected values.\n"
    "4) If pass/fail, what to do next.\n"
    "5) Final recommended action and brief safety reminder.\n"
    "Keep steps short. Use bullets. Assume the engineer has tools and training.\n\n"
    "Question: {query_str}\n"
    "Context:\n{context_str}\n"
    "Answer:"
)

DIRECT_TEMPLATE = PromptTemplate(
    "You are Boiler Mate. Answer concisely for factual lookups.\n"
    "Return only the key fact(s) with minimal words. If a single value applies, give just that value.\n"
    "If uncertain, ask for the missing detail (e.g., brand+model).\n\n"
    "Question: {query_str}\n"
    "Context:\n{context_str}\n"
    "Answer:"
)

# ---------- Query Classification ----------
FACTUAL_TERMS = [
    "gc number", "gc no", "gc#", "g.c.", "part number", "code meaning",
    "dimensions", "height", "width", "depth", "weight",
    "kw", "output", "rating", "inlet pressure", "max pressure", "min pressure",
    "year", "model number", "serial", "clearance", "flue length"
]
DIAG_TERMS = [
    "fault", "no ignition", "won't ignite", "lockout", "reset", "leak",
    "losing pressure", "noise", "diagnose", "steps", "how to", "doesn't heat",
    "not heating", "no hot water", "error", "f", "e", "l", "code"
]

def classify_query(q: str) -> str:
    ql = q.lower().strip()
    if any(term in ql for term in FACTUAL_TERMS):
        return "concise"
    if any(term in ql for term in DIAG_TERMS):
        return "detailed"
    if (len(re.findall(r"\w+", ql)) <= 6) and ("?" in ql or "gc" in ql):
        return "concise"
    return "detailed"

# ---------- Brand/Model Parsing ----------
def parse_brand_model(file_name: str):
    base = os.path.splitext(os.path.basename(file_name))[0]
    parts = re.split(r"[ _\-]+", base)
    brand = parts[0].title() if parts else "Unknown"
    stop_words = {"manual", "guide", "install", "installation", "user", "servicing", "service"}
    model_tokens = []
    for p in parts[1:]:
        low = p.lower()
        if low in stop_words:
            break
        model_tokens.append(p)
    model = " ".join(model_tokens).strip()
    model = re.sub(r"\s+", " ", model)
    return brand or "Unknown", model or "Unknown"

# ---------- Load Manuals ----------
def _load_docs_with_meta():
    if not os.path.exists(DOCS_DIR):
        return []
    docs = SimpleDirectoryReader(DOCS_DIR, filename_as_id=True).load_data()
    for d in docs:
        fn = d.metadata.get("file_name") or d.metadata.get("filename") or "unknown.pdf"
        brand, model = parse_brand_model(fn)
        d.metadata["file_name"] = fn
        d.metadata["brand"] = brand
        d.metadata["model"] = model
    return docs

def build_or_load_index():
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        docs = _load_docs_with_meta()
        index = VectorStoreIndex.from_documents(docs) if docs else VectorStoreIndex([])
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

# ---------- Catalog ----------
def get_catalog():
    catalog = defaultdict(set)
    if not os.path.exists(DOCS_DIR):
        return {}
    for fn in os.listdir(DOCS_DIR):
        if fn.lower().endswith(".pdf"):
            brand, model = parse_brand_model(fn)
            catalog[brand].add(model)
    return {b: sorted(list(models)) for b, models in sorted(catalog.items(), key=lambda x: x[0].lower())}

# ---------- Engines ----------
def make_engines(index, brand: str | None = None, model: str | None = None):
    filters = None
    if brand and brand != "All":
        fs = [ExactMatchFilter(key="brand", value=brand)]
        if model and model != "All":
            fs.append(ExactMatchFilter(key="model", value=model))
        filters = MetadataFilters(filters=fs)

    detailed_kwargs = dict(text_qa_template=QA_TEMPLATE, response_mode="compact", similarity_top_k=6)
    concise_kwargs  = dict(text_qa_template=DIRECT_TEMPLATE, response_mode="compact", similarity_top_k=2)

    if filters:
        detailed = index.as_query_engine(**detailed_kwargs, filters=filters)
        concise  = index.as_query_engine(**concise_kwargs,  filters=filters)
    else:
        detailed = index.as_query_engine(**detailed_kwargs)
        concise  = index.as_query_engine(**concise_kwargs)

    return detailed, concise

# ---------- Logging ----------
LOG_FILE = "data/chat_logs.csv"
REFERRAL_FILE = "data/referrals.csv"
os.makedirs("data", exist_ok=True)

def _write_csv(path, row, header):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)

def log_interaction(session_id, postcode, question, response, action="chat"):
    _write_csv(LOG_FILE, [
        datetime.now().isoformat(),
        session_id,
        postcode,
        action,
        question,
        response
    ], header=["timestamp", "session_id", "postcode", "action", "question", "response"])

def log_referral(session_id, postcode, engineer_name, engineer_phone, engineer_email, status="shown"):
    _write_csv(REFERRAL_FILE, [
        datetime.now().isoformat(),
        session_id,
        postcode,
        engineer_name,
        engineer_phone,
        engineer_email,
        status
    ], header=["timestamp", "session_id", "postcode", "engineer_name", "engineer_phone", "engineer_email", "status"])

# ---------- Engineer Finder ----------
def find_local_engineers(user_postcode: str, max_results: int = 3):
    """Match engineers by postcode prefix (first 2–3 characters)."""
    try:
        df = pd.read_csv(ENGINEERS_FILE)
        if "postcode" not in df.columns:
            return []

        area = user_postcode.strip().upper()[:3]
        matches = df[df["postcode"].str.upper().str.startswith(area)]

        if matches.empty:
            return []

        return matches.head(max_results).to_dict("records")
    except Exception:
        return []

# ---------- CLI ----------
def main():
    idx = build_or_load_index()
    detailed_engine, concise_engine = make_engines(idx)
    print("Boiler Mate CLI. Type your question (or 'exit'):")
    while True:
        q = input("\nQ: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        mode = classify_query(q)
        engine = concise_engine if mode == "concise" else detailed_engine
        resp = engine.query(q)
        print(f"\n[{mode}] {resp.response}")
# --------- CSV Logging ----------
LOG_FILE = "chat_logs.csv"
REFERRAL_FILE = "referrals.csv"

def log_interaction(session_id, postcode, question, response, action="chat"):
    """
    Save chatbot Q&A interactions into chat_logs.csv
    """
    _write_csv(LOG_FILE, [
        datetime.now().isoformat(),
        session_id,
        postcode,
        action,
        question,
        response
    ], header=["timestamp", "session_id", "postcode", "action", "question", "response"])

def log_referral(session_id, postcode, engineer_name, engineer_phone, engineer_email, status="shown"):
    """
    Save engineer referrals into referrals.csv
    """
    _write_csv(REFERRAL_FILE, [
        datetime.now().isoformat(),
        session_id,
        postcode,
        engineer_name,
        engineer_phone,
        engineer_email,
        status
    ], header=["timestamp", "session_id", "postcode", "engineer_name", "engineer_phone", "engineer_email", "status"])

def _write_csv(path, row, header):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)

if __name__ == "__main__":
    main()
