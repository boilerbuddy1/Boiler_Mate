import os
import re
from collections import defaultdict
import streamlit as st

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

# 🔑 API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ---------- Models ----------
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # local, free
Settings.llm = OpenAI(model="gpt-4o-mini")  # cheap + fast

# ---------- Paths ----------
PERSIST_DIR = "storage_bge_small"
DOCS_DIR = "manuals"

# ---------- Templates ----------
QA_TEMPLATE = PromptTemplate(
    "You are Boiler Mate, assisting qualified Gas Safe engineers.\n"
    "Always explain step-by-step with practical, on-site diagnostics:\n"
    "- What to test\n- How to test (e.g. multimeter)\n- Expected values\n"
    "Finish with final recommended action + safety reminder.\n\n"
    "Question: {query_str}\nContext:\n{context_str}\nAnswer:"
)

DIRECT_TEMPLATE = PromptTemplate(
    "You are Boiler Mate. Answer concisely for factual lookups.\n"
    "Return only the key fact(s). If a single value applies, give just that value.\n\n"
    "Question: {query_str}\nContext:\n{context_str}\nAnswer:"
)

FACTUAL_TERMS = ["gc number", "gc no", "part number", "dimensions", "kw", "output", "year"]
DIAG_TERMS = ["fault", "no ignition", "lockout", "reset", "leak", "error", "no hot water"]

def classify_query(q: str) -> str:
    ql = q.lower().strip()
    if any(t in ql for t in FACTUAL_TERMS): return "concise"
    if any(t in ql for t in DIAG_TERMS): return "detailed"
    return "detailed"

def parse_brand_model(file_name: str):
    base = os.path.splitext(os.path.basename(file_name))[0]
    parts = re.split(r"[ _\-]+", base)
    brand = parts[0].title() if parts else "Unknown"
    stop_words = {"manual", "guide", "install", "user"}
    model_tokens = [p for p in parts[1:] if p.lower() not in stop_words]
    model = " ".join(model_tokens).strip()
    return brand or "Unknown", model or "Unknown"

def _load_docs_with_meta():
    docs = SimpleDirectoryReader(DOCS_DIR, filename_as_id=True).load_data()
    for d in docs:
        fn = d.metadata.get("file_name") or "unknown.pdf"
        brand, model = parse_brand_model(fn)
        d.metadata.update({"file_name": fn, "brand": brand, "model": model})
    return docs

def build_or_load_index():
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)
    docs = _load_docs_with_meta()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

def get_catalog():
    catalog = defaultdict(set)
    for fn in os.listdir(DOCS_DIR):
        if fn.lower().endswith(".pdf"):
            brand, model = parse_brand_model(fn)
            catalog[brand].add(model)
    return {b: sorted(list(m)) for b, m in sorted(catalog.items())}

def make_engines(index, brand=None, model=None):
    filters = None
    if brand and brand != "All":
        fs = [ExactMatchFilter(key="brand", value=brand)]
        if model and model != "All":
            fs.append(ExactMatchFilter(key="model", value=model))
        filters = MetadataFilters(filters=fs)
    detailed = index.as_query_engine(text_qa_template=QA_TEMPLATE, response_mode="compact", similarity_top_k=6, filters=filters)
    concise = index.as_query_engine(text_qa_template=DIRECT_TEMPLATE, response_mode="compact", similarity_top_k=2, filters=filters)
    return detailed, concise
