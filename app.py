import os
import re
import uuid
import pandas as pd
import streamlit as st

# --- OpenAI Key from Secrets ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from boiler_mate import (
    build_or_load_index,
    make_engines,
    classify_query,
    get_catalog,
    log_interaction,
    log_referral,
)

# --- Page Setup ---
st.set_page_config(page_title="Boiler Mate", page_icon="🔥", layout="centered")

# --- Admin Mode Detection ---
try:
    query_params = st.query_params  # Streamlit >= 1.31
except Exception:
    query_params = st.experimental_get_query_params()  # fallback

admin_mode = False
if "admin" in query_params:
    if isinstance(query_params["admin"], list):
        admin_mode = query_params["admin"][0] == "supersecret123"
    else:
        admin_mode = query_params["admin"] == "supersecret123"

# ===================== ADMIN DASHBOARD =====================
if admin_mode:
    st.title("📊 Boiler Mate Admin Dashboard (Private)")
    st.caption("For internal analytics only – not visible to users.")

    chat_logs = pd.read_csv("data/chat_logs.csv") if os.path.exists("data/chat_logs.csv") else pd.DataFrame()
    referral_logs = pd.read_csv("data/referrals.csv") if os.path.exists("data/referrals.csv") else pd.DataFrame()

    st.subheader("💬 Chat Logs")
    if chat_logs.empty:
        st.info("No chat logs yet.")
    else:
        st.dataframe(chat_logs)

    st.subheader("🔧 Referral Logs")
    if referral_logs.empty:
        st.info("No referrals yet.")
    else:
        st.dataframe(referral_logs)

    st.stop()  # stop execution here so chatbot UI is hidden

# ===================== CHATBOT UI =====================
st.title("👨‍🔧🔥 Boiler Mate")
st.caption("Diagnostics & quick lookups from your boiler manuals. (Beta)")

# --- Engineer Data ---
@st.cache_data
def load_engineers():
    return pd.read_csv("data/engineers.csv")

engineers_df = load_engineers()

def find_engineers_by_postcode(user_postcode, max_results=3):
    if not user_postcode:
        return []

    pc = user_postcode.strip().upper()
    outward_code = pc.split(" ")[0]  # e.g. SW1A
    prefix2 = pc[:2]
    prefix3 = pc[:3]

    matches = engineers_df[
        engineers_df['postcode'].str.upper().str.startswith(outward_code)
        | engineers_df['postcode'].str.upper().str.startswith(prefix3)
        | engineers_df['postcode'].str.upper().str.startswith(prefix2)
    ]

    return matches.head(max_results).to_dict(orient="records")

# --- Postcode Capture ---
with st.container():
    if "postcode" not in st.session_state:
        st.session_state.postcode = ""
    st.session_state.postcode = st.text_input("Enter your postcode (for local recommendations)", st.session_state.postcode)

# --- Brand/Model Dropdowns ---
catalog = get_catalog()
brands = ["All"] + list(catalog.keys())

col1, col2 = st.columns(2)
with col1:
    brand_sel = st.selectbox("Brand", brands, index=0)
with col2:
    models = ["All"] + (catalog.get(brand_sel, []) if brand_sel != "All" else [])
    model_sel = st.selectbox("Model", models, index=0)

# --- Build Engines ---
@st.cache_resource(show_spinner=True)
def _load_with_filters(brand, model):
    idx = build_or_load_index()
    return make_engines(idx, brand=brand, model=model)

detailed_engine, concise_engine = _load_with_filters(brand_sel, model_sel)

# --- Answer Mode ---
mode_choice = st.radio("Answer style", ["Auto (recommended)", "Detailed", "Concise"], horizontal=True)

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_engineers" not in st.session_state:
    st.session_state.show_engineers = False  # persists engineer results

# --- Chat History ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def gc_pretty(text: str) -> str:
    m = re.search(r"\bGC\s*(?:No\.?|number|#)?[:\s]*([0-9]{6,8})\b", text, flags=re.I)
    return f"GC number is {m.group(1)}." if m else text

# --- Chat Input ---
prompt = st.chat_input("Ask Boiler Mate…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Choose Engine
    if mode_choice == "Detailed":
        engine = detailed_engine
        mode = "detailed"
    elif mode_choice == "Concise":
        engine = concise_engine
        mode = "concise"
    else:
        mode = classify_query(prompt)
        engine = concise_engine if mode == "concise" else detailed_engine

    # Get Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = engine.query(prompt)
                text = gc_pretty((resp.response or "").strip()) if mode == "concise" else (resp.response or "").strip()

                st.markdown(text if text else "_No relevant content found in the selected manuals._")

                if getattr(resp, "source_nodes", None):
                    with st.expander("📄 Sources"):
                        for node in resp.source_nodes[:3]:
                            meta = node.node.metadata
                            st.write(f"- **{meta.get('file_name','?')}**, page {meta.get('page_label','?')} • {meta.get('brand','?')} {meta.get('model','?')}")

                # --- Auto-detect postcode in query ---
                postcode_match = re.search(r"\b([A-Z]{1,2}[0-9][A-Z0-9]?(?:\s*\d[A-Z]{2})?)\b", prompt.upper())
                if postcode_match:
                    user_pc = postcode_match.group(1)
                    matches = find_engineers_by_postcode(user_pc)
                    if matches:
                        st.markdown("### 🔧 Recommended Local Engineers:")
                        for e in matches:
                            st.write(f"**{e['name']}** — 📞 {e['phone']} — 📍 {e['postcode']} — ✉️ {e['email']}")
                        for e in matches:
                            log_referral(st.session_state.session_id, user_pc, e["name"], e["phone"], e["email"], "auto")

                # --- Contact Engineer Button ---
                if st.button("📞 Contact a Local Gas Safe Engineer"):
                    st.session_state.show_engineers = True

                if st.session_state.show_engineers:
                    pc = st.session_state.postcode or ""
                    recs = find_engineers_by_postcode(pc, max_results=3)
                    if not pc:
                        st.warning("Please enter your postcode above so we can match local engineers.")
                    elif not recs:
                        st.info("No local engineers found for that area yet. We’re expanding coverage—check back soon.")
                    else:
                        st.success("Here are local Gas Safe engineers:")
                        for e in recs:
                            st.markdown(f"- **{e['name']}**  \n📞 {e['phone']}  |  ✉️ {e['email']}  |  📍 {e['postcode']}")
                        for e in recs:
                            log_referral(st.session_state.session_id, pc, e["name"], e["phone"], e["email"], "manual")

                # --- Log Interaction ---
                log_interaction(st.session_state.session_id, st.session_state.postcode or "unknown", prompt, text, mode)

                st.session_state.messages.append({"role": "assistant", "content": text or "_No content found._"})

            except Exception as e:
                st.error(f"Error answering: {e}")

