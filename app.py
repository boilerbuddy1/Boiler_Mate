import os
import re
import uuid
import pandas as pd
import streamlit as st

# --- Set OpenAI key from Streamlit Secrets ---
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

# ---------- Page setup ----------
st.set_page_config(page_title="Boiler Mate", page_icon="🔥", layout="centered")
st.title("👨‍🔧🔥 Boiler Mate")
st.caption("Diagnostics & quick lookups from your boiler manuals. (Beta)")

# ---------- Engineer Data ----------
@st.cache_data
def load_engineers():
    return pd.read_csv("data/engineers.csv")

engineers_df = load_engineers()

def normalize_postcode(pc: str) -> str:
    """Return outward code (first part before space)."""
    if not pc:
        return ""
    return pc.strip().upper().split(" ")[0]

def find_engineers_by_postcode(user_postcode, max_results=3):
    if "postcode" not in engineers_df.columns:
        return []

    oc = normalize_postcode(user_postcode)
    if not oc:
        return []

    # Exact outward code match
    matches = engineers_df[engineers_df["postcode"].str.upper().str.startswith(oc)]

    # Fallback: first 2 characters
    if matches.empty:
        matches = engineers_df[engineers_df["postcode"].str.upper().str.startswith(oc[:2])]

    return matches.head(max_results).to_dict(orient="records")

# ---------- Postcode capture ----------
with st.container():
    if "postcode" not in st.session_state:
        st.session_state.postcode = ""
    st.session_state.postcode = st.text_input(
        "Enter your postcode (for local recommendations)",
        st.session_state.postcode
    )

# ---------- Brand/Model dropdowns ----------
catalog = get_catalog()
brands = ["All"] + list(catalog.keys())

col1, col2 = st.columns(2)
with col1:
    brand_sel = st.selectbox("Brand", brands, index=0)
with col2:
    models = ["All"] + (catalog.get(brand_sel, []) if brand_sel != "All" else [])
    model_sel = st.selectbox("Model", models, index=0)

# ---------- Build engines ----------
@st.cache_resource(show_spinner=True)
def _load_with_filters(brand, model):
    idx = build_or_load_index()
    return make_engines(idx, brand=brand, model=model)

detailed_engine, concise_engine = _load_with_filters(brand_sel, model_sel)

# ---------- Answer mode ----------
mode_choice = st.radio("Answer style", ["Auto (recommended)", "Detailed", "Concise"], horizontal=True)

# ---------- Session state ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_engineers" not in st.session_state:
    st.session_state.show_engineers = False

# Show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def gc_pretty(text: str) -> str:
    m = re.search(r"\bGC\s*(?:No\.?|number|#)?[:\s]*([0-9]{6,8})\b", text, flags=re.I)
    if m:
        return f"GC number is {m.group(1)}."
    return text

# ---------- Chat input ----------
prompt = st.chat_input("Ask Boiler Mate…")
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Choose engine
    if mode_choice == "Detailed":
        engine = detailed_engine
        mode = "detailed"
    elif mode_choice == "Concise":
        engine = concise_engine
        mode = "concise"
    else:
        mode = classify_query(prompt)
        engine = concise_engine if mode == "concise" else detailed_engine

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = engine.query(prompt)
                text = (resp.response or "").strip()
                if mode == "concise":
                    text = gc_pretty(text)

                # Render answer
                st.markdown(text if text else "_No relevant content found in the selected manuals._")

                # Sources
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
                            log_referral(
                                session_id=st.session_state.session_id,
                                postcode=user_pc,
                                engineer_name=e["name"],
                                engineer_phone=e["phone"],
                                engineer_email=e["email"],
                                status="auto"
                            )

                # Log interaction
                log_interaction(
                    session_id=st.session_state.session_id,
                    postcode=st.session_state.postcode or "unknown",
                    question=prompt,
                    response=text,
                    action=mode
                )

                # Save assistant reply
                st.session_state.messages.append({"role": "assistant", "content": text or "_No content found._"})

            except Exception as e:
                st.error(f"Error answering: {e}")

# ---------- Contact engineer button (persistent) ----------
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
            st.markdown(
                f"- **{e['name']}**  \n"
                f"  📞 {e['phone']}  |  ✉️ {e['email']}  |  📍 {e['postcode']}"
            )
        for e in recs:
            log_referral(
                session_id=st.session_state.session_id,
                postcode=pc,
                engineer_name=e["name"],
                engineer_phone=e["phone"],
                engineer_email=e["email"],
                status="manual"
            )
