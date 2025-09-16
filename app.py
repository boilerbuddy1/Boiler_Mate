import re
import streamlit as st
from boiler_mate import build_or_load_index, make_engines, classify_query, get_catalog

st.set_page_config(page_title="Boiler Mate", page_icon="🔥", layout="centered")
st.title("👨‍🔧🔥 Boiler Mate")
st.caption("Diagnostics & quick lookups from your boiler manuals.")

# ----- Load catalog -----
catalog = get_catalog()
brands = ["All"] + list(catalog.keys())
brand_sel = st.selectbox("Brand", brands)
models = ["All"] + (catalog.get(brand_sel, []) if brand_sel != "All" else [])
model_sel = st.selectbox("Model", models)

# ----- Load engines -----
@st.cache_resource
def _load_with_filters(brand, model):
    idx = build_or_load_index()
    return make_engines(idx, brand=brand, model=model)
detailed_engine, concise_engine = _load_with_filters(brand_sel, model_sel)

# ----- Mode -----
mode_choice = st.radio("Answer style", ["Auto", "Detailed", "Concise"], horizontal=True)

# ----- Chat -----
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask Boiler Mate…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if mode_choice == "Detailed":
        engine, mode = detailed_engine, "detailed"
    elif mode_choice == "Concise":
        engine, mode = concise_engine, "concise"
    else:
        mode = classify_query(prompt)
        engine = concise_engine if mode == "concise" else detailed_engine

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = engine.query(prompt)
            text = str(resp.response or "")

            if mode == "concise":
                m = re.search(r"\bGC\s*(?:No\.?|number|#)?[:\s]*([0-9]{6,8})\b", text, flags=re.I)
                if m: text = f"GC number is {m.group(1)}."

            st.markdown(text)
            with st.expander("📄 Sources"):
                for node in resp.source_nodes[:3]:
                    meta = node.node.metadata
                    st.write(f"- **{meta.get('file_name','?')}**, page {meta.get('page_label','?')}")

    st.session_state.messages.append({"role": "assistant", "content": text})
