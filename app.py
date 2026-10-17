"""
app.py - a small Streamlit web app around the fine-tuned model.

    streamlit run app.py

By default it loads the model from ./outputs/mt5-translator. To use another
folder (or a Hugging Face model id) set the NMT_MODEL environment variable.
"""

import os

import streamlit as st

from core import LANG_NAMES, MAX_LEN, load_model_and_tokenizer, translate

MODEL_PATH = os.environ.get("NMT_MODEL", "outputs/mt5-translator")

st.set_page_config(page_title="Neural Machine Translation", page_icon="🌐")


@st.cache_resource(show_spinner="Loading the translation model (first time only)...")
def get_model(path):
    return load_model_and_tokenizer(path)


st.title("🌐 Neural Machine Translation")
st.caption("English ⇄ Japanese ⇄ Chinese, using a fine-tuned mT5 Transformer.")

try:
    tokenizer, model = get_model(MODEL_PATH)
except Exception as exc:
    st.error(
        f"Could not load a model from '{MODEL_PATH}'.\n\n"
        "Train one with `python train.py` (see colab_train.ipynb), then put the output "
        "folder at `outputs/mt5-translator`, or set the NMT_MODEL environment variable.\n\n"
        f"Details: {exc}"
    )
    st.stop()

text = st.text_area("Text to translate", "What is your name?", height=140)
col1, col2 = st.columns([2, 1])
with col1:
    target = st.selectbox("Translate into", list(LANG_NAMES), format_func=LANG_NAMES.get, index=1)
with col2:
    beams = st.slider("Beam size", 1, 8, 4, help="Higher = tries more candidate translations (slower, often better).")

if st.button("Translate", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        if len(tokenizer(text).input_ids) > MAX_LEN:
            st.warning(f"Long input: only the first {MAX_LEN} tokens are translated. Try one or two sentences at a time.")
        with st.spinner("Translating..."):
            result = translate(text, target, tokenizer, model, num_beams=beams)
        st.subheader(f"{LANG_NAMES[target]}")
        st.success(result or "(empty output)")

with st.expander("How does it work?"):
    st.write(
        "mT5 is a multilingual encoder-decoder Transformer. We fine-tuned it on the ALT parallel corpus "
        "after adding one special token per language. The token placed in front of the input "
        "(for example `<jp>`) tells the model which language to write; the model figures out the "
        "source language by itself. Output is generated with beam search."
    )
