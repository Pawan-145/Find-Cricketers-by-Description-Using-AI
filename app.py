import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.title("🧠 Character Semantic Search (MiniLM)")

@st.cache_resource
def load_resources():
    with open("cricketers.json", "r", encoding="utf-8") as f:
        characters = json.load(f)

    index = faiss.read_index("cricketers_index.faiss")

    
    model = SentenceTransformer("all-MiniLM-L6-v2")

    return characters, index, model


characters, index, model = load_resources()

query = st.text_input("Describe a character:")

if query:
    query_vector = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)

    k = 3
    distances, indices = index.search(query_vector, k)

    st.subheader("Top Matches")

    for i, idx in enumerate(indices[0]):
        character = characters[idx]

        st.markdown(f"## {character['name']}")

        if character["image_url"]:
            st.image(character["image_url"], width=250)

        st.write(character["description"][:500] + "...")
        st.write(f"Similarity Score: {distances[0][i]:.3f}")
        st.markdown("---")