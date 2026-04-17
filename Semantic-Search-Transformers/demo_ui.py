import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

DATA_FILE = Path("./arxivData.json")
EMBEDDINGS_FILE = Path("./embeddings.pkl")
MODEL_NAME = "distilbert-base-nli-stsb-mean-tokens"


@st.cache_data(show_spinner=False)
def load_dataframe() -> pd.DataFrame:
    data = pd.read_json(DATA_FILE)
    return data.drop(columns=["author", "link", "tag"], errors="ignore")


@st.cache_resource(show_spinner=True)
def load_embeddings() -> np.ndarray:
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
    embeddings = np.asarray(embeddings, dtype="float32")
    return embeddings


@st.cache_resource(show_spinner=True)
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner=True)
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def validate_files() -> list[str]:
    missing = []
    for path in [DATA_FILE, EMBEDDINGS_FILE]:
        if not path.exists():
            missing.append(str(path))
    return missing


def main() -> None:
    st.set_page_config(page_title="Semantic Search Demo", layout="wide")
    st.title("Semantic Search Demo: arXiv Papers")
    st.caption("Transformer embeddings + FAISS nearest-neighbor search")

    missing = validate_files()
    if missing:
        st.error("Required files are missing:")
        for m in missing:
            st.write(f"- {m}")
        st.info(
            "Run your notebook cells that generate embeddings first, then refresh this app."
        )
        st.stop()

    with st.spinner("Loading data, embeddings, model, and index..."):
        df = load_dataframe()
        embeddings = load_embeddings()
        model = load_model()
        index = build_index(embeddings)

    st.sidebar.header("Search Settings")
    top_k = st.sidebar.slider("Top K results", min_value=3, max_value=20, value=10)

    query = st.text_area(
        "Enter your query",
        value="Transformer architecture for machine translation with attention",
        height=140,
    )

    run_search = st.button("Search", type="primary", use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Papers", f"{len(df):,}")
    c2.metric("Embedding size", f"{embeddings.shape[1]}")
    c3.metric("Index vectors", f"{index.ntotal:,}")

    if run_search:
        cleaned_query = query.strip()
        if not cleaned_query:
            st.warning("Please enter a query.")
            st.stop()

        with st.spinner("Encoding query and searching..."):
            query_vec = model.encode([cleaned_query], convert_to_numpy=True)
            query_vec = np.asarray(query_vec, dtype="float32")
            distances, indices = index.search(query_vec, top_k)

        st.subheader("Results")

        result_rows = []
        for rank, (idx, dist) in enumerate(
            zip(indices[0].tolist(), distances[0].tolist()), start=1
        ):
            if idx < 0 or idx >= len(df):
                continue
            row = df.iloc[idx]
            result_rows.append(
                {
                    "Rank": rank,
                    "Distance (L2)": float(dist),
                    "Paper ID": row.get("id", "N/A"),
                    "Title": row.get("title", "N/A"),
                    "Summary": row.get("summary", ""),
                }
            )

        if not result_rows:
            st.warning("No results found.")
            st.stop()

        st.dataframe(
            pd.DataFrame(result_rows),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Show result cards"):
            for item in result_rows:
                st.markdown(
                    f"### {item['Rank']}. {item['Title']}\n"
                    f"**Paper ID:** {item['Paper ID']}  \n"
                    f"**L2 Distance:** {item['Distance (L2)']:.4f}"
                )
                st.write(item["Summary"])
                st.divider()


if __name__ == "__main__":
    main()
