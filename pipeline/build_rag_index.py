"""
build_rag_index.py
------------------
Embed all 824,249 DDI interaction descriptions using PubMedBERT and build a
FAISS index for semantic retrieval in the RAG pipeline.

Format (matches paper):
    "{Drug A} interaction with {Drug B} is: {description}"

Checkpoint-based: saves embeddings every CHUNK_SIZE interactions.
If interrupted, re-run the same command -- it resumes from the last checkpoint.

Output (data/rag_index/):
    faiss.index          -- FAISS IndexFlatIP (cosine similarity via L2-norm)
    metadata.pkl         -- list of dicts: interaction_id, drugbank_id_a/b, name_a/b, text
    index_info.json      -- stats: num_vectors, embedding_dim, model

Usage:
    python pipeline/build_rag_index.py
    python pipeline/build_rag_index.py --chunk-size 5000   # smaller checkpoints
    python pipeline/build_rag_index.py --rebuild            # ignore checkpoints, start fresh
"""

import os, sys, json, pickle, argparse, time
import numpy as np
import pandas as pd

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
WORKING_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPROVED_DIR = os.path.join(WORKING_DIR, "data", "step3_approved")
MAP_PATH     = os.path.join(WORKING_DIR, "data", "step4_graph", "node_mapping.csv")
OUTPUT_DIR   = os.path.join(WORKING_DIR, "data", "rag_index")
CKPT_DIR     = os.path.join(OUTPUT_DIR, "checkpoints")

MODEL_NAME   = "pritamdeka/S-PubMedBert-MS-MARCO"
BATCH_SIZE   = 256   # larger batch for short texts
CHUNK_SIZE   = 10_000

# ---------------------------------------------------------------------------

def sep(label=""):
    w = 68
    if label:
        p = (w - len(label) - 2) // 2
        print("-" * p + " " + label + " " + "-" * (w - p - len(label) - 2))
    else:
        print("-" * w)


def load_data():
    sep("LOADING TABLES")
    interactions = pd.read_csv(os.path.join(APPROVED_DIR, "drug_interactions_dedup.csv"))
    print(f"  Interactions : {len(interactions):,}")

    drugs = pd.read_csv(os.path.join(APPROVED_DIR, "drugs.csv"),
                        usecols=["drugbank_id", "name"])
    name_map = dict(zip(drugs["drugbank_id"], drugs["name"]))
    print(f"  Drugs        : {len(drugs):,}")

    return interactions, name_map


def build_texts(interactions, name_map):
    sep("FORMATTING INTERACTION TEXTS")
    records = []
    for _, row in interactions.iterrows():
        id_a  = row["drugbank_id_a"]
        id_b  = row["drugbank_id_b"]
        name_a = name_map.get(id_a, id_a)
        name_b = name_map.get(id_b, id_b)
        desc   = str(row["description"]) if pd.notna(row["description"]) else ""
        text   = f"{name_a} interaction with {name_b} is: {desc}"
        records.append({
            "interaction_id": int(row["interaction_id"]),
            "drugbank_id_a":  id_a,
            "drugbank_id_b":  id_b,
            "name_a":         name_a,
            "name_b":         name_b,
            "text":           text,
        })
    print(f"  Total records : {len(records):,}")
    print(f"  Sample        : {records[0]['text'][:120]}...")
    return records


def get_completed_chunks():
    if not os.path.isdir(CKPT_DIR):
        return set()
    done = set()
    for f in os.listdir(CKPT_DIR):
        if f.startswith("chunk_") and f.endswith(".npy"):
            try:
                done.add(int(f.split("_")[1].split(".")[0]))
            except ValueError:
                pass
    return done


def embed_chunks(records, model, chunk_size, rebuild):
    os.makedirs(CKPT_DIR, exist_ok=True)
    completed = set() if rebuild else get_completed_chunks()

    total   = len(records)
    n_chunks = (total + chunk_size - 1) // chunk_size

    sep(f"EMBEDDING {total:,} INTERACTIONS ({n_chunks} chunks)")
    if completed:
        print(f"  Resuming: {len(completed)}/{n_chunks} chunks already done")

    t0 = time.time()
    for chunk_idx in range(n_chunks):
        if chunk_idx in completed:
            print(f"  [skip] chunk {chunk_idx+1:3d}/{n_chunks} (cached)")
            continue

        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, total)
        batch_records = records[start:end]
        texts = [r["text"] for r in batch_records]

        t_chunk = time.time()
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-norm for cosine via inner product
            convert_to_numpy=True,
        )

        # save embeddings + metadata for this chunk
        np.save(os.path.join(CKPT_DIR, f"chunk_{chunk_idx:04d}.npy"), embeddings)
        with open(os.path.join(CKPT_DIR, f"chunk_{chunk_idx:04d}_meta.pkl"), "wb") as f:
            pickle.dump(batch_records, f)

        elapsed = time.time() - t_chunk
        eta_s   = elapsed * (n_chunks - chunk_idx - 1)
        eta_m   = eta_s / 60
        print(f"  [done] chunk {chunk_idx+1:3d}/{n_chunks}  "
              f"({len(texts):,} texts, {elapsed:.0f}s)  "
              f"ETA ~{eta_m:.0f} min")

    print(f"  All chunks done in {(time.time()-t0)/60:.1f} min")


def build_faiss_index(n_chunks):
    import faiss

    sep("BUILDING FAISS INDEX")
    all_embeddings = []
    all_metadata   = []

    for i in range(n_chunks):
        emb  = np.load(os.path.join(CKPT_DIR, f"chunk_{i:04d}.npy"))
        with open(os.path.join(CKPT_DIR, f"chunk_{i:04d}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        all_embeddings.append(emb)
        all_metadata.extend(meta)

    matrix = np.vstack(all_embeddings).astype("float32")
    dim    = matrix.shape[1]
    print(f"  Matrix shape : {matrix.shape}")

    # IndexFlatIP = exact inner product (= cosine similarity after L2-norm)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    print(f"  FAISS vectors: {index.ntotal:,}")

    # save
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss.index"))
    with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(all_metadata, f)

    info = {
        "num_vectors":    int(index.ntotal),
        "embedding_dim":  dim,
        "model":          MODEL_NAME,
        "metric":         "cosine (IndexFlatIP + L2-normalized vectors)",
    }
    with open(os.path.join(OUTPUT_DIR, "index_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Saved: faiss.index, metadata.pkl, index_info.json")
    return index, all_metadata


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--rebuild",    action="store_true",
                        help="Ignore checkpoints and start from scratch")
    args = parser.parse_args()

    sep("STEP 6 - RAG VECTOR INDEX")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Chunk size : {args.chunk_size:,}  (checkpoint every N interactions)")
    print(f"  Output     : {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total = time.time()

    # 1. load
    interactions, name_map = load_data()

    # 2. format texts
    records = build_texts(interactions, name_map)

    # 3. load model
    sep("LOADING MODEL")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Loaded: {MODEL_NAME}  (dim={model.get_sentence_embedding_dimension()})")

    # 4. embed in chunks with checkpointing
    n_chunks = (len(records) + args.chunk_size - 1) // args.chunk_size
    embed_chunks(records, model, args.chunk_size, args.rebuild)

    # 5. build FAISS index
    build_faiss_index(n_chunks)

    sep("DONE")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Index ready at: {OUTPUT_DIR}")
    print()
    print("  Next: python pipeline/rag_query.py")
