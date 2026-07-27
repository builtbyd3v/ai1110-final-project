from __future__ import annotations

import argparse
from pathlib import Path

from src.rag import (
    DEFAULT_EMBEDDING_MODEL,
    build_documents,
    embed_texts,
    load_cache,
    save_cache,
)
from src.recommender import load_songs


def build_embedding_cache(
    csv_path: str = "data/songs.csv",
    cache_path: str = ".cache/song_embeddings.json",
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> Path:
    songs = load_songs(csv_path)
    docs = build_documents(songs)
    embeddings = embed_texts([doc.text for doc in docs], model=model)
    mapped = {
        doc.song_id: embeddings[doc.text]
        for doc in docs
    }
    output = Path(cache_path)
    save_cache(docs, mapped, output)
    print(f"Saved {len(docs)} embeddings to {output}")
    return output


def inspect_cache(cache_path: str = ".cache/song_embeddings.json") -> None:
    cache = load_cache(Path(cache_path))
    print(f"Documents: {len(cache['documents'])}")
    print(f"Embeddings: {len(cache['embeddings'])}")
    first = cache["documents"][0]
    print(f"First doc: {first['song_id']} - {first['title']} by {first['artist']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or inspect the local song embedding cache.")
    parser.add_argument("--csv", default="data/songs.csv")
    parser.add_argument("--cache", default=".cache/song_embeddings.json")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    if args.inspect:
        inspect_cache(args.cache)
    else:
        build_embedding_cache(args.csv, args.cache, args.model)


if __name__ == "__main__":
    main()
