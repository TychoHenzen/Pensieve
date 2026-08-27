"""Fetch a pinned prose snapshot from `monology/pile-uncopyrighted` for filler text.

Downloads `val.jsonl.zst` (471 MB compressed) and decompresses it as a
stream, reading one document at a time and stopping as soon as it holds
roughly `--max-words` words. It never expands the whole file, because it
stops reading the stream the moment it has enough. The result is a plain
jsonl file, one `{"text": ...}` object per line, the shape
`eval.stream.corpus.load_corpus` reads. The repository stores this script
and the snapshot's sha256, never the snapshot itself; `.gitignore` keeps
the corpus directory out of version control.

`zstandard` decodes the `.zst` stream, since Python 3.13 ships no zstd
decoder. It is an optional extra, installed with `pip install -e .[corpus]`.

Usage:
    python scripts/fetch_corpus.py --max-words 5000000 --out data/corpus/pile-val.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path

SOURCE_URL = (
    "https://huggingface.co/datasets/monology/pile-uncopyrighted/"
    "resolve/main/val.jsonl.zst"
)
DEFAULT_MAX_WORDS = 5_000_000
DEFAULT_OUT = Path("data/corpus/pile-val.jsonl")
READ_CHUNK_BYTES = 1 << 16


def _iter_documents(url: str) -> Iterator[dict]:
    """Yield decoded JSON documents from the remote `.zst` stream, one at a time.

    Reads and decompresses the response in fixed-size chunks rather than
    pulling the whole 471 MB file into memory, and stops requesting more
    bytes as soon as the caller stops iterating.
    """
    try:
        import zstandard
    except ImportError as exc:
        raise SystemExit(
            "zstandard is required to decompress the corpus stream. "
            "Install it with: pip install -e .[corpus]"
        ) from exc

    with urllib.request.urlopen(url) as response:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(response) as reader:
            buffer = b""
            while True:
                chunk = reader.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                for raw_line in lines:
                    line = raw_line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
            tail = buffer.strip()
            if tail:
                yield json.loads(tail)


def fetch_corpus(max_words: int, out_path: Path, url: str = SOURCE_URL) -> Path:
    """Write documents from `url` to `out_path` until `max_words` words are held.

    Stops as soon as the running word count reaches `max_words`, so it
    never reads or decompresses more of the source stream than it needs.
    Returns `out_path`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    word_count = 0
    with out_path.open("w", encoding="utf-8") as out_file:
        for document in _iter_documents(url):
            text = document.get("text", "")
            words = text.split()
            if not words:
                continue
            out_file.write(json.dumps({"text": text}) + "\n")
            word_count += len(words)
            if word_count >= max_words:
                break
    return out_path


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--url", default=SOURCE_URL)
    args = parser.parse_args(argv)

    out_path = fetch_corpus(args.max_words, args.out, args.url)
    digest = _sha256_of(out_path)
    print(f"wrote {out_path} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
