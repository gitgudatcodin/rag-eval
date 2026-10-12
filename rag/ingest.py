"""Document loading and chunking."""
import glob
import os

from . import config


def _read_txt(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _read_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents(docs_dir=None):
    """Return list of (doc_name, text) for .txt/.md/.pdf files."""
    docs_dir = docs_dir or config.DOCS_DIR
    docs = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*"))):
        name = os.path.basename(path)
        if path.endswith((".txt", ".md")):
            docs.append((name, _read_txt(path)))
        elif path.endswith(".pdf"):
            docs.append((name, _read_pdf(path)))
    return docs


def chunk_text(text, size=None, overlap=None):
    """Word-aware character chunking with overlap. Returns list of chunk strings."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    words, chunks, cur, cur_len = text.split(), [], [], 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and cur_len + add > size:
            chunks.append(" ".join(cur))
            # carry overlap words into the next chunk
            keep, keep_len = [], 0
            for ww in reversed(cur):
                if keep_len + len(ww) + 1 > overlap:
                    break
                keep.append(ww)
                keep_len += len(ww) + 1
            cur, cur_len = list(reversed(keep)), keep_len
        cur.append(w)
        cur_len += add
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_chunks(docs_dir=None):
    """Return list of dicts: {id, doc, chunk_index, text}."""
    out = []
    for doc_name, text in load_documents(docs_dir):
        for i, ch in enumerate(chunk_text(text)):
            out.append({"id": f"{doc_name}#{i}", "doc": doc_name,
                        "chunk_index": i, "text": ch})
    return out
