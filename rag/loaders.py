"""Turn uploaded files into page-level records that keep their source and page number."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Page:
    text: str
    source: str            # file name shown in citations
    page: int              # 1-based; DOCX/TXT are treated as a single page
    meta: dict = field(default_factory=dict)


SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def load_file(path: str | Path) -> list[Page]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext} (supported: {', '.join(sorted(SUPPORTED))})")

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [Page((p.extract_text() or "").strip(), path.name, i + 1) for i, p in enumerate(reader.pages)]
    elif ext == ".docx":
        import docx2txt

        pages = [Page(docx2txt.process(str(path)).strip(), path.name, 1)]
    else:
        pages = [Page(path.read_text(encoding="utf-8", errors="ignore").strip(), path.name, 1)]

    return [p for p in pages if p.text]


def load_files(paths) -> list[Page]:
    out: list[Page] = []
    for p in paths:
        out.extend(load_file(p))
    return out
