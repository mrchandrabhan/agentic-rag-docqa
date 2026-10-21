"""Split pages into overlapping chunks while carrying source/page metadata into every chunk."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loaders import Page


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    page: int

    def citation(self) -> str:
        return f"{self.source}, p.{self.page}"


def chunk_pages(pages: list[Page], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(len(chunks), piece, page.source, page.page))
    return chunks
