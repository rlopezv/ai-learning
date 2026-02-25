from dataclasses import dataclass

@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_id: int
