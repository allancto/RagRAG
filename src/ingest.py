"""Document ingestion: parse PDFs, Markdown, HTML, and plain text."""

import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict

# For PDF parsing
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# For HTML parsing
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


def load_pdf(file_path: str) -> str:
    """Extract text from PDF using PyPDF2."""
    if not PdfReader:
        raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")

    text = []
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text())
    return "\n".join(text)


def load_markdown(file_path: str) -> str:
    """Load plain text from Markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_html(file_path: str) -> str:
    """Extract text from HTML using BeautifulSoup."""
    if not BeautifulSoup:
        raise ImportError("BeautifulSoup4 not installed. Install with: pip install beautifulsoup4")

    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text()


def load_text(file_path: str) -> str:
    """Load plain text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_document(file_path: str) -> Tuple[str, str]:
    """
    Load any supported document format.

    Returns:
        Tuple of (text_content, doc_type)
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        text = load_pdf(str(file_path))
        return text, 'paper'
    elif suffix in ['.md', '.markdown']:
        text = load_markdown(str(file_path))
        return text, 'framework'
    elif suffix == '.html':
        text = load_html(str(file_path))
        return text, 'framework'
    elif suffix == '.txt':
        text = load_text(str(file_path))
        return text, 'guide'
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def chunk_text(text: str, chunk_size: int = 512, overlap_ratio: float = 0.1) -> List[str]:
    """
    Split text into overlapping chunks.

    Strategy:
    1. Split by double newlines (paragraphs) first
    2. If paragraph > chunk_size, split further by sentences
    3. Add overlap from previous chunk

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in words (approximation)
        overlap_ratio: Fraction of previous chunk to overlap (0.1 = 10%)

    Returns:
        List of text chunks
    """
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    overlap_size = int(chunk_size * overlap_ratio)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_words = para.split()
        para_size = len(para_words)

        # If single paragraph is larger than chunk_size, split it
        if para_size > chunk_size:
            # First, flush current chunk if it has content
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_size = 0

            # Split large paragraph by sentences
            sentences = para.split('. ')
            sentence_chunk = []
            sentence_size = 0

            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_words = sent.split()
                sent_size = len(sent_words)

                if sentence_size + sent_size > chunk_size and sentence_chunk:
                    chunks.append('. '.join(sentence_chunk) + '.')
                    # Add overlap
                    overlap_words = sentence_chunk[-1].split()[:overlap_size]
                    sentence_chunk = overlap_words
                    sentence_size = len(overlap_words)

                sentence_chunk.append(sent)
                sentence_size += sent_size

            if sentence_chunk:
                chunks.append('. '.join(sentence_chunk) + '.')
        else:
            # Add paragraph to current chunk
            if current_size + para_size > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                # Add overlap
                current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                current_size = len(current_chunk)

            current_chunk.append(para)
            current_size += para_size

    # Flush final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def generate_chunk_id(source: str, chunk_index: int, text: str) -> str:
    """
    Generate unique ID for a chunk based on source and content hash.

    Args:
        source: Document source filename
        chunk_index: Index of chunk in document
        text: Chunk text content

    Returns:
        Unique identifier
    """
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    source_name = Path(source).stem
    return f"{source_name}_{chunk_index}_{content_hash}"


def ingest_document(
    file_path: str,
    chunk_size: int = 512,
    overlap_ratio: float = 0.1
) -> List[Dict[str, str]]:
    """
    Ingest a document and return chunks with metadata.

    Args:
        file_path: Path to document
        chunk_size: Target chunk size in words
        overlap_ratio: Overlap ratio for chunks

    Returns:
        List of dicts with 'text', 'metadata', and 'id' keys
    """
    # Load document
    text, doc_type = load_document(file_path)
    source = Path(file_path).name

    # Chunk text
    chunks = chunk_text(text, chunk_size=chunk_size, overlap_ratio=overlap_ratio)

    # Generate metadata for each chunk
    result = []
    for idx, chunk in enumerate(chunks):
        chunk_id = generate_chunk_id(source, idx, chunk)
        metadata = {
            'source': source,
            'doc_type': doc_type,
            'chunk_index': str(idx),
            'chunk_id': chunk_id
        }
        result.append({
            'text': chunk,
            'metadata': metadata,
            'id': chunk_id
        })

    return result


def ingest_directory(
    directory: str,
    chunk_size: int = 512,
    overlap_ratio: float = 0.1,
    extensions: List[str] = None
) -> List[Dict[str, str]]:
    """
    Ingest all documents in a directory.

    Args:
        directory: Path to corpus directory
        chunk_size: Target chunk size in words
        overlap_ratio: Overlap ratio for chunks
        extensions: File extensions to process (default: .pdf, .md, .html, .txt)

    Returns:
        List of all chunks from all documents
    """
    if extensions is None:
        extensions = ['.pdf', '.md', '.markdown', '.html', '.txt']

    all_chunks = []
    dir_path = Path(directory)

    for file_path in dir_path.rglob('*'):
        if file_path.suffix.lower() not in extensions:
            continue

        try:
            chunks = ingest_document(str(file_path), chunk_size=chunk_size, overlap_ratio=overlap_ratio)
            all_chunks.extend(chunks)
            print(f"✓ Ingested {file_path.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"✗ Failed to ingest {file_path.name}: {e}")

    return all_chunks
