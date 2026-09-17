import io
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. EXTRACT TEXT FROM PDF OR TXT
# ============================================================

def extract_text(uploaded_file):
    """
    Extract text from an uploaded PDF or TXT file.
    """

    file_name = uploaded_file.name.lower()

    # PDF
    if file_name.endswith(".pdf"):
        pdf_bytes = uploaded_file.getvalue()

        reader = PdfReader(io.BytesIO(pdf_bytes))

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text.strip()

    # TXT
    elif file_name.endswith(".txt"):
        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="ignore"
        ).strip()

    else:
        raise ValueError(
            "Unsupported file type. Please upload a PDF or TXT file."
        )


# ============================================================
# 2. SPLIT TEXT INTO CHUNKS
# ============================================================

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split document text into overlapping chunks.

    chunk_size = number of words in each chunk
    overlap = number of words shared between chunks
    """

    if not text or not text.strip():
        return []

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        # Prevent invalid values
        step = chunk_size - overlap

        if step <= 0:
            raise ValueError(
                "chunk_size must be greater than overlap."
            )

        start += step

    return chunks


# ============================================================
# 3. CREATE EMBEDDINGS + FAISS VECTOR DATABASE
# ============================================================

def create_vector_store(chunks):
    """
    Convert document chunks into embeddings
    and store them inside a FAISS index.
    """

    if not chunks:
        raise ValueError(
            "No text chunks available to create vector store."
        )

    # Load embedding model
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # Convert chunks into numerical vectors
    embeddings = model.encode(
        chunks,
        show_progress_bar=False
    )

    # FAISS requires float32
    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to FAISS
    index.add(embeddings)

    return index, model


# ============================================================
# 4. SEARCH SIMILAR DOCUMENT CHUNKS
# ============================================================

def search_similar_chunks(
    question,
    chunks,
    index,
    model,
    top_k=3
):
    """
    Find the most relevant document chunks
    for the user's question.
    """

    if not question.strip():
        return []

    if not chunks:
        return []

    # Convert question into an embedding
    question_embedding = model.encode(
        [question],
        show_progress_bar=False
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    # Don't request more chunks than available
    top_k = min(top_k, len(chunks))

    # Search FAISS
    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for i in indices[0]:

        if 0 <= i < len(chunks):
            results.append(chunks[i])

    return results


# ============================================================
# 5. CREATE CONTEXT FOR THE LLM
# ============================================================

def build_context(relevant_chunks):
    """
    Combine retrieved chunks into one context
    that can be given to the LLM.
    """

    if not relevant_chunks:
        return ""

    context = "\n\n".join(
        relevant_chunks
    )

    return context


# ============================================================
# 6. COMPLETE DOCUMENT PROCESSING FUNCTION
# ============================================================

def process_document(uploaded_file):
    """
    Complete document processing pipeline:

    Upload
       ↓
    Extract text
       ↓
    Create chunks
       ↓
    Create embeddings
       ↓
    Create FAISS vector store
    """

    # Extract text
    text = extract_text(uploaded_file)

    if not text:
        raise ValueError(
            "No readable text was found in the document."
        )

    # Create chunks
    chunks = chunk_text(
        text,
        chunk_size=500,
        overlap=50
    )

    if not chunks:
        raise ValueError(
            "Could not create text chunks."
        )

    # Create vector store
    index, model = create_vector_store(
        chunks
    )

    return {
        "text": text,
        "chunks": chunks,
        "index": index,
        "model": model
    }



# Compatibility name used by advanced/app.py
extract_text_from_file = extract_text