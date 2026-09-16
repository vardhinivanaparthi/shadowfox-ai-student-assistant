from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# ---------------- PDF TEXT EXTRACTION ----------------

def extract_text_from_pdf(file):
    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# ---------------- TEXT CHUNKING ----------------

def chunk_text(text, chunk_size=500):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ---------------- EMBEDDING MODEL ----------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------- FAISS INDEX CREATION ----------------

def create_faiss_index(chunks):
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embeddings


# ---------------- SIMILARITY SEARCH ----------------

def search_similar_chunks(question, chunks, index, top_k=3):
    question_embedding = model.encode([question])

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for i in indices[0]:
        if i < len(chunks):
            results.append(chunks[i])

    return results