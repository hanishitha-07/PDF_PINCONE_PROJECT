"""
PDF Embedding and Semantic Search System
========================================

Workflow:
  1. Read PDF files from the pdfs folder
  2. Extract readable text with PyMuPDF (fitz)
  3. Split text into overlapping chunks (800 chars, 150 overlap)
  4. Load the all-MiniLM-L6-v2 Sentence Transformers model
  5. Generate 384-dimensional embeddings for each chunk
  6. Connect to Pinecone, create the index if needed, upload vectors
  7. Semantic search: type questions, get top 3 matches (type 'exit' to quit)

Configuration is read from the .env file (API key is never hardcoded).
"""

import os
import glob
import fitz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ---------------------------- Configuration ---------------------------- #
PDF_FOLDER = "pdfs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
INDEX_NAME = "pdf-search-index"
SIMILARITY_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
UPSERT_BATCH_SIZE = 100


# ---------------------------- Step 1: PDF Text Extraction ---------------------------- #
def extract_text(pdf_path):
    """Extract all readable text from a PDF file using PyMuPDF.

    Returns the extracted text, or an empty string if the file is missing
    or contains no readable text.
    """
    if not os.path.exists(pdf_path):
        print(f"  [SKIP] File not found: {pdf_path}")
        return ""

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        print(f"  [SKIP] Could not open {pdf_path}: {exc}")
        return ""

    text_parts = []
    for page in document:
        try:
            text_parts.append(page.get_text())
        except Exception as exc:
            print(f"  [WARN] Could not read a page in {pdf_path}: {exc}")

    document.close()

    text = "\n".join(text_parts)
    if not text.strip():
        print(f"  [WARN] No readable text found in {pdf_path} (empty PDF).")
        return ""

    return text


def find_pdf_files(folder):
    """Return a sorted list of PDF file paths in the given folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(glob.glob(os.path.join(folder, "*.pdf")))


# ---------------------------- Step 2: Text Chunking ---------------------------- #
def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of the given size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end >= text_length:
            break

        start += chunk_size - overlap

    return chunks


# ---------------------------- Step 3: Embedding Model ---------------------------- #
def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    """Load the Sentence Transformers model."""
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print("Embedding model loaded.")
    return model


def generate_embeddings(model, chunks):
    """Generate embeddings for a list of text chunks.

    Returns a list of embedding vectors (each a list of floats).
    """
    print(f"Generating embeddings for {len(chunks)} chunks ...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.tolist()
    dim = len(embeddings[0]) if embeddings else 0
    print(f"Generated {len(embeddings)} embeddings, each {dim}-dimensional.")
    return embeddings


# ---------------------------- Step 4: Pinecone Integration ---------------------------- #
def get_pinecone_client():
    """Connect to Pinecone using the API key stored in the .env file."""
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", INDEX_NAME)

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY not found in .env file. "
            "Create a .env file based on .env.example and add your Pinecone API key."
        )

    print("Connecting to Pinecone ...")
    pc = Pinecone(api_key=api_key)
    print("Connected to Pinecone.")
    return pc, index_name


def index_is_ready(pc, index_name):
    """Check whether a Pinecone index is ready to accept requests."""
    try:
        desc = pc.describe_index(name=index_name)
        status = getattr(desc, "status", None)
        if status is None:
            return True
        # Pinecone may expose readiness as a 'ready' flag or a 'state' string.
        if isinstance(status, dict):
            if status.get("ready") is True:
                return True
            if status.get("state", "").lower() in ("ready", "active"):
                return True
        else:
            if getattr(status, "ready", False) is True:
                return True
            state = getattr(status, "state", "")
            if str(state).lower() in ("ready", "active"):
                return True
    except Exception:
        return False
    return False


def wait_for_index_ready(pc, index_name, timeout_seconds=120):
    """Wait until the Pinecone index is ready, up to the given timeout."""
    import time

    print(f"Waiting for index '{index_name}' to become ready ...")
    start = time.time()
    while time.time() - start < timeout_seconds:
        if index_is_ready(pc, index_name):
            print("Index is ready.")
            return True
        time.sleep(5)
    print(
        f"[WARN] Index '{index_name}' did not become ready within "
        f"{timeout_seconds} seconds. Proceeding anyway; upload may fail."
    )
    return False


def ensure_index(pc, index_name):
    """Create the Pinecone index if it does not exist; verify compatibility otherwise."""
    existing_indexes = pc.list_indexes()
    existing_names = [idx.name for idx in existing_indexes.indexes]

    if index_name not in existing_names:
        print(f"Index '{index_name}' does not exist. Creating it ...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric=SIMILARITY_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        wait_for_index_ready(pc, index_name)
        print(
            f"Index '{index_name}' created "
            f"(dimension={EMBEDDING_DIMENSION}, metric={SIMILARITY_METRIC})."
        )
    else:
        print(f"Index '{index_name}' already exists. Verifying compatibility ...")
        for idx in existing_indexes.indexes:
            if idx.name == index_name:
                if idx.dimension != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Index '{index_name}' has dimension {idx.dimension}, "
                        f"but this project requires {EMBEDDING_DIMENSION}. "
                        f"Delete the index or change PINECONE_INDEX_NAME in .env."
                    )
                if str(idx.metric).lower() != SIMILARITY_METRIC.lower():
                    raise ValueError(
                        f"Index '{index_name}' uses metric '{idx.metric}', "
                        f"but this project requires '{SIMILARITY_METRIC}'."
                    )
                print(
                    f"Compatibility verified: dimension={idx.dimension}, "
                    f"metric={idx.metric}."
                )
                break

    return pc.Index(index_name)


def stable_vector_id(pdf_filename, chunk_number):
    """Build a unique, stable vector ID from the PDF filename and chunk number."""
    base_name = os.path.splitext(os.path.basename(pdf_filename))[0]
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_name)
    return f"{safe_name}_chunk_{chunk_number}"


def get_existing_vector_ids(index, ids):
    """Return the set of vector IDs that already exist in the index.

    Used to avoid re-uploading vectors that are already present.
    """
    existing = set()
    for i in range(0, len(ids), 1000):
        batch = ids[i:i + 1000]
        try:
            response = index.fetch(ids=batch)
            vectors = getattr(response, "vectors", None) or {}
            existing.update(vectors.keys())
        except Exception as exc:
            print(f"  [WARN] Could not check existing vectors: {exc}. Will upload all vectors.")
            return set()
    return existing


def upload_embeddings(index, pdf_filename, chunks, embeddings):
    """Upload chunk embeddings to Pinecone in batches, skipping existing ones."""
    total_chunks = len(chunks)
    print(
        f"Preparing to upload {total_chunks} vector(s) from "
        f"'{os.path.basename(pdf_filename)}' ..."
    )

    planned_ids = [stable_vector_id(pdf_filename, i) for i in range(total_chunks)]
    existing_ids = get_existing_vector_ids(index, planned_ids)

    vectors = []
    for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = stable_vector_id(pdf_filename, chunk_index)
        if vector_id in existing_ids:
            continue  # already uploaded; skip to avoid duplicate work
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": chunk,
                "filename": os.path.basename(pdf_filename),
                "chunk": chunk_index,
            },
        })

    if not vectors:
        print("  All vectors already exist in the index. Nothing to upload.")
        return 0

    uploaded = 0
    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i:i + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch)
        uploaded += len(batch)
        print(f"  Uploaded batch {i // UPSERT_BATCH_SIZE + 1} ({len(batch)} vectors)")

    print(
        f"  Upload complete: {uploaded} new vector(s) from "
        f"'{os.path.basename(pdf_filename)}'."
    )
    return uploaded


# ---------------------------- Step 5: Semantic Search ---------------------------- #
def semantic_search(index, model):
    """Interactive semantic search loop. Type 'exit' to quit."""
    print("\n" + "=" * 60)
    print("Semantic Search ready!")
    print("Type a question and press Enter to search your PDFs.")
    print("Type 'exit' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            print("  Please enter a question (or 'exit' to quit).\n")
            continue
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Embed the question using the same model.
        query_embedding = model.encode([question])[0].tolist()

        # Search Pinecone for the top 3 most similar chunks.
        results = index.query(
            top_k=3,
            vector=query_embedding,
            include_metadata=True,
        )

        matches = getattr(results, "matches", None) or []
        if not matches:
            print("  No matching chunks found.\n")
            continue

        print(f"\nTop {len(matches)} results for: '{question}'\n")
        for rank, match in enumerate(matches, start=1):
            metadata = match.metadata or {}
            score = getattr(match, "score", None)
            print(f"--- Result {rank} (similarity score: {score}) ---")
            print(f"  PDF: {metadata.get('filename', 'unknown')}")
            print(f"  Chunk: {metadata.get('chunk', 'unknown')}")
            print(f"  Text: {metadata.get('text', '')}")
            print()


# ---------------------------- Step 6: Main Program ---------------------------- #
def main():
    print("PDF Embedding and Semantic Search System")
    print("-" * 50)

    # Step 1: Find and extract text from PDFs.
    pdf_files = find_pdf_files(PDF_FOLDER)
    if not pdf_files:
        print(f"No PDF files found in the '{PDF_FOLDER}' folder.")
        print("Please place one or more PDF files inside the pdfs folder and run again.")
        return

    print(f"Found {len(pdf_files)} PDF file(s) in '{PDF_FOLDER}'.")

    all_records = []  # list of (pdf_path, chunk_index, chunk_text)
    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path}")
        text = extract_text(pdf_path)
        if not text:
            continue

        chunks = create_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  Extracted {len(text)} characters -> {len(chunks)} chunk(s)")

        for chunk_index, chunk in enumerate(chunks):
            all_records.append((pdf_path, chunk_index, chunk))

        total_chunks += len(chunks)

    print(f"\nTotal chunks across all PDFs: {total_chunks}")
    if total_chunks == 0:
        print("No chunks were created. Nothing to embed or upload.")
        return

    # (Optional) Save chunks to a text file for inspection.
    with open("chunks.txt", "w", encoding="utf-8") as file:
        for pdf_path, chunk_index, chunk in all_records:
            file.write(f"\n--- {os.path.basename(pdf_path)} - CHUNK {chunk_index + 1} ---\n")
            file.write(chunk)
            file.write("\n")
    print("All chunks saved to chunks.txt")

    # Step 3: Load embedding model.
    model = load_embedding_model(EMBEDDING_MODEL_NAME)

    # Group chunks by PDF so each PDF's chunks are embedded together.
    embeddings_by_pdf = {}
    for pdf_path, chunk_index, chunk in all_records:
        embeddings_by_pdf.setdefault(pdf_path, []).append(chunk)

    # Step 4: Connect to Pinecone and ensure the index exists.
    pc, index_name = get_pinecone_client()
    index = ensure_index(pc, index_name)

    # Step 5: Upload embeddings per PDF.
    total_uploaded = 0
    for pdf_path in pdf_files:
        if pdf_path not in embeddings_by_pdf:
            continue
        chunks = embeddings_by_pdf[pdf_path]
        embeddings = generate_embeddings(model, chunks)
        total_uploaded += upload_embeddings(index, pdf_path, chunks, embeddings)

    print(f"\nTotal new vectors uploaded: {total_uploaded}")
    try:
        stats = index.describe_index_stats()
        total_vectors = getattr(stats, "total_vector_count", None)
        if total_vectors is not None:
            print(f"Index now contains approximately {total_vectors} vector(s).")
    except Exception as exc:
        print(f"  [WARN] Could not retrieve index stats: {exc}")

    # Step 6: Semantic search loop.
    semantic_search(index, model)


if __name__ == "__main__":
    main()
