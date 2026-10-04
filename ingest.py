import os

import chromadb
import json
import glob
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# Load environment variables from the .env file.
# We are specifying the path explicitly because of the
# python-dotenv issue we encountered with Python 3.13.
load_dotenv(dotenv_path=".env")


# Retrieve the OpenAI API key from the environment.
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. Check your .env file."
    )


# Create the OpenAI client.
openai_client = OpenAI(api_key=api_key)


# Embedding model used by this RAG tutorial.
EMBEDDING_MODEL = "text-embedding-3-small"

chroma_client = chromadb.PersistentClient(path="./chromadb")

collection = chroma_client.get_or_create_collection(
    name="pc-emporium"
)

def get_embedding(text):
    """
    Convert text into a vector embedding using OpenAI.
    """
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding

def chunk_text(text, min_chunk_length=100):
    """
    Split text into chunks by paragraph (double newline).
    Chunks shorter than min_chunk_length characters are discarded,
    as they are unlikely to contain useful information on their own.
    """
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
    return chunks


def ingest_policies():
    """
    Read policy Markdown files, split them into chunks,
    generate embeddings, and store them in ChromaDB.
    """
    policies_directory = Path("data/policies")

    for policy_path in policies_directory.glob("*.md"):
        print(f"\nProcessing policy: {policy_path.name}")

        text = policy_path.read_text(encoding="utf-8")

        chunks = chunk_text(text)

        print(f"Chunks: {len(chunks)}")

        for index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)

            document_id = (
                f"policy-{policy_path.stem}-{index}"
            )

            collection.add(
                ids=[document_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[
                    {
                        "source": policy_path.name,
                        "type": "policy",
                        "chunk_index": index,
                        "status": (
                            "deprecated"
                            if policy_path.name == "returns-policy-2021.md"
                            else "active"
                        )
                    }
                ],
            )

            print(
                f"  Added chunk {index + 1}/{len(chunks)}"
            )

def ingest_faqs():
    """
    Load FAQs from faqs.json, combine the question and answer into a single
    chunk, generate an embedding, and store in ChromaDB.

    Each FAQ is treated as one chunk because FAQs are already short.
    """
    print("\n--- Ingesting FAQs ---")

    with open("data/faqs.json", "r") as f:
        faqs = json.load(f)
        
    print(f"  Found {len(faqs)} FAQ records")

    for faq in faqs:
        # Combine question and answer so the embedding represents both.
        text = f"Q: {faq['question']}\nA: {faq['answer']}"

        embedding = get_embedding(text)

        print(f"  {faq['question']}: 1 chunk")

        collection.add(
            ids=[faq["id"]],
            embeddings=[embedding],
            documents=[text],
            metadatas=[
                {
                    "source": "faq",
                    "category": faq["category"],
                    "question": faq["question"],
                    "status": "active"
                }
            ]
        )

    print(f"  Done. {len(faqs)} FAQs ingested.")

def ingest_blog_posts():
    """
    Load Markdown blog posts, separate the title from the body,
    chunk the body, generate embeddings, and store each chunk
    in ChromaDB.
    """
    print("\n--- Ingesting blog posts ---")

    blog_files = glob.glob("data/blog-posts/*.md")

    print(f"  Found {len(blog_files)} blog files")

    for filepath in blog_files:
        filename = os.path.basename(filepath)

        # Remove ".md" to produce a simple document slug.
        slug = filename.replace(".md", "")

        with open(filepath, "r") as f:
            content = f.read()

        # First Markdown line contains the title.
        lines = content.split("\n")

        title = lines[0].replace("# ", "").strip()

        # Everything after the title is the blog-post body.
        body = "\n\n".join(lines[1:]).strip()

        chunks = chunk_text(body)

        print(
            f"  {title}: {len(chunks)} chunk(s)"
        )

        for i, chunk in enumerate(chunks):
            chunk_id = f"blog-{slug}-chunk-{i}"

            embedding = get_embedding(chunk)

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[
                    {
                        "source": "blog",
                        "slug": slug,
                        "title": title,
                        "status": "active"
                    }
                ]
            )

    print(
        f"  Done. {len(blog_files)} blog posts ingested."
    )

if __name__ == "__main__":
    print("Starting ingestion...")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    ingest_policies()
    ingest_faqs()
    ingest_blog_posts()

    total = collection.count()

    print(
        f"\nIngestion complete. "
        f"{total} chunks stored in ChromaDB."
    )