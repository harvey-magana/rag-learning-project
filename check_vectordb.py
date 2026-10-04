import chromadb


chroma_client = chromadb.PersistentClient(
    path="./chromadb"
)

collection = chroma_client.get_or_create_collection(
    name="pc-emporium"
)


print("=== Total chunks in ChromaDB ===")
print(f"{collection.count()} chunks stored")
print()


print("=== Chunks by source ===")

policy_results = collection.get(
    where={"type": "policy"}
)

faq_results = collection.get(
    where={"source": "faq"}
)

blog_results = collection.get(
    where={"source": "blog"}
)

print(f"  policy: {len(policy_results['ids'])} chunks")
print(f"  faq: {len(faq_results['ids'])} chunks")
print(f"  blog: {len(blog_results['ids'])} chunks")
print()


print("=== Sample chunk ===")

results = collection.get(
    limit=1,
    include=[
        "embeddings",
        "documents",
        "metadatas"
    ]
)

active_results = collection.get(
    where={"status": "active"}
)

deprecated_results = collection.get(
    where={"status": "deprecated"}
)

print(
    f"Active chunks: {len(active_results['ids'])}"
)

print(
    f"Deprecated chunks: {len(deprecated_results['ids'])}"
)

if not results["documents"]:
    print("No documents were found in the collection.")

else:
    document = results["documents"][0]
    embedding = results["embeddings"][0]
    metadata = results["metadatas"][0]

    print("Metadata:")
    print(f"  {metadata}")
    print()

    print("Text content (first 300 characters):")
    print(f"  {document[:300]}...")
    print()

    print("Vector embedding:")
    print(f"  Dimensions: {len(embedding)} numbers")
    print(
        f"  First 10 values: "
        f"{[round(float(v), 4) for v in embedding[:10]]}"
    )