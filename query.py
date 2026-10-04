import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables.
load_dotenv(dotenv_path=".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. Check your .env file."
    )


# OpenAI client.
openai_client = OpenAI(api_key=api_key)


# Connect to the vector database created by ingest.py.
chroma_client = chromadb.PersistentClient(path="./chromadb")

collection = chroma_client.get_or_create_collection(
    name="pc-emporium"
)


EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"

# Number of chunks returned by retrieval.
N_RESULTS = 5


def get_embedding(text):
    """
    Generate an embedding for a piece of text using OpenAI.
    """
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def retrieve(question):
    """
    Convert the question into an embedding and query ChromaDB
    for the most semantically similar chunks.
    """
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=N_RESULTS,
        where={
            "status": "active"
        },
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results

def generate_answer(question, results):
    """
    Generate an answer using only the context retrieved from ChromaDB.
    """
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for chunk, metadata in zip(chunks, metadatas):
        source = metadata.get("source", "unknown source")

        context_parts.append(
            f"Source: {source}\n"
            f"Content:\n{chunk}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """
You are a helpful assistant for PC Emporium.

Answer the user's question using only the supplied context.

Follow these rules:

1. Prioritize context that directly answers the user's question.
2. Do not include information from tangentially related policies unless it is
   necessary to answer the question.
3. You may answer using a direct logical implication of the supplied context
   when the conclusion is unambiguous. For example, if the context states that
   a shipping cost is charged or calculated at checkout, then a question asking
   whether that shipping is free should be answered "No."
4. Do not invent, assume, or add facts that are not supported by the context.
5. If multiple retrieved passages conflict, prefer information from the most
   directly relevant active policy.
6. Say "I do not have enough information to answer." only when the supplied
   context does not directly answer the question and does not support a clear,
   unambiguous conclusion.
7. Keep the answer focused on the question asked. Do not summarize every
   retrieved passage.
8. At the end of the answer, include a section titled "Sources:".
9. Under "Sources:", list only the source filenames that directly supported
   the answer.
10. Do not list sources that were retrieved but not actually used.
11. Do not invent source filenames.
12. If the answer is "I do not have enough information to answer.",
    write "Sources:\n- None".
"""

    user_prompt = f"""
Context:

{context}

Question:

{question}
"""

    response = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    test_questions = [
        "What is the returns policy?",
        "How do I return a product?",
        "Can I cancel an order after it has shipped?",
        "What happens if an item is backordered?",
        "How long do I have to return an item?",
        "Do you offer free international shipping?",
        "What is the warranty policy?",
        "Who is the CEO of PC Emporium?"
    ]

    for question in test_questions:
        print("\n" + "=" * 80)
        print(f"Question: {question}")

        results = retrieve(question)

        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        print(f"\nTop {N_RESULTS} retrieved chunks:")

        for i, (chunk, metadata, distance) in enumerate(
            zip(chunks, metadatas, distances),
            start=1
        ):
            print(
                f"\n[{i}] "
                f"Source: {metadata.get('source', 'N/A')} | "
                f"Distance: {round(distance, 4)}"
            )

            print(f"    {chunk[:150]}...")

        answer = generate_answer(
            question,
            results
        )

        print("\n=== Generated Answer ===")
        print(answer)