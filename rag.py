import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def chunk_markdown(content, max_chunk_size=1000):
    # Split content into lines
    lines = content.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0
    main_section = ""
    sub_section = ""

    for line in lines:
        # Detect headers
        if line.startswith("## "):
            # If we have a previous chunk, save it
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            main_section = line
            current_chunk = [line]
            current_size = len(line)
        elif line.startswith("### "):
            # If current chunk is too large, save it
            if current_size > max_chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [main_section, line]
                current_size = len(main_section) + len(line)
            else:
                sub_section = line
                current_chunk.append(line)
                current_size += len(line)
        else:
            # Regular content line
            line_size = len(line)
            if current_size + line_size > max_chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                # Start new chunk with context
                current_chunk = [main_section, sub_section, line]
                current_size = len(main_section) + len(sub_section) + line_size
            else:
                current_chunk.append(line)
                current_size += line_size

    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def process_cheatsheet(filepath):
    with open(filepath, "r") as file:
        content = file.read()

    # Create chunks
    chunks = chunk_markdown(content)

    # Print or process chunks
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print("-" * 50)
        print(chunk)
        print("-" * 50)

    return chunks


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class RAG:
    """Retrieval Augmented Generation system for question answering using a cheatsheet."""

    def __init__(
        self,
        client: OpenAI,
        cheatsheet_path: str = "cheatsheet.md",
        embedding_model: str = "text-embedding-3-small",
        max_chunk_size: int = 1000,
    ):
        """Initialize the RAG system.

        Args:
            client: OpenAI client instance
            cheatsheet_path: Path to the markdown cheatsheet
            embedding_model: Name of the embedding model to use
            max_chunk_size: Maximum size of each chunk in characters
        """
        self.client = client
        self.embedding_model = embedding_model
        self.max_chunk_size = max_chunk_size
        self._chunks = self._load_and_process_cheatsheet(cheatsheet_path)
        self._chunk_embeddings = self._compute_embeddings(self._chunks)

    def _load_and_process_cheatsheet(self, filepath: str) -> list:
        """Load and chunk the cheatsheet."""
        with open(filepath, "r") as file:
            content = file.read()
        return chunk_markdown(content, self.max_chunk_size)

    def _compute_embeddings(self, texts: list) -> list:
        """Compute embeddings for a list of texts."""
        return [self._get_embedding(chunk) for chunk in texts]

    def _get_embedding(self, text: str) -> list:
        """Get embedding for a single text."""
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def _compute_similarity(
        self, query_embedding: list, chunk_embedding: list
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        return cosine_similarity(query_embedding, chunk_embedding)

    def get_relevant_chunks(self, query: str, top_k: int = 1) -> list:
        """Get the top-k most relevant chunks for a query.

        Args:
            query: The query text
            top_k: Number of chunks to return

        Returns:
            List of most relevant chunks
        """
        query_embedding = self._get_embedding(query)
        similarities = [
            self._compute_similarity(query_embedding, emb)
            for emb in self._chunk_embeddings
        ]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self._chunks[i] for i in top_indices]


if __name__ == "__main__":
    filepath = "cheatsheet.md"
    chunks = process_cheatsheet(filepath)
