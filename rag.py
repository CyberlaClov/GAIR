import json
import os
import re

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


# def chunk_markdown(content, max_chunk_size=1000):
#     # Split content into lines
#     lines = content.split("\n")
#     chunks = []
#     current_chunk = []
#     current_size = 0
#     main_section = ""
#     sub_section = ""

#     for line in lines:
#         # Detect headers
#         if line.startswith("## "):
#             # If we have a previous chunk, save it
#             if current_chunk:
#                 chunks.append("\n".join(current_chunk))
#             main_section = line
#             current_chunk = [line]
#             current_size = len(line)
#         elif line.startswith("### "):
#             # If current chunk is too large, save it
#             if current_size > max_chunk_size and current_chunk:
#                 chunks.append("\n".join(current_chunk))
#                 current_chunk = [main_section, line]
#                 current_size = len(main_section) + len(line)
#             else:
#                 sub_section = line
#                 current_chunk.append(line)
#                 current_size += len(line)
#         else:
#             # Regular content line
#             line_size = len(line)
#             if current_size + line_size > max_chunk_size and current_chunk:
#                 chunks.append("\n".join(current_chunk))
#                 # Start new chunk with context
#                 current_chunk = [main_section, sub_section, line]
#                 current_size = len(main_section) + len(sub_section) + line_size
#             else:
#                 current_chunk.append(line)
#                 current_size += line_size

#     # Add the last chunk if there's anything left
#     if current_chunk:
#         chunks.append("\n".join(current_chunk))

#     return chunks


# def process_cheatsheet(filepath):
#     with open(filepath, "r") as file:
#         content = file.read()

#     # Create chunks
#     chunks = chunk_markdown(content)

#     # Print or process chunks
#     for i, chunk in enumerate(chunks):
#         print(f"\nChunk {i+1}:")
#         print("-" * 50)
#         print(chunk)
#         print("-" * 50)

#     return chunks


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
        chunks_to_load: int = 1,
    ):
        """Initialize the RAG system.

        Args:
            client: OpenAI client instance
            cheatsheet_path: Path to the markdown cheatsheet
            embedding_model: Name of the embedding model to use
            max_chunk_size: Maximum size of each chunk in characters
            embeddings_path: Path to save/load embeddings
        """
        self.client = client
        self.embedding_model = embedding_model
        self.cheatsheet_path = cheatsheet_path
        self.embeddings_path = cheatsheet_path.replace(".md", "") + "_embeddings.json"
        self.max_chunk_size = max_chunk_size
        self._chunks = self._load_and_process_cheatsheet(cheatsheet_path)
        self.chunks_count = len(self._chunks)
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Default GPT-4 encoding
        self._chunk_embeddings = self._load_or_compute_embeddings()
        self.chunks_to_load = chunks_to_load

    def _load_and_process_cheatsheet(self, filepath: str) -> list:
        """Load and chunk the cheatsheet using hierarchical chunking."""
        with open(filepath, "r") as file:
            content = file.read()
        return self._hierarchical_chunk_markdown(content, self.max_chunk_size)

    def _hierarchical_chunk_markdown(self, content, max_chunk_size=1000, overlap=100):
        """Chunks markdown content using a hierarchical approach with overlapping context.

        This chunking method:
        1. Preserves the hierarchy of sections (##, ###, etc.)
        2. Creates overlapping chunks to maintain context across chunk boundaries
        3. Keeps section headers with their content
        4. Ensures better context preservation for API documentation

        Args:
            content (str): The markdown content to chunk
            max_chunk_size (int): Maximum size of each chunk
            overlap (int): Number of characters to overlap between chunks

        Returns:
            list: List of chunks with preserved hierarchy
        """
        # Split into major sections first (##)
        major_sections = re.split(r"(?=\n## )", content)
        chunks = []

        for major_section in major_sections:
            if not major_section.strip():
                continue

            # Get the major section title
            major_title = re.match(r"## [^\n]+", major_section)
            major_title = major_title.group(0) if major_title else ""

            # Split into subsections
            subsections = re.split(r"(?=\n### )", major_section)

            current_chunk = []
            current_size = 0

            for i, subsection in enumerate(subsections):
                if not subsection.strip():
                    continue

                # Always include major title for context if this is a new chunk
                if not current_chunk:
                    current_chunk.append(major_title)
                    current_size = len(major_title)

                # Add subsection
                if current_size + len(subsection) > max_chunk_size:
                    # Save current chunk
                    chunks.append("\n".join(current_chunk))

                    # Start new chunk with overlap from previous content
                    overlap_content = (
                        current_chunk[-1][-overlap:] if current_chunk else ""
                    )
                    current_chunk = [major_title]
                    if overlap_content:
                        current_chunk.append(overlap_content)
                    current_chunk.append(subsection)
                    current_size = sum(len(c) for c in current_chunk)
                else:
                    current_chunk.append(subsection)
                    current_size += len(subsection)

            # Add any remaining content
            if current_chunk:
                chunks.append("\n".join(current_chunk))

        return chunks

    def _compute_embeddings(self, texts: list, show_progress: bool = True) -> list:
        """Compute embeddings for a list of texts."""
        if show_progress:
            return [
                self._get_embedding(chunk)
                for chunk in tqdm(texts, desc="Computing embeddings")
            ]
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

    def _load_or_compute_embeddings(self) -> list:
        """Load embeddings from file or compute them if not available."""
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, "r") as file:
                return json.load(file)
        else:
            embeddings = self._compute_embeddings(self._chunks)
            self._save_embeddings(embeddings)
            return embeddings

    def _save_embeddings(self, embeddings: list):
        """Save embeddings to a file."""
        with open(self.embeddings_path, "w") as file:
            json.dump(embeddings, file)

    def get_relevant_chunks(self, query: str) -> list:
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

        top_indices = np.argsort(similarities)[-self.chunks_to_load :][::-1]
        # print(top_indices)

        return [self._chunks[i] for i in top_indices]

    def count_tokens(self) -> int:
        """Count the total number of tokens in the cheatsheet.

        Returns:
            int: Total number of tokens across all chunks
        """
        total_tokens = 0
        for chunk in self._chunks:
            total_tokens += len(self.encoding.encode(chunk))
        return total_tokens


if __name__ == "__main__":
    filepath = "full_reliability_documentation.md"
    client = OpenAI()
    rag = RAG(client, cheatsheet_path=filepath, chunks_to_load=3)
    print(f"Number of chunks: {rag.chunks_count}")
    print(f"Total tokens: {rag.count_tokens()}")
    query = """Each of the following test acceleration models is most often  used with only one varying stress type, except for which?
[Choices]: [a] Arrhenius | [b] Coffin-Manson | [c] Inverse Power Law  | [d] Eyring"""
    contexts = rag.get_relevant_chunks(query)

    combined_context = "\n---\n".join(contexts)
    print(combined_context)
