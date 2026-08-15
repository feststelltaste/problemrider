#!/usr/bin/env python3
"""Generate semantically similar solutions using the shared embedding analyzer.

Usage:
    python scripts/calculate_related_solutions.py --dry-run
    python scripts/calculate_related_solutions.py --use-local --local-url http://localhost:1234
    python scripts/calculate_related_solutions.py --file asynchronous-operations

Embeddings are cached locally in embeddings/solutions/ and are not committed.
"""

import argparse

from calculate_related_problems import (
    DEFAULT_LOCAL_EMBEDDING_URL,
    MIN_SIMILARITY,
    SimpleEmbeddingAnalyzer,
)


SOLUTION_EMBEDDING_SECTIONS = (
    ("How to Apply", "how_to_apply"),
    ("Tradeoffs", "tradeoffs"),
    ("How It Could Be", "how_it_could_be"),
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate related solutions using Qwen3-Embedding-0.6B"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use a local embedding service instead of sentence-transformers",
    )
    parser.add_argument(
        "--local-url",
        type=str,
        help=f"URL of local embedding service (default: {DEFAULT_LOCAL_EMBEDDING_URL})",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process only a specific solution file (for example, 'asynchronous-operations')",
    )
    args = parser.parse_args()

    analyzer = SimpleEmbeddingAnalyzer(
        problems_dir="_solutions",
        embeddings_dir="embeddings/solutions",
        related_field="related_solutions",
        item_label="solutions",
        embedding_sections=SOLUTION_EMBEDDING_SECTIONS,
        use_local=args.use_local,
        local_url=args.local_url,
    )
    analyzer.load_items()
    analyzer.create_embeddings()
    analyzer.update_all_files(
        dry_run=args.dry_run,
        min_similarity=MIN_SIMILARITY,
        specific_file=args.file,
    )


if __name__ == "__main__":
    main()
