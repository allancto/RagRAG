#!/usr/bin/env python3
"""
RagRAG Setup Script

Builds the vector store from corpus files and fetches community content.
Run this after cloning the repository to set up a working RAG system.

Usage:
    python scripts/setup.py
    python scripts/setup.py --skip-community  # Skip Reddit/StackOverflow fetch
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Set up RagRAG vector store")
    parser.add_argument(
        "--skip-community",
        action="store_true",
        help="Skip fetching Reddit/StackOverflow content"
    )
    parser.add_argument(
        "--skip-papers",
        action="store_true",
        help="Skip ingesting paper summaries from Semantic Scholar"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RagRAG Setup")
    print("=" * 60)

    # Import after path setup
    from src.ingest import ingest_directory
    from src.store import add_chunks, get_store
    from src.semantic_scholar import discover_rag_papers
    from src.community import fetch_all_community_content

    # Check if store already has data
    store = get_store()
    stats = store.get_collection_stats()
    if stats['total_documents'] > 0:
        print(f"\nWarning: Vector store already contains {stats['total_documents']} documents.")
        response = input("Continue and add more? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    all_chunks = []

    # 1. Ingest corpus files (PDFs + framework docs)
    print("\n" + "-" * 60)
    print("Step 1: Ingesting corpus files")
    print("-" * 60)

    corpus_dir = project_root / "corpus"

    print("\nIngesting papers...")
    paper_chunks = ingest_directory(str(corpus_dir / "papers"), extensions=['.pdf'])
    all_chunks.extend(paper_chunks)
    print(f"  -> {len(paper_chunks)} chunks from papers")

    print("\nIngesting framework docs...")
    fw_chunks = ingest_directory(str(corpus_dir / "frameworks"))
    all_chunks.extend(fw_chunks)
    print(f"  -> {len(fw_chunks)} chunks from framework docs")

    # 2. Fetch paper summaries from Semantic Scholar
    if not args.skip_papers:
        print("\n" + "-" * 60)
        print("Step 2: Fetching paper summaries from Semantic Scholar")
        print("-" * 60)
        print("(This provides metadata for papers not in the local corpus)")

        try:
            paper_summaries = discover_rag_papers(min_citations=50, papers_per_topic=5)
            all_chunks.extend(paper_summaries)
            print(f"  -> {len(paper_summaries)} paper summaries")
        except Exception as e:
            print(f"  Warning: Failed to fetch paper summaries: {e}")
            print("  (This is optional - continuing without them)")
    else:
        print("\n[Skipping Semantic Scholar paper summaries]")

    # 3. Fetch community content
    if not args.skip_community:
        print("\n" + "-" * 60)
        print("Step 3: Fetching community content")
        print("-" * 60)
        print("(Reddit and StackOverflow discussions about RAG)")

        try:
            community = fetch_all_community_content(
                reddit_limit=100,
                stackoverflow_limit=100,
                min_reddit_score=5,
                min_so_score=1
            )

            reddit_chunks = community.get('reddit', [])
            so_chunks = community.get('stackoverflow', [])

            all_chunks.extend(reddit_chunks)
            all_chunks.extend(so_chunks)

            print(f"  -> {len(reddit_chunks)} Reddit posts")
            print(f"  -> {len(so_chunks)} StackOverflow questions")
        except Exception as e:
            print(f"  Warning: Failed to fetch community content: {e}")
            print("  (This is optional - continuing without it)")
    else:
        print("\n[Skipping community content]")

    # 4. Add all chunks to vector store
    print("\n" + "-" * 60)
    print("Step 4: Building vector store")
    print("-" * 60)
    print(f"Adding {len(all_chunks)} total chunks...")

    add_chunks(all_chunks)

    # Final stats
    stats = store.get_collection_stats()
    print(f"\nDone! Vector store now contains {stats['total_documents']} documents.")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nYou can now query the RAG:")
    print('  from src.rag import query_rag')
    print('  response = query_rag("How do I evaluate a RAG system?")')
    print("\nNote: You'll need to set ANTHROPIC_API_KEY in .env for generation.")


if __name__ == "__main__":
    main()
