"""Main RAG orchestrator and CLI interface."""

import sys
from src.retrieve import retrieve_and_format
from src.generate import generate_response
from config import TOP_K


def ask(question: str, top_k: int = TOP_K, verbose: bool = False) -> str:
    """
    Full RAG pipeline: retrieve relevant chunks, then generate response.

    Args:
        question: User's question
        top_k: Number of chunks to retrieve
        verbose: Print retrieval info

    Returns:
        Generated answer
    """
    # Retrieve relevant context
    context, results = retrieve_and_format(question, top_k=top_k)

    if verbose:
        print(f"\nRetrieved {len(results)} chunks from: {', '.join(results.sources)}")
        print("-" * 40)

    # Generate response
    answer = generate_response(question, context)

    return answer


def main():
    """Interactive CLI for RagRAG."""
    print("=" * 50)
    print("  RagRAG - Ask questions about building RAGs")
    print("=" * 50)
    print("\nType 'quit' or 'exit' to leave.\n")

    while True:
        try:
            question = input("\nAsk: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            print("\nThinking...")
            answer = ask(question, verbose=True)

            print(f"\nAnswer:\n{answer}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()
