"""Community content fetching from Reddit, StackOverflow, and other sources."""

import requests
import time
from typing import List, Dict, Optional
from html import unescape
import re


# Rate limiting
REQUEST_DELAY = 1.0  # Be nice to APIs


def clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities
    text = unescape(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =============================================================================
# REDDIT
# =============================================================================

REDDIT_HEADERS = {'User-Agent': 'RagRAG/1.0 (Educational RAG Corpus Builder)'}

RAG_SUBREDDITS = [
    'LocalLLaMA',
    'MachineLearning',
    'LangChain',
    'artificial',
    'learnmachinelearning'
]


def fetch_reddit_posts(
    subreddit: str,
    query: str,
    limit: int = 25,
    min_score: int = 5
) -> List[Dict]:
    """
    Fetch posts from a subreddit matching a query.

    Args:
        subreddit: Subreddit name
        query: Search query
        limit: Max posts to fetch
        min_score: Minimum upvotes

    Returns:
        List of post dictionaries
    """
    url = f'https://www.reddit.com/r/{subreddit}/search.json'
    params = {
        'q': query,
        'restrict_sr': 'on',
        'limit': limit,
        'sort': 'relevance',
        't': 'all'  # All time
    }

    response = requests.get(url, params=params, headers=REDDIT_HEADERS)
    if response.status_code != 200:
        return []

    data = response.json()
    posts = data.get('data', {}).get('children', [])

    # Filter by score and convert to our format
    result = []
    for post in posts:
        p = post.get('data', {})
        score = p.get('score', 0)

        if score < min_score:
            continue

        result.append({
            'title': p.get('title', ''),
            'body': p.get('selftext', ''),
            'score': score,
            'url': f"https://reddit.com{p.get('permalink', '')}",
            'subreddit': subreddit,
            'num_comments': p.get('num_comments', 0),
            'created_utc': p.get('created_utc', 0)
        })

    return result


def reddit_post_to_chunk(post: Dict) -> Dict:
    """Convert a Reddit post to a chunk for ingestion."""
    title = post.get('title', 'Untitled')
    body = post.get('body', '')

    # Build text
    parts = [
        f"Reddit Discussion: {title}",
        f"Subreddit: r/{post.get('subreddit', 'unknown')} | {post.get('score', 0)} upvotes | {post.get('num_comments', 0)} comments",
        "",
        body if body else "(Link post - no text content)"
    ]

    text = "\n".join(parts)

    # Build metadata
    metadata = {
        'source': post.get('url', ''),
        'doc_type': 'community_reddit',
        'title': title,
        'subreddit': post.get('subreddit', ''),
        'score': str(post.get('score', 0))
    }

    # Generate ID from URL
    url = post.get('url', '')
    post_id = url.split('/')[-3] if '/comments/' in url else url[-20:]

    return {
        'text': text,
        'metadata': metadata,
        'id': f"reddit_{post_id}"
    }


def fetch_reddit_rag_content(
    queries: List[str] = None,
    posts_per_query: int = 25,
    min_score: int = 5
) -> List[Dict]:
    """
    Fetch RAG-related content from Reddit.

    Args:
        queries: Search queries (default: RAG-related terms)
        posts_per_query: Posts to fetch per query per subreddit
        min_score: Minimum upvotes

    Returns:
        List of chunks ready for ingestion
    """
    if queries is None:
        queries = [
            'RAG retrieval augmented generation',
            'vector database embedding',
            'chunking strategy',
            'LlamaIndex',
            'LangChain RAG'
        ]

    seen_urls = set()
    all_chunks = []

    for subreddit in RAG_SUBREDDITS:
        for query in queries:
            print(f"  r/{subreddit}: '{query}'")
            try:
                posts = fetch_reddit_posts(subreddit, query, limit=posts_per_query, min_score=min_score)

                for post in posts:
                    url = post.get('url', '')
                    if url not in seen_urls:
                        seen_urls.add(url)
                        chunk = reddit_post_to_chunk(post)
                        all_chunks.append(chunk)

                time.sleep(REQUEST_DELAY)

            except Exception as e:
                print(f"    Error: {e}")

    return all_chunks


# =============================================================================
# STACKOVERFLOW
# =============================================================================

def fetch_stackoverflow_questions(
    query: str,
    pagesize: int = 25,
    min_score: int = 1,
    tagged: str = None
) -> List[Dict]:
    """
    Fetch questions from StackOverflow.

    Args:
        query: Search query
        pagesize: Number of results
        min_score: Minimum vote score
        tagged: Comma-separated tags to filter by

    Returns:
        List of question dictionaries
    """
    url = 'https://api.stackexchange.com/2.3/search/advanced'
    params = {
        'order': 'desc',
        'sort': 'votes',
        'q': query,
        'site': 'stackoverflow',
        'pagesize': pagesize,
        'filter': '!nNPvSNdWme'  # Include body, answers
    }

    if tagged:
        params['tagged'] = tagged

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []

    data = response.json()
    items = data.get('items', [])

    # Filter by score
    return [q for q in items if q.get('score', 0) >= min_score]


def stackoverflow_to_chunk(question: Dict) -> Dict:
    """Convert a StackOverflow question to a chunk."""
    title = clean_html(question.get('title', 'Untitled'))
    body = clean_html(question.get('body', ''))

    # Get tags
    tags = question.get('tags', [])
    tags_str = ', '.join(tags[:5])

    # Build text
    parts = [
        f"StackOverflow Question: {title}",
        f"Tags: {tags_str} | {question.get('score', 0)} votes | {question.get('answer_count', 0)} answers",
        "",
        body[:2000] if body else "(No body)"  # Truncate very long bodies
    ]

    # Include accepted answer if available
    if question.get('is_answered') and question.get('accepted_answer_id'):
        parts.append("\n[Has accepted answer]")

    text = "\n".join(parts)

    # Build metadata
    q_id = question.get('question_id', 'unknown')
    metadata = {
        'source': question.get('link', f'https://stackoverflow.com/q/{q_id}'),
        'doc_type': 'community_stackoverflow',
        'title': title,
        'tags': tags_str,
        'score': str(question.get('score', 0))
    }

    return {
        'text': text,
        'metadata': metadata,
        'id': f"so_{q_id}"
    }


def fetch_stackoverflow_rag_content(
    queries: List[str] = None,
    questions_per_query: int = 25,
    min_score: int = 1
) -> List[Dict]:
    """
    Fetch RAG-related content from StackOverflow.

    Args:
        queries: Search queries
        questions_per_query: Questions per query
        min_score: Minimum votes

    Returns:
        List of chunks ready for ingestion
    """
    if queries is None:
        queries = [
            'RAG retrieval augmented generation',
            'vector database embedding python',
            'LangChain retrieval',
            'LlamaIndex',
            'chromadb',
            'sentence transformers embedding',
            'semantic search python'
        ]

    seen_ids = set()
    all_chunks = []

    for query in queries:
        print(f"  SO: '{query}'")
        try:
            questions = fetch_stackoverflow_questions(query, pagesize=questions_per_query, min_score=min_score)

            for q in questions:
                q_id = q.get('question_id')
                if q_id not in seen_ids:
                    seen_ids.add(q_id)
                    chunk = stackoverflow_to_chunk(q)
                    all_chunks.append(chunk)

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"    Error: {e}")

    return all_chunks


# =============================================================================
# COMBINED FETCHER
# =============================================================================

def fetch_all_community_content(
    reddit_limit: int = 100,
    stackoverflow_limit: int = 100,
    min_reddit_score: int = 5,
    min_so_score: int = 1
) -> Dict[str, List[Dict]]:
    """
    Fetch community content from all sources.

    Args:
        reddit_limit: Target number of Reddit posts
        stackoverflow_limit: Target number of SO questions
        min_reddit_score: Minimum Reddit upvotes
        min_so_score: Minimum SO votes

    Returns:
        Dict with 'reddit' and 'stackoverflow' chunk lists
    """
    print("Fetching Reddit content...")
    reddit_chunks = fetch_reddit_rag_content(min_score=min_reddit_score)
    reddit_chunks = reddit_chunks[:reddit_limit]
    print(f"  Got {len(reddit_chunks)} Reddit posts")

    print("\nFetching StackOverflow content...")
    so_chunks = fetch_stackoverflow_rag_content(min_score=min_so_score)
    so_chunks = so_chunks[:stackoverflow_limit]
    print(f"  Got {len(so_chunks)} SO questions")

    return {
        'reddit': reddit_chunks,
        'stackoverflow': so_chunks
    }
