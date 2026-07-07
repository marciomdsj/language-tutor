"""Real content fetcher — articles from RSS feeds and web sources.

This module provides real-world English content for exercises like article
summaries, reading comprehension, and writing prompts.  It tries multiple
sources in order of reliability:

    1. RSS feeds (TechCrunch, Ars Technica, HackerNews, Nature)
    2. LLM-generated content as fallback (if all feeds fail)

The fetched content is trimmed to a suitable length for a language exercise
(~200-400 words) and returned with metadata (title, source, URL).
"""

from __future__ import annotations

import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.request import urlopen, Request

# Target length for exercise content (words)
TARGET_WORDS = 300
MAX_WORDS = 500

# RSS feeds — tech-focused, high quality English
RSS_FEEDS = [
    {
        "name": "Ars Technica",
        "url": "https://feeds.arstechnica.com/arstechnica/index",
        "topic": "technology",
    },
    {
        "name": "TechCrunch",
        "url": "https://techcrunch.com/feed/",
        "topic": "technology",
    },
    {
        "name": "Hacker News (best)",
        "url": "https://hnrss.org/best",
        "topic": "technology",
    },
    {
        "name": "Nature News",
        "url": "https://www.nature.com/nature.rss",
        "topic": "science",
    },
]


@dataclass
class Article:
    """A fetched article ready for use in an exercise."""

    title: str
    content: str
    source: str
    url: str
    topic: str
    word_count: int


def fetch_article(topic: str | None = None) -> Article:
    """Fetch a real article from RSS feeds, or generate one as fallback.

    Tries each RSS feed in random order until one succeeds.  If all
    fail (network issues, parsing errors), falls back to asking the
    LLM to generate a realistic article.

    Args:
        topic: Optional topic filter (e.g. "technology", "science").
              If None, picks from any available feed.

    Returns:
        An Article ready for use in an exercise.
    """
    feeds = RSS_FEEDS.copy()
    if topic:
        feeds = [f for f in feeds if f["topic"] == topic] or feeds
    random.shuffle(feeds)

    for feed_info in feeds:
        try:
            article = _fetch_from_rss(feed_info)
            if article:
                return article
        except Exception:
            continue

    # Fallback: LLM generates content
    return _generate_article(topic or "technology")


def _fetch_from_rss(feed_info: dict) -> Article | None:
    """Fetch a random article from an RSS feed.

    Args:
        feed_info: Dict with 'name', 'url', 'topic' keys.

    Returns:
        An Article, or None if fetching/parsing fails.
    """
    req = Request(
        feed_info["url"],
        headers={"User-Agent": "LanguageTutor/1.0 (educational project)"},
    )
    with urlopen(req, timeout=10) as response:
        raw = response.read()

    root = ET.fromstring(raw)

    # Parse RSS items (handles both RSS 2.0 and Atom)
    items = root.findall(".//item") or root.findall(
        ".//{http://www.w3.org/2005/Atom}entry"
    )
    if not items:
        return None

    # Pick a random article from recent items
    item = random.choice(items[:10])

    title = _get_text(item, "title") or _get_text(
        item, "{http://www.w3.org/2005/Atom}title"
    )
    link = _get_text(item, "link") or _get_attr(
        item, "{http://www.w3.org/2005/Atom}link", "href"
    )
    description = (
        _get_text(item, "description")
        or _get_text(item, "{http://www.w3.org/2005/Atom}summary")
        or _get_text(item, "{http://www.w3.org/2005/Atom}content")
        or ""
    )

    if not title or not description:
        return None

    # Clean HTML tags from description
    clean_text = re.sub(r"<[^>]+>", "", description).strip()
    clean_text = re.sub(r"\s+", " ", clean_text)

    # Trim to target length
    words = clean_text.split()
    if len(words) > MAX_WORDS:
        clean_text = " ".join(words[:TARGET_WORDS]) + "..."

    # If RSS only gave a short summary, expand it with LLM
    if len(clean_text.split()) < 80:
        clean_text = _expand_article(title.strip(), clean_text, feed_info["topic"])

    return Article(
        title=title.strip(),
        content=clean_text,
        source=feed_info["name"],
        url=link or "",
        topic=feed_info["topic"],
        word_count=len(clean_text.split()),
    )


def _expand_article(title: str, summary: str, topic: str) -> str:
    """Expand a short RSS summary into a full article using the LLM.

    Keeps the real title and summary as the basis, but asks the LLM to
    write a plausible ~200 word article expanding on the topic.

    Args:
        title: The real article title from RSS.
        summary: The short summary from RSS.
        topic: The topic category.

    Returns:
        Expanded article text (~200 words).
    """
    from language_tutor.llm import _call_llm, _get_content

    response = _call_llm(
        messages=[
            {
                "role": "user",
                "content": (
                    f"Based on this real news headline and summary, write a plausible "
                    f"~200 word article as if from a {topic} news outlet.\n\n"
                    f"Headline: {title}\n"
                    f"Summary: {summary}\n\n"
                    f"Write the article body only (no title). "
                    f"Write in English at C1 level. Be factual and informative."
                ),
            }
        ],
    )
    return _get_content(response).strip()


def _generate_article(topic: str) -> Article:
    """Ask the LLM to generate a realistic article as fallback.

    Args:
        topic: The topic to write about.

    Returns:
        An LLM-generated Article.
    """
    from language_tutor.llm import _call_llm, _get_content

    response = _call_llm(
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write a short article (~200 words) about a recent development "
                    f"in {topic}. Write as if it were from a real news outlet. "
                    f"Include a clear title on the first line. "
                    f"Write in English at a level appropriate for C1 readers."
                ),
            }
        ],
    )
    text = _get_content(response)
    lines = text.strip().split("\n", 1)
    title = lines[0].strip().strip("#").strip("*").strip()
    body = lines[1].strip() if len(lines) > 1 else text

    return Article(
        title=title,
        content=body,
        source="AI-generated",
        url="",
        topic=topic,
        word_count=len(body.split()),
    )


def _get_text(element: ET.Element, tag: str) -> str | None:
    """Get text content of a child element."""
    child = element.find(tag)
    return child.text if child is not None and child.text else None


def _get_attr(element: ET.Element, tag: str, attr: str) -> str | None:
    """Get an attribute of a child element."""
    child = element.find(tag)
    return child.get(attr) if child is not None else None
