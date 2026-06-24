"""
Distill — Weekly Knowledge Update Agent

Fetches new Lenny newsletters and podcasts from RSS, converts to markdown,
adds to the data directory, updates index.json, and incrementally updates
the FAISS vector index without full re-ingestion.

Run manually or weekly:
    .venv312/bin/python3 update_knowledge.py

Options:
    --dry-run   Show what would be added without writing anything
    --force     Re-ingest even if file already exists
"""

import argparse
import json
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import faiss
import frontmatter
import numpy as np
import requests
from markdownify import markdownify as md
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR        = Path("/Users/liyer_1/Downloads/lennys-newsletterpodcastdata-all")
INDEX_JSON      = DATA_DIR / "01-start-here/index.json"
NEWSLETTER_DIR  = DATA_DIR / "02-newsletters"
PODCAST_DIR     = DATA_DIR / "03-podcasts"
FAISS_PATH      = Path("index.faiss")
CHUNKS_PATH     = Path("chunks.pkl")
SLUG_MAP_PATH   = Path("newsletter_slug_map.json")
SEEN_PATH       = Path(".update_seen.json")  # tracks already-ingested URLs

NEWSLETTER_FEED = "https://www.lennysnewsletter.com/feed"
PODCAST_FEED    = NEWSLETTER_FEED  # both come from the same Substack feed

EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_WORDS  = 400
OVERLAP_WORDS = 50
BATCH_SIZE   = 32

NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "itunes":  "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "dc":      "http://purl.org/dc/elements/1.1/",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def word_chunks(text: str) -> list[str]:
    words = text.split()
    chunks, step = [], CHUNK_WORDS - OVERLAP_WORDS
    for i in range(0, len(words), step):
        c = " ".join(words[i: i + CHUNK_WORDS])
        if c.strip():
            chunks.append(c)
    return chunks


def slug_from_url(url: str) -> str:
    m = re.search(r"/p/([a-z0-9][a-z0-9\-]+)", url)
    return m.group(1) if m else ""


def safe_filename(title: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:max_len]


def load_seen() -> set:
    if SEEN_PATH.exists():
        return set(json.loads(SEEN_PATH.read_text()))
    return set()


def save_seen(seen: set):
    SEEN_PATH.write_text(json.dumps(sorted(seen), indent=2))


def parse_date(pub_date: str) -> str:
    try:
        dt = parsedate_to_datetime(pub_date)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def html_to_markdown(html: str) -> str:
    return md(html, heading_style="ATX", bullets="-").strip()


def fetch_feed(url: str) -> list[ET.Element]:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Distill/1.0"})
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    return root.find("channel").findall("item")


# ── Newsletter ingestion ──────────────────────────────────────────────────────

def process_newsletter_item(item: ET.Element) -> dict | None:
    title    = item.findtext("title", "").strip()
    link     = item.findtext("link", "").strip()
    pub_date = item.findtext("pubDate", "")
    encoded  = item.find("content:encoded", NS)
    html     = encoded.text if encoded is not None and encoded.text else ""

    if not html or not title:
        return None

    content   = html_to_markdown(html)
    date_str  = parse_date(pub_date)
    filename  = safe_filename(title)
    slug      = slug_from_url(link)
    rel_path  = f"02-newsletters/{filename}.md"
    full_path = DATA_DIR / rel_path

    # Extract tags from content keywords (simple heuristic)
    tags = ["newsletter"]
    for tag in ["product", "growth", "leadership", "strategy", "ai", "startup", "career", "b2b", "b2c"]:
        if tag in content.lower():
            tags.append(tag)

    # Word count
    word_count = len(content.split())

    return {
        "title":      title,
        "filename":   rel_path,
        "tags":       list(set(tags)),
        "word_count": word_count,
        "date":       date_str,
        "subtitle":   "",
        "link":       link,
        "slug":       slug,
        "_content":   content,
        "_full_path": full_path,
    }


# ── Podcast ingestion ─────────────────────────────────────────────────────────

def process_podcast_item(item: ET.Element) -> dict | None:
    title    = item.findtext("title", "").strip()
    link     = item.findtext("link", "").strip()
    pub_date = item.findtext("pubDate", "")
    encoded  = item.find("content:encoded", NS)
    desc     = item.findtext("description", "")
    html     = (encoded.text if encoded is not None and encoded.text else "") or desc

    if not title:
        return None

    content  = html_to_markdown(html) if html else ""
    date_str = parse_date(pub_date)
    filename = safe_filename(title)
    rel_path = f"03-podcasts/{filename}.md"
    full_path = DATA_DIR / rel_path

    # Extract guest name — Lenny podcast titles often follow "Topic | Guest Name"
    guest = ""
    if "|" in title:
        guest = title.split("|")[-1].strip()
    elif " with " in title.lower():
        guest = title.lower().split(" with ")[-1].strip().title()

    tags = ["podcast"]
    for tag in ["product", "growth", "leadership", "strategy", "ai", "startup", "career"]:
        if tag in content.lower() or tag in title.lower():
            tags.append(tag)

    return {
        "title":      title,
        "filename":   rel_path,
        "tags":       list(set(tags)),
        "word_count": len(content.split()),
        "date":       date_str,
        "description": desc[:200],
        "guest":      guest,
        "link":       link,
        "_content":   content,
        "_full_path": full_path,
    }


# ── Index + FAISS update ──────────────────────────────────────────────────────

def incremental_update(new_items: list[dict], model: SentenceTransformer,
                       item_type: str):
    """Embed new chunks and merge into existing FAISS index."""
    # Load existing
    faiss_index = faiss.read_index(str(FAISS_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    existing_filenames = {c["filename"] for c in chunks}
    new_chunks = []

    for item in new_items:
        rel = item["filename"]
        if rel in existing_filenames:
            continue
        content = item.get("_content", "")
        if not content:
            continue
        for i, chunk_text in enumerate(word_chunks(content)):
            c = {
                "text":      chunk_text,
                "title":     item["title"],
                "type":      item_type,
                "date":      item["date"],
                "tags":      item.get("tags", []),
                "filename":  rel,
                "chunk_idx": i,
            }
            if item_type == "podcast":
                c["guest"] = item.get("guest", "")
            new_chunks.append(c)

    if not new_chunks:
        return 0

    print(f"  Embedding {len(new_chunks)} new chunks…")
    texts = [c["text"] for c in new_chunks]
    embs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False,
                        convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embs)
    faiss_index.add(embs)

    chunks.extend(new_chunks)

    faiss.write_index(faiss_index, str(FAISS_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return len(new_chunks)


def write_markdown(item: dict, item_type: str):
    full_path: Path = item["_full_path"]
    full_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in item.items()
            if not k.startswith("_") and k not in ("link", "slug")}
    meta["type"] = item_type
    post = frontmatter.Post(item["_content"], **meta)
    full_path.write_text(frontmatter.dumps(post))


def update_index_json(new_newsletters: list[dict], new_podcasts: list[dict]):
    idx = json.loads(INDEX_JSON.read_text())
    existing_files = {n["filename"] for n in idx.get("newsletters", [])} | \
                     {p["filename"] for p in idx.get("podcasts", [])}

    added_n = added_p = 0
    for item in new_newsletters:
        if item["filename"] not in existing_files:
            entry = {k: v for k, v in item.items() if not k.startswith("_") and k != "link"}
            idx.setdefault("newsletters", []).insert(0, entry)
            added_n += 1

    for item in new_podcasts:
        if item["filename"] not in existing_files:
            entry = {k: v for k, v in item.items() if not k.startswith("_") and k != "link"}
            idx.setdefault("podcasts", []).insert(0, entry)
            added_p += 1

    if added_n or added_p:
        INDEX_JSON.write_text(json.dumps(idx, indent=2))

    return added_n, added_p


def update_slug_map(new_newsletters: list[dict]):
    slug_map = {}
    if SLUG_MAP_PATH.exists():
        slug_map = json.loads(SLUG_MAP_PATH.read_text())
    added = 0
    for item in new_newsletters:
        if item.get("slug") and item["filename"] not in slug_map:
            slug_map[item["filename"]] = item["slug"]
            added += 1
    if added:
        SLUG_MAP_PATH.write_text(json.dumps(slug_map, indent=2))
    return added


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Update Distill knowledge base from Lenny RSS")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--force",   action="store_true", help="Re-ingest existing items")
    args = parser.parse_args()

    seen = load_seen() if not args.force else set()
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Distill Knowledge Update Agent")
    print(f"{'─'*50}")

    # ── Fetch all items from single feed ──────────────────────────────────
    print("\n🔄 Fetching Lenny RSS feed…")
    new_newsletters = []
    new_podcasts = []
    try:
        items = fetch_feed(NEWSLETTER_FEED)
        print(f"  Found {len(items)} items in feed")
        for item in items:
            link = item.findtext("link", "")
            if link in seen:
                continue
            title = item.findtext("title", "")
            # Classify: items with " | GuestName" pattern are podcast episodes
            is_podcast = ("|" in title and
                          not title.startswith("🧠") and
                          not title.startswith("🎙️ How I AI"))
            if is_podcast:
                processed = process_podcast_item(item)
                if not processed:
                    continue
                full_path: Path = processed["_full_path"]
                if full_path.exists() and not args.force:
                    seen.add(link)
                    continue
                new_podcasts.append(processed)
            else:
                processed = process_newsletter_item(item)
                if not processed:
                    continue
                full_path: Path = processed["_full_path"]
                if full_path.exists() and not args.force:
                    seen.add(link)
                    continue
                new_newsletters.append(processed)
            seen.add(link)

        print(f"  New newsletters: {len(new_newsletters)}")
        for n in new_newsletters:
            print(f"    📰 {n['date']} · {n['title'][:65]}")
        print(f"  New podcasts: {len(new_podcasts)}")
        for p in new_podcasts:
            print(f"    🎙️  {p['date']} · {p['title'][:65]}")
    except Exception as e:
        print(f"  ⚠️  Feed error: {e}")

    if not new_newsletters and not new_podcasts:
        print("\n✅ Nothing new — knowledge base is up to date.")
        save_seen(seen)
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would add {len(new_newsletters)} newsletters + {len(new_podcasts)} podcasts.")
        return

    # ── Write markdown files ───────────────────────────────────────────────
    print("\n✍️  Writing markdown files…")
    for item in new_newsletters:
        write_markdown(item, "newsletter")
        print(f"  Wrote {item['filename']}")
    for item in new_podcasts:
        write_markdown(item, "podcast")
        print(f"  Wrote {item['filename']}")

    # ── Update index.json ──────────────────────────────────────────────────
    added_n, added_p = update_index_json(new_newsletters, new_podcasts)
    print(f"\n📋 Updated index.json (+{added_n} newsletters, +{added_p} podcasts)")

    # ── Update slug map ────────────────────────────────────────────────────
    new_slugs = update_slug_map(new_newsletters)
    print(f"🔗 Updated slug map (+{new_slugs} new slugs)")

    # ── Embed and update FAISS ─────────────────────────────────────────────
    print(f"\n🧠 Loading embedding model…")
    model = SentenceTransformer(EMBED_MODEL)

    new_chunk_count = 0
    for item_type, items in [("newsletter", new_newsletters), ("podcast", new_podcasts)]:
        if items:
            n = incremental_update(items, model, item_type)
            new_chunk_count += n
            print(f"  +{n} chunks from {len(items)} {item_type}(s)")

    # ── Save seen ──────────────────────────────────────────────────────────
    save_seen(seen)

    # ── Knowledge gap report ───────────────────────────────────────────────
    gaps_path = Path("knowledge_gaps.json")
    if gaps_path.exists():
        try:
            gaps = json.loads(gaps_path.read_text())
            print(f"\n🔍 Knowledge Gap Report ({len(gaps)} low-confidence queries logged):")
            # Show top 5 most-asked gap topics
            from collections import Counter
            topics = Counter(g["query"][:60] for g in gaps)
            for topic, count in topics.most_common(5):
                print(f"   × {topic} ({count}x)")
            print(f"   Full list: knowledge_gaps.json")
        except Exception:
            pass

    print(f"\n{'─'*50}")
    print(f"✅ Done! Added {len(new_newsletters)} newsletters + {len(new_podcasts)} podcasts")
    print(f"   {new_chunk_count} new vectors added to FAISS index")
    print(f"\n   Restart Distill to load the updated index:")
    print(f"   pkill -f 'chainlit run' && .venv312/bin/chainlit run chainlit_app.py --port 8081")


if __name__ == "__main__":
    main()
