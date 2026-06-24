# Getting your corpus

Distill works with any collection of markdown files — Lenny's newsletter and podcast archive,
Paul Graham essays, internal documentation, book notes, research papers, anything.

---

## Corpus format

Each `.md` file should have YAML frontmatter. The only required fields are `title` and `date`.
All others are optional:

```yaml
---
title: "How to Set Goals That Actually Work"
date: "2024-03-15"
type: newsletter        # any label: newsletter, podcast, article, book, video, etc.
                        # defaults to "article" if omitted
url: "https://example.com/your-article"   # shown as a link in source cards
guest: "Shreyas Doshi"  # for podcast/interview content — shown as the source label
tags: ["goals", "strategy"]
---

Your content here...
```

Files without frontmatter or with empty content are skipped automatically.

---

## Using Lenny's Newsletter & Podcast archive

Lenny's content is available to paid subscribers and is **not included in this repo**.

1. Subscribe at [lennysnewsletter.com](https://www.lennysnewsletter.com) (paid tier)
2. Download the full archive from your subscriber settings
3. Unzip it somewhere (e.g. `~/lenny-data`)
4. Set `DATA_DIR=/path/to/lenny-data` in your `.env`
5. Run `python ingest.py`

The archive comes with its own directory structure. `ingest.py` walks it recursively and
reads `type`, `date`, `title`, and `url` from each file's frontmatter automatically.

---

## Using your own corpus

Any directory of `.md` files works. Structure it however you like — `ingest.py` recurses
into all subdirectories.

```
my-corpus/
  essays/
    first-principles.md
    on-writing.md
  books/
    the-mom-test.md
  notes/
    *.md
```

Set in `.env`:

```
DATA_DIR=/path/to/my-corpus
ASSISTANT_NAME=Sage
CORPUS_DESCRIPTION=Paul Graham essays, YC lecture notes, and startup reading list
```

Then rebuild the index:

```bash
python ingest.py
```

---

## Rebuilding after adding content

Just run `ingest.py` again — it rebuilds from scratch each time. Then restart the RAG
service so it loads the new index.
