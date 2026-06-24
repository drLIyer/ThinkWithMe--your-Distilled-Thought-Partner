# Getting the Lenny Data

Distill is built around Lenny Rachitsky's newsletter and podcast archive. The content is
**not included in this repo** — it is proprietary to Lenny and available exclusively to
paid subscribers.

## Step 1 — Subscribe and download the archive

1. Subscribe to [Lenny's Newsletter](https://www.lennysnewsletter.com) (paid tier)
2. In your subscriber settings, download the full archive — you'll get a `.zip` of markdown files
3. Unzip it somewhere on your machine (e.g. `~/lenny-data`)

The archive contains ~650+ markdown files with YAML frontmatter, organized like this:

```
lenny-data/
  01-start-here/
    index.json          ← manifest of all content
    LICENSE.md
  02-newsletters/
    *.md                ← one file per newsletter issue
  03-podcasts/
    *.md                ← one file per podcast episode
```

## Step 2 — Set DATA_DIR in your .env

```
DATA_DIR=/path/to/your/lenny-data
```

## Step 3 — Build the index

```bash
python ingest.py
```

This reads every markdown file, chunks it at 400 words with 50-word overlap, embeds each
chunk with `all-MiniLM-L6-v2`, and writes two files to the repo root:

- `index.faiss` — the vector index
- `chunks.pkl` — chunk text + metadata

This takes 2–5 minutes on first run. You only need to re-run it if the data changes.

## Using a different corpus

Distill is not Lenny-specific. If you have your own collection of markdown files (blog
posts, internal docs, book notes — anything), point `DATA_DIR` at them and run `ingest.py`.
The only requirement is that each file has a `title` and `date` field in its YAML
frontmatter, and a `type` field set to either `newsletter` or `podcast`.

Set `CORPUS_DESCRIPTION` in `.env` to tell the assistant what it knows about, e.g.:

```
CORPUS_DESCRIPTION=Paul Graham's essays on startups, writing, and thinking
ASSISTANT_NAME=PG
```
