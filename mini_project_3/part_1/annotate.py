"""
annotate.py
Reads samples_raw.jsonl and produces annotations.csv.
Auto-filled columns: dataset, doc_index, num_chars, num_sentences
Blank columns for manual annotation: topic, quality, correctly_cleaned, private, harmful
Output: annotations.csv
"""

import csv
import json
import re

IN_FILE = "samples_raw.jsonl"
OUT_FILE = "annotations.csv"

FIELDNAMES = [
    "dataset",            # a) dataset source
    "doc_index",          # b) document index
    "topic",              # c) topic — fill in manually
    "quality",            # d) quality 1–10 — fill in manually
    "correctly_cleaned",  # e) correctly cleaned from HTML — yes/no/partial, fill in manually
    "private",            # f) private (A) — yes/no, fill in manually
    "harmful",            # f) harmful (B) — yes/no, fill in manually
    "num_chars",          # g) number of characters
    "num_sentences",      # h) number of sentences
    "text_preview",       # helper: first 300 chars to aid manual annotation
]


def count_sentences(text: str) -> int:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([s for s in sentences if s])


def main():
    records = []
    with open(IN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            records.append({
                "dataset": rec.get("dataset", ""),
                "doc_index": rec.get("doc_index", ""),
                "topic": "",
                "quality": "",
                "correctly_cleaned": "",
                "private": "",
                "harmful": "",
                "num_chars": len(text),
                "num_sentences": count_sentences(text),
                "text_preview": text[:300].replace("\n", " "),
            })

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} rows to {OUT_FILE}")

if __name__ == "__main__":
    main()