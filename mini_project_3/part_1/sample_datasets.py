"""
sample_datasets.py
Streams 50 documents from each of 4 pretraining datasets and writes them to samples_raw.jsonl.
Run: python sample_datasets.py
Output: samples_raw.jsonl  (one JSON object per line)
"""

import json
import sys
from datasets import load_dataset

SAMPLES_PER_SOURCE = 50
OUT_FILE = "samples_raw.jsonl"

SOURCES = [
    {
        "name": "fineweb",
        "loader_args": dict(
            path="HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        ),
        "text_key": "text",
    },
    {
        "name": "finepdfs",
        "loader_args": dict(
            path="HuggingFaceFW/finepdfs",
            split="train",
            streaming=True,
        ),
        "text_key": "text",
    },
    {
        "name": "c4",
        "loader_args": dict(
            path="allenai/c4",
            name="en",
            split="train",
            streaming=True,
        ),
        "text_key": "text",
    },
    {
        "name": "redpajama",
        "loader_args": dict(
            path="json",
            data_files="hf://datasets/togethercomputer/RedPajama-Data-V2/sample/documents/2023-06/0000/en_middle.json.gz",
            split="train",
            streaming=True,
        ),
        "text_key": "raw_content",
    },
]


def sample_source(source: dict, n: int) -> list[dict]:
    print(f"  Loading {source['name']} ...", flush=True)
    ds = load_dataset(**source["loader_args"])
    records = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        records.append({
            "dataset": source["name"],
            "doc_index": i,
            "text": row.get(source["text_key"], ""),
            "url": row.get("url", row.get("source", "")),
            "id": str(row.get("id", row.get("doc_id", ""))),
        })
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n}", flush=True)
    return records


def main():
    all_records = []
    for source in SOURCES:
        print(f"\n[{source['name']}]")
        try:
            records = sample_source(source, SAMPLES_PER_SOURCE)
            all_records.extend(records)
            print(f"{len(records)} samples collected.")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_records)} records to {OUT_FILE}")


if __name__ == "__main__":
    main()