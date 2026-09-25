from __future__ import annotations

import csv
import hashlib
import heapq
import sys
from pathlib import Path


TRAIN = Path(sys.argv[1])
OUT = Path(sys.argv[2])

N = 25
SEED = "C54_MINING_V0_TRAIN25_001"


def priority(fen: str) -> int:
    payload = (
        SEED.encode("utf-8")
        + b"\0"
        + fen.encode("utf-8")
    )

    return int.from_bytes(
        hashlib.sha256(payload).digest(),
        "big",
    )


# Keep the 25 smallest hashes without loading 816k rows
# into memory. Heap stores negative priority so the largest
# currently-selected value stays at the top.
heap = []

with TRAIN.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
    )

    if reader.fieldnames is None:
        raise RuntimeError("TRAIN TSV has no header")

    required = {
        "fen",
        "source",
        "sf_score_type",
        "sf_score",
        "sf_bound",
        "sf_wdl_w",
        "sf_wdl_d",
        "sf_wdl_l",
        "sf_bestmove",
    }

    missing = required - set(reader.fieldnames)

    if missing:
        raise RuntimeError(
            f"Missing TRAIN columns: {sorted(missing)}"
        )

    total = 0

    for row_number, row in enumerate(reader, start=2):
        total += 1

        fen = row["fen"]
        p = priority(fen)

        item = (
            -p,
            row_number,
            row,
        )

        if len(heap) < N:
            heapq.heappush(
                heap,
                item,
            )
        else:
            # Since priority is stored negative, a numerically
            # larger item means a smaller real SHA priority.
            if item[0] > heap[0][0]:
                heapq.heapreplace(
                    heap,
                    item,
                )


selected = []

for negative_priority, row_number, row in heap:
    selected.append(
        (
            -negative_priority,
            row_number,
            row,
        )
    )

selected.sort(
    key=lambda x: x[0]
)

if len(selected) != N:
    raise RuntimeError(
        f"Expected {N} positions, got {len(selected)}"
    )


fieldnames = [
    "batch_index",
    "selection_sha256",
    "train_row_number",
]

# Preserve every original TRAIN column after provenance fields.
fieldnames += reader.fieldnames


with OUT.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    for batch_index, (
        p,
        row_number,
        row,
    ) in enumerate(selected, start=1):

        selection_hash = f"{p:064x}"

        out = {
            "batch_index": batch_index,
            "selection_sha256": selection_hash,
            "train_row_number": row_number,
        }

        out.update(row)

        writer.writerow(out)


print("TRAIN rows scanned :", total)
print("Selected           :", len(selected))
print("Seed               :", SEED)
print("Output             :", OUT)

print()
print("SOURCE DISTRIBUTION:")

counts = {}

for _, _, row in selected:
    source = row["source"]
    counts[source] = counts.get(source, 0) + 1

for source in sorted(counts):
    print(
        f"{source:12s}: {counts[source]}"
    )

print()
print("SELECTED POSITIONS:")

for i, (p, row_number, row) in enumerate(
    selected,
    start=1,
):
    print(
        f"{i:2d} "
        f"row={row_number:7d} "
        f"source={row['source']:8s} "
        f"frozen={row['sf_score_type']} "
        f"{row['sf_score']:>5s} "
        f"move={row['sf_bestmove']:5s} "
        f"sha={p:064x}"
    )
