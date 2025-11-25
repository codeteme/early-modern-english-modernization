"""Setup: Create parallel corpus and compute word weights (ROBUST VERSION)."""

import os
import csv
import re
import math
import difflib
from collections import Counter

# --- Configuration ---
SIMILARITY_THRESHOLD = 0.4  # Lowered slightly to catch more valid pairs
WINDOW_SIZE = 10  # Look-ahead window for alignment
STOPWORDS = {
    "the",
    "and",
    "to",
    "of",
    "a",
    "i",
    "my",
    "in",
    "is",
    "that",
    "you",
    "it",
    "not",
    "be",
    "with",
    "for",
    "your",
    "but",
    "me",
    "he",
    "she",
    "his",
    "her",
}


def is_junk(line):
    """Filter out metadata, headers, and stage directions to prevent drift."""
    line = line.strip()
    if not line:
        return True
    if line.startswith(("FILE:", "Title:", "Source:", "Version:", "=====")):
        return True
    # Filter stage directions (often start with Exit/Enter or are in brackets)
    if re.match(r"^(Exit|Exeunt|Enter)\b", line, re.IGNORECASE):
        return True
    if line.startswith(("[", "(")):
        return True
    return False


def read_and_chunk_files(filepath):
    """Reads a file and splits it into chunks based on 'FILE:' markers."""
    chunks = []
    current_chunk = []

    print(f"Reading {filepath}...")
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("FILE:"):
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = []
                continue
            if not is_junk(line):
                current_chunk.append(line)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def align_lines_greedy(old_lines, new_lines):
    """Aligns lines using SequenceMatcher to skip mismatches."""
    aligned = []
    i = 0
    j = 0
    N, M = len(old_lines), len(new_lines)

    while i < N and j < M:
        best_score = -1.0
        best_j = -1

        # Look ahead in NEW lines to find match
        search_limit = min(j + WINDOW_SIZE, M)
        for k in range(j, search_limit):
            ratio = difflib.SequenceMatcher(None, old_lines[i], new_lines[k]).ratio()
            if ratio > best_score:
                best_score = ratio
                best_j = k

        if best_score >= SIMILARITY_THRESHOLD:
            aligned.append((old_lines[i], new_lines[best_j]))
            i += 1
            j = best_j + 1
        else:
            # No match found, skip this OLD line (it might be an extra line)
            i += 1

    return aligned


def create_parallel_corpus(
    old_file="../../data/old_merged.txt",
    new_file="../../data/new_merged.txt",
    train_split=0.8,
):
    print("=" * 60 + "\nSTEP 1: CREATING PARALLEL CORPUS (ROBUST)\n" + "=" * 60)

    old_chunks = read_and_chunk_files(old_file)
    new_chunks = read_and_chunk_files(new_file)

    all_pairs = []
    limit = min(len(old_chunks), len(new_chunks))

    print(f"\nAligning {limit} file sections (this may take a moment)...")
    for k in range(limit):
        pairs = align_lines_greedy(old_chunks[k], new_chunks[k])
        all_pairs.extend(pairs)
        if k % 5 == 0:
            print(".", end="", flush=True)

    print(f"\nTotal aligned pairs: {len(all_pairs)}")

    split_idx = int(len(all_pairs) * train_split)

    with open("../../data/train_parallel.txt", "w", encoding="utf-8") as f:
        for old, new in all_pairs[:split_idx]:
            f.write(f"{old}\n{new}\n")

    with open("../../data/test_parallel.txt", "w", encoding="utf-8") as f:
        for old, new in all_pairs[split_idx:]:
            f.write(f"{old}\n{new}\n")

    print(f"✓ Train: {split_idx} pairs\n✓ Test: {len(all_pairs) - split_idx} pairs\n")


def compute_word_weights(
    train_file="../../data/train_parallel.txt",
    old_corpus="../../data/old_merged.txt",
    new_corpus="../../data/new_merged.txt",
):
    print("=" * 60 + "\nSTEP 2: COMPUTING WORD WEIGHTS\n" + "=" * 60)

    # Load background frequencies
    with open(new_corpus, "r", encoding="utf-8", errors="ignore") as f:
        new_freq = Counter(re.findall(r"\b\w+\b", f.read().lower()))

    pair_counts = Counter()
    old_word_total = Counter()

    # Bad targets to strictly avoid mapping to
    BAD_TARGETS = {"the", "a", "an", "of", "to", "in", "and"}

    with open(train_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        old_words = re.findall(r"\b\w+\b", lines[i].lower())
        new_words = re.findall(r"\b\w+\b", lines[i + 1].lower())

        # Only align if lengths are reasonably close
        if abs(len(old_words) - len(new_words)) < 4:
            limit = min(len(old_words), len(new_words))
            for k in range(limit):
                old_w, new_w = old_words[k], new_words[k]

                # --- THE FIX: Stopword Filter ---
                # Prevent mapping rare words (len>3) to common articles
                if old_w not in BAD_TARGETS and len(old_w) > 3 and new_w in BAD_TARGETS:
                    continue

                pair_counts[(old_w, new_w)] += 1
                old_word_total[old_w] += 1

    weighted_pairs = []
    for (old, modern), count in pair_counts.items():
        if count < 2:
            continue  # Noise filter

        p_trans = count / old_word_total[old]
        channel_cost = -math.log(p_trans)
        lm_cost = (
            -math.log(new_freq[modern] / sum(new_freq.values()))
            if new_freq[modern]
            else 10
        )
        weight = 0.7 * channel_cost + 0.3 * lm_cost

        weighted_pairs.append({"old": old, "modern": modern, "weight": weight})

    weighted_pairs.sort(key=lambda x: x["weight"])

    with open("../../data/word_weights.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["old", "modern", "weight"])
        writer.writeheader()
        writer.writerows(weighted_pairs)

    print(f"✓ Saved {len(weighted_pairs)} clean word weights")
    print("Top 10 translations:")
    for p in weighted_pairs[:10]:
        print(f"  {p['old']} -> {p['modern']} ({p['weight']:.2f})")


if __name__ == "__main__":
    create_parallel_corpus()
    compute_word_weights()
    print("\n Robust Setup complete! Now run: python 01_translate_heuristic.py")
