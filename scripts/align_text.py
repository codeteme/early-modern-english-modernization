#!/usr/bin/env python3
"""
make_pairs.py

Build aligned Early-Modern -> Modern spelling phrase pairs
from data/old_merged.txt and data/new_merged.txt.

Run from anywhere:
    python scripts/make_pairs.py

Outputs:
    aligned_old_to_modern.csv  (written next to this script)
"""

import re
import csv
from pathlib import Path
from difflib import SequenceMatcher

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

OLD_PATH = DATA_DIR / "old_merged.txt"
NEW_PATH = DATA_DIR / "new_merged.txt"
OUT_PATH = SCRIPT_DIR / "aligned_old_to_modern.csv"

# ---------------------------------------------------------
# Patterns
# ---------------------------------------------------------

HEADER_PREFIXES = (
    "FILE:", "Title:", "Version:", "Source:"
)

SPEAKER_RE = re.compile(
    r"^\s*([A-Z][a-zA-Z']+|\d+\.\s*[A-Z][a-zA-Z']+)\.\s*$"
)
ALLCAPS_STAGE_RE = re.compile(r"^[A-Z '\-]{3,}$")
ENTER_EXIT_RE = re.compile(r"^\s*(Enter|Exit|Exeunt|Flourish)\b", re.I)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def is_header_or_meta(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith("="):
        return True
    if any(s.startswith(p) for p in HEADER_PREFIXES):
        return True
    return False

def is_speaker_only(line: str) -> bool:
    return bool(SPEAKER_RE.match(line.strip()))

def is_stage_direction_only(line: str) -> bool:
    s = line.strip()
    if ENTER_EXIT_RE.match(s):
        return True
    if ALLCAPS_STAGE_RE.match(s) and len(s.split()) <= 4:
        return True
    return False

def clean_line(line: str) -> str:
    s = line.strip()
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s

def extract_phrases(text: str, merge_blocks: bool = True):
    lines = text.splitlines()
    phrases = []
    buf = []

    def flush():
        nonlocal buf
        if buf:
            phrases.append(" ".join(buf).strip())
            buf = []

    for raw in lines:
        line = raw.rstrip("\n")

        if is_header_or_meta(line) or is_speaker_only(line) or is_stage_direction_only(line):
            flush()
            continue

        s = clean_line(line)
        if not s:
            flush()
            continue

        if merge_blocks:
            buf.append(s)
        else:
            phrases.append(s)

    flush()
    return phrases

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def tokenize_words(text):
    """Split text into words, preserving punctuation context"""
    # Split on whitespace but keep track of original positions
    tokens = []
    for word in text.split():
        # Remove leading/trailing punctuation but keep the word
        clean = re.sub(r'^[^\w]+|[^\w]+$', '', word)
        if clean:
            tokens.append(clean)
    return tokens

def align_words_within_phrase(old_phrase, new_phrase, min_sim=0.3):
    """Align words within matched phrase pairs"""
    old_words = tokenize_words(old_phrase)
    new_words = tokenize_words(new_phrase)

    word_pairs = []

    # Use dynamic programming for better word alignment
    # Simple greedy approach with local search
    i = j = 0
    while i < len(old_words) and j < len(new_words):
        old_word = old_words[i]
        new_word = new_words[j]

        # Calculate similarity
        s = sim(old_word.lower(), new_word.lower())

        # Look ahead slightly to find better matches
        best_match = (s, i, j)
        for di in range(min(3, len(old_words) - i)):
            for dj in range(min(3, len(new_words) - j)):
                if di == 0 and dj == 0:
                    continue
                test_s = sim(old_words[i + di].lower(), new_words[j + dj].lower())
                # Prefer matches that don't skip too many words
                penalty = abs(di - dj) * 0.1
                score = test_s - penalty
                if score > best_match[0]:
                    best_match = (test_s, i + di, j + dj)

        _, best_i, best_j = best_match

        # Add the pair if similarity is reasonable
        if best_match[0] >= min_sim or (
            # Allow lower similarity for very similar length words
            len(old_words[best_i]) == len(new_words[best_j]) and best_match[0] >= 0.25
        ):
            word_pairs.append((old_words[best_i], new_words[best_j]))

        i = best_i + 1
        j = best_j + 1

    return word_pairs

def align_with_local_repair(old_phrases, new_phrases, window=5, min_sim=0.4):
    """First align phrases, then align words within phrases"""
    phrase_aligned = []
    word_aligned = []
    i = j = 0

    while i < len(old_phrases) and j < len(new_phrases):
        o = old_phrases[i]
        n = new_phrases[j]
        s0 = sim(o.lower(), n.lower())

        if s0 >= min_sim:
            phrase_aligned.append((o, n))
            i += 1
            j += 1
            continue

        # drift repair
        best = (s0, i, j)
        for di in range(window + 1):
            for dj in range(window + 1):
                if di == 0 and dj == 0:
                    continue
                ii, jj = i + di, j + dj
                if ii < len(old_phrases) and jj < len(new_phrases):
                    s = sim(old_phrases[ii].lower(), new_phrases[jj].lower())
                    if s > best[0]:
                        best = (s, ii, jj)

        best_s, best_i, best_j = best
        if best_s >= min_sim:
            # move to alignment point
            i = best_i
            j = best_j
            phrase_aligned.append((old_phrases[i], new_phrases[j]))
            i += 1
            j += 1
        else:
            # fail-safe skip
            i += 1
            j += 1

    # Now extract word-level alignments from phrase pairs
    print(f"Aligned {len(phrase_aligned)} phrase pairs, extracting word pairs...")

    for old_phrase, new_phrase in phrase_aligned:
        word_pairs = align_words_within_phrase(old_phrase, new_phrase)
        word_aligned.extend(word_pairs)

    # Deduplicate and filter
    seen = set()
    filtered = []
    for old_word, new_word in word_aligned:
        # Skip if identical
        if old_word.lower() == new_word.lower():
            continue
        # Skip if too short
        if len(old_word) < 3 or len(new_word) < 3:
            continue

        # Quality check: require minimum similarity
        similarity = sim(old_word.lower(), new_word.lower())
        if similarity < 0.4:
            continue

        # Quality check: length shouldn't differ too much
        len_ratio = min(len(old_word), len(new_word)) / max(len(old_word), len(new_word))
        if len_ratio < 0.5:
            continue

        # Quality check: should share some characters
        old_set = set(old_word.lower())
        new_set = set(new_word.lower())
        overlap = len(old_set & new_set) / len(old_set | new_set)
        if overlap < 0.3:
            continue

        # Skip duplicates
        key = (old_word.lower(), new_word.lower())
        if key in seen:
            continue
        seen.add(key)
        filtered.append((old_word, new_word))

    return filtered

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def split_by_file_markers(text):
    """Split concatenated files by FILE: markers"""
    sections = []
    current = []

    for line in text.splitlines():
        if line.strip().startswith("FILE:"):
            if current:
                sections.append("\n".join(current))
                current = []
        current.append(line)

    if current:
        sections.append("\n".join(current))

    return sections

def main():
    if not OLD_PATH.exists():
        raise FileNotFoundError(f"Missing {OLD_PATH}")
    if not NEW_PATH.exists():
        raise FileNotFoundError(f"Missing {NEW_PATH}")

    print("Loading texts...")
    old_text = OLD_PATH.read_text(encoding="utf-8", errors="ignore")
    new_text = NEW_PATH.read_text(encoding="utf-8", errors="ignore")

    # Split by file sections
    print("Splitting into file sections...")
    old_sections = split_by_file_markers(old_text)
    new_sections = split_by_file_markers(new_text)
    print(f"  Old sections: {len(old_sections)}")
    print(f"  New sections: {len(new_sections)}")

    all_aligned = []

    # Align each section separately to avoid cross-file contamination
    num_sections = min(len(old_sections), len(new_sections))
    for idx in range(num_sections):
        print(f"\nProcessing section {idx+1}/{num_sections}...")
        old_phrases = extract_phrases(old_sections[idx], merge_blocks=True)
        new_phrases = extract_phrases(new_sections[idx], merge_blocks=True)

        if not old_phrases or not new_phrases:
            continue

        print(f"  Old phrases: {len(old_phrases)}, New phrases: {len(new_phrases)}")
        aligned = align_with_local_repair(old_phrases, new_phrases)
        all_aligned.extend(aligned)
        print(f"  Found {len(aligned)} word pairs in this section")

    print(f"\nWriting {len(all_aligned)} word pairs to CSV...")
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old", "modern"])
        for o, n in all_aligned:
            w.writerow([o, n])

    print(f"\n{'='*60}")
    print("ALIGNMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total unique word pairs: {len(all_aligned)}")
    print(f"Output file: {OUT_PATH}")
    print(f"\nSample pairs:")
    for i, (o, n) in enumerate(all_aligned[:15]):
        print(f"  {o:20} → {n}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
