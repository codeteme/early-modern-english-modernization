"""Preprocess real Shakespeare data with improved alignment."""

import sys
import os
import re

sys.path.append(".")

from scripts.utils import preprocess_text


def parse_parallel_file(input_file):
    """
    Parse file with alternating lines.

    Expected format:
    In delivering my son from me, I bury a second husband.
    In saying goodbye to my son, it's like I'm losing another husband.
    <blank line or next pair>
    """
    early_lines = []
    modern_lines = []

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    # Process pairs
    i = 0
    while i < len(lines) - 1:
        early = lines[i]
        modern = lines[i + 1]

        # Skip if either line is too short
        if len(early) < 5 or len(modern) < 5:
            i += 2
            continue

        early_lines.append(preprocess_text(early))
        modern_lines.append(preprocess_text(modern))
        i += 2

    return early_lines, modern_lines


def split_into_phrases(early, modern):
    """
    Split long sentences into aligned phrases for better learning.

    Example:
    Early: "In delivering my son from me, I bury a second husband."
    Modern: "In saying goodbye to my son, it's like I'm losing another husband."

    Splits into phrases at punctuation.
    """
    # Split on punctuation
    early_phrases = re.split(r"[,;]", early)
    modern_phrases = re.split(r"[,;]", modern)

    # If same number of phrases, return pairs
    if len(early_phrases) == len(modern_phrases):
        pairs = []
        for e, m in zip(early_phrases, modern_phrases):
            e = e.strip()
            m = m.strip()
            if e and m:
                pairs.append((e, m))
        return pairs

    # Otherwise return whole sentence
    return [(early, modern)]


def main():
    """Preprocess Shakespeare parallel data."""

    input_file = "data/raw/shakespeare_parallel.txt"

    if not os.path.exists(input_file):
        print(f"Please create {input_file} with parallel text.")
        print("\nExpected format (alternating lines):")
        print("In delivering my son from me, I bury a second husband.")
        print("In saying goodbye to my son, it's like I'm losing another husband.")
        print("<blank line or next pair>")
        print("...")
        return

    print("Parsing parallel file...")
    early_lines, modern_lines = parse_parallel_file(input_file)

    print(f"Found {len(early_lines)} parallel sentence pairs")

    # Optionally split long sentences into phrases
    all_early = []
    all_modern = []

    for early, modern in zip(early_lines, modern_lines):
        # Split into phrases if possible
        phrase_pairs = split_into_phrases(early, modern)

        for e, m in phrase_pairs:
            all_early.append(e)
            all_modern.append(m)

    print(f"After phrase splitting: {len(all_early)} training examples")

    # Add word-level pairs for better learning
    # Extract common word mappings
    word_pairs = extract_word_pairs(early_lines, modern_lines)

    for early_word, modern_word in word_pairs:
        all_early.append(early_word)
        all_modern.append(modern_word)

    print(f"After adding word pairs: {len(all_early)} training examples")

    # Split: 80% train, 10% dev, 10% test
    n = len(all_early)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)

    # Shuffle to mix sentence and word examples
    import random

    combined = list(zip(all_early, all_modern))
    random.seed(42)
    random.shuffle(combined)
    all_early, all_modern = zip(*combined)

    os.makedirs("data/processed", exist_ok=True)

    # Train
    with open("data/processed/train_early.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_early[:n_train]))
    with open("data/processed/train_modern.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_modern[:n_train]))

    # Dev
    with open("data/processed/dev_early.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_early[n_train : n_train + n_dev]))
    with open("data/processed/dev_modern.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_modern[n_train : n_train + n_dev]))

    # Test
    with open("data/processed/test_early.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_early[n_train + n_dev :]))
    with open("data/processed/test_modern.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_modern[n_train + n_dev :]))

    print(f"\nProcessed {n} total examples:")
    print(f"  Train: {n_train}")
    print(f"  Dev: {n_dev}")
    print(f"  Test: {n - n_train - n_dev}")

    # Show some examples
    print("\nExample training pairs:")
    for i in range(min(5, n_train)):
        print(f"\nEarly:  {all_early[i]}")
        print(f"Modern: {all_modern[i]}")


def extract_word_pairs(early_lines, modern_lines):
    """
    Extract common word-level mappings from sentence pairs.

    This helps the model learn word-level transformations.
    """
    from collections import Counter
    import re

    word_mappings = []

    # Common Early Modern words and their modern equivalents
    # We'll learn these from the data
    early_words = Counter()
    modern_words = Counter()

    for early, modern in zip(early_lines, modern_lines):
        early_tokens = re.findall(r"\b\w+\b", early.lower())
        modern_tokens = re.findall(r"\b\w+\b", modern.lower())

        early_words.update(early_tokens)
        modern_words.update(modern_tokens)

    # Find words that appear frequently in early but rarely in modern
    # (likely archaic words)
    archaic_words = []
    for word, count in early_words.items():
        if count >= 3 and modern_words[word] < count / 2:
            archaic_words.append(word)

    # For each archaic word, try to find its modern equivalent
    # by looking at sentence pairs where it appears
    word_pairs = []

    for archaic in archaic_words[:50]:  # Limit to top 50
        # Find sentences containing this word
        for early, modern in zip(early_lines, modern_lines):
            if archaic in early.lower():
                # Simple heuristic: look for similar-length word in modern
                early_tokens = re.findall(r"\b\w+\b", early.lower())
                modern_tokens = re.findall(r"\b\w+\b", modern.lower())

                if archaic in early_tokens:
                    idx = early_tokens.index(archaic)
                    # Try to find corresponding word in modern
                    if idx < len(modern_tokens):
                        modern_equiv = modern_tokens[idx]
                        if modern_equiv != archaic:
                            word_pairs.append((archaic, modern_equiv))
                            break  # Found one mapping, move to next archaic word

    print(f"\nExtracted {len(word_pairs)} word-level mappings:")
    for e, m in word_pairs[:10]:
        print(f"  {e} → {m}")

    return word_pairs


if __name__ == "__main__":
    main()
