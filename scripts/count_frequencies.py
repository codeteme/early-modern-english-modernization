"""Count word frequencies from actual corpus for aligned pairs."""

import csv
from collections import Counter
import re
import os


def count_word_frequencies(
    csv_path="scripts/aligned_old_to_modern.csv",
    old_corpus_path="data/old_merged.txt",
    new_corpus_path="data/new_merged.txt",
    output_path="weights/word_frequencies_from_corpus.csv",
):
    """Count how many times each word appears in actual corpus."""

    print("Loading corpus files...")
    with open(old_corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        old_text = f.read().lower()

    with open(new_corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        new_text = f.read().lower()

    print("Extracting words from corpora...")
    old_words = re.findall(r"\b\w+\b", old_text)
    new_words = re.findall(r"\b\w+\b", new_text)

    old_freq = Counter(old_words)
    new_freq = Counter(new_words)

    total_old = len(old_words)
    total_new = len(new_words)

    print(f"  Old corpus: {total_old:,} total words, {len(old_freq):,} unique")
    print(f"  New corpus: {total_new:,} total words, {len(new_freq):,} unique")

    print(f"\nLoading aligned pairs from {csv_path}...")
    aligned_pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_word = row["old"].strip().lower()
            modern_word = row["modern"].strip().lower()
            aligned_pairs.append((old_word, modern_word))

    print(f"  Loaded {len(aligned_pairs)} aligned pairs")

    print("\nCounting frequencies...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "old_word",
                "modern_word",
                "old_count",
                "old_probability",
                "modern_count",
                "modern_probability",
                "appears_in_old_corpus",
                "appears_in_modern_corpus",
            ]
        )

        for old_word, modern_word in aligned_pairs:
            old_count = old_freq.get(old_word, 0)
            new_count = new_freq.get(modern_word, 0)

            old_prob = (old_count / total_old) * 100 if total_old > 0 else 0
            new_prob = (new_count / total_new) * 100 if total_new > 0 else 0

            appears_old = "YES" if old_count > 0 else "NO"
            appears_new = "YES" if new_count > 0 else "NO"

            writer.writerow(
                [
                    old_word,
                    modern_word,
                    old_count,
                    f"{old_prob:.6f}",
                    new_count,
                    f"{new_prob:.6f}",
                    appears_old,
                    appears_new,
                ]
            )

    old_words_found = sum(1 for old, _ in aligned_pairs if old_freq.get(old, 0) > 0)
    modern_words_found = sum(1 for _, mod in aligned_pairs if new_freq.get(mod, 0) > 0)
    both_found = sum(
        1
        for old, mod in aligned_pairs
        if old_freq.get(old, 0) > 0 and new_freq.get(mod, 0) > 0
    )

    print(f"\nStatistics:")
    print(f"  Total aligned pairs: {len(aligned_pairs)}")
    print(
        f"  Old words found: {old_words_found} ({old_words_found/len(aligned_pairs)*100:.1f}%)"
    )
    print(
        f"  Modern words found: {modern_words_found} ({modern_words_found/len(aligned_pairs)*100:.1f}%)"
    )
    print(f"  Both found: {both_found} ({both_found/len(aligned_pairs)*100:.1f}%)")
    print(f"\n✓ Saved to {output_path}")


if __name__ == "__main__":
    count_word_frequencies()
