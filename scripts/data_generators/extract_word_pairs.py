#!/usr/bin/env python3
"""
Extract word-level spelling pairs from parallel Shakespeare texts
Creates a larger training dataset for the LSTM model
"""

import os
import re
from pathlib import Path
from collections import Counter
import pandas as pd
from difflib import SequenceMatcher


def clean_text(text):
    """Clean and normalize text"""
    # Remove title/header lines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip metadata lines
        if (
            line.startswith("Title:")
            or line.startswith("Version:")
            or line.startswith("Source:")
            or line == "=" * 70
            or not line
        ):
            continue
        cleaned_lines.append(line)
    return " ".join(cleaned_lines)


def tokenize_words(text):
    """Tokenize text into words, preserving original forms"""
    # Split on whitespace and extract words with apostrophes, hyphens
    words = re.findall(r"[a-zA-Z]+(?:[''-][a-zA-Z]+)*", text)
    return words


def align_and_extract_pairs(old_text, new_text):
    """Extract word pairs from aligned parallel texts"""
    old_words = tokenize_words(old_text)
    new_words = tokenize_words(new_text)

    pairs = []

    # Use SequenceMatcher to align words
    matcher = SequenceMatcher(None, old_words, new_words)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1) == 1:
            # One-to-one word replacement (spelling difference)
            old_word = old_words[i1]
            new_word = new_words[j1]

            # Only keep if they're similar enough (likely spelling variant)
            # and not completely different words
            similarity = SequenceMatcher(
                None, old_word.lower(), new_word.lower()
            ).ratio()
            if similarity > 0.5 and old_word.lower() != new_word.lower():
                pairs.append((old_word, new_word))
        elif tag == "equal":
            # Identical words - add some for identity mapping
            for k in range(i1, min(i2, i1 + 2)):  # Limit to avoid too many
                word = old_words[k]
                if len(word) >= 3:  # Skip very short words
                    pairs.append((word, word))

    return pairs


def main():
    """Extract word pairs from all Shakespeare text pairs"""
    data_dir = Path("data/texts")

    all_pairs = []

    # Find all directories with parallel texts
    for old_file in data_dir.rglob("old_text.txt"):
        new_file = old_file.parent / "new_text.txt"

        if not new_file.exists():
            continue

        print(f"Processing {old_file.parent.name}...")

        # Read files
        with open(old_file, "r", encoding="utf-8") as f:
            old_text = f.read()

        with open(new_file, "r", encoding="utf-8") as f:
            new_text = f.read()

        # Clean texts
        old_text = clean_text(old_text)
        new_text = clean_text(new_text)

        # Extract pairs
        pairs = align_and_extract_pairs(old_text, new_text)
        all_pairs.extend(pairs)
        print(f"  Extracted {len(pairs)} word pairs")

    print(f"\nTotal pairs extracted: {len(all_pairs)}")

    # Deduplicate and count
    pair_counts = Counter(all_pairs)
    print(f"Unique pairs: {len(pair_counts)}")

    # Create DataFrame
    df = pd.DataFrame(
        [(old, new) for (old, new), count in pair_counts.items()],
        columns=["old", "modern"],
    )

    # Remove pairs where both are identical
    df_diff = df[df["old"].str.lower() != df["modern"].str.lower()].copy()

    # Keep some identity mappings for robustness
    df_same = df[df["old"].str.lower() == df["modern"].str.lower()].copy()
    df_same = df_same.sample(n=min(200, len(df_same)), random_state=42)

    # Combine
    df_final = pd.concat([df_diff, df_same]).drop_duplicates().reset_index(drop=True)

    print(f"Final dataset: {len(df_final)} pairs")
    print(f"  - Different spellings: {len(df_diff)}")
    print(f"  - Identity mappings: {len(df_same)}")

    # Save
    output_path = "../data/processed/aligned_old_to_modern_extracted.csv"
    df_final.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Show sample
    print("\nSample pairs:")
    print(df_final.head(20).to_string(index=False))

    # Combine with existing augmented data
    existing_df = pd.read_csv("scripts/aligned_old_to_modern_augmented.csv")
    print(f"\nExisting augmented data: {len(existing_df)} pairs")

    combined_df = (
        pd.concat([existing_df, df_final])
        .drop_duplicates(subset=["old", "modern"])
        .reset_index(drop=True)
    )
    print(f"Combined dataset: {len(combined_df)} pairs")

    output_combined = "data/processed/aligned_old_to_modern_combined.csv"
    combined_df.to_csv(output_combined, index=False)
    print(f"Saved combined dataset to {output_combined}")


if __name__ == "__main__":
    main()
