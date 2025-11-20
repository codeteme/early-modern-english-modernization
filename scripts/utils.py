"""Utility functions."""

import re


def preprocess_text(text):
    """Clean and normalize text."""
    # Lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Keep only basic characters
    text = re.sub(r"[^a-z0-9\s\.,!?\'-]", "", text)

    return text.strip()


def load_parallel_data(early_file, modern_file):
    """Load parallel corpus from two files."""
    with open(early_file, "r", encoding="utf-8") as f:
        early_lines = [preprocess_text(line) for line in f]

    with open(modern_file, "r", encoding="utf-8") as f:
        modern_lines = [preprocess_text(line) for line in f]

    assert len(early_lines) == len(
        modern_lines
    ), "Early and modern files must have same number of lines"

    return list(zip(modern_lines, early_lines))


def compute_metrics(predictions, references):
    """Compute evaluation metrics."""
    from Levenshtein import distance

    exact_match = sum(p == r for p, r in zip(predictions, references))
    exact_match_acc = exact_match / len(predictions)

    # Character Error Rate
    total_cer = 0
    for pred, ref in zip(predictions, references):
        if len(ref) > 0:
            total_cer += distance(pred, ref) / len(ref)

    avg_cer = total_cer / len(predictions)

    return {
        "exact_match_accuracy": exact_match_acc,
        "character_error_rate": avg_cer,
        "exact_matches": exact_match,
        "total": len(predictions),
    }
