#!/usr/bin/env python3
"""
Evaluation script for Noisy Channel Model.
Tests on ../../data/test_parallel.txt with Accuracy % reporting.
"""

import sys
import pickle
import re
import difflib
from pathlib import Path

# Add project root to path so we can import scripts.noisy_channel...
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from scripts.noisy_channel.language_model import CharNgramLM
from scripts.noisy_channel.channel_model import ChannelModel
from scripts.noisy_channel.decoder import NoisyChannelDecoder


def load_model(model_path="models/word_pairs_model.pkl"):
    """Load the trained Noisy Channel model."""
    full_path = project_root / model_path
    print(f"Loading model from {full_path}...")

    try:
        with open(full_path, "rb") as f:
            models = pickle.load(f)
        return models["decoder"]
    except FileNotFoundError:
        print(f"Error: Model not found at {full_path}")
        print("Run 02_train_model.py first.")
        sys.exit(1)


def normalize_sentence(decoder, sentence):
    """
    Normalize a sentence word-by-word.
    1. Splits by whitespace.
    2. Separates punctuation (Prefix + Core + Suffix).
    3. Decodes the Core word.
    4. Reassembles.
    """
    tokens = sentence.split()
    normalized_tokens = []

    for token in tokens:
        # Regex to strip punctuation: (Prefix)(Word)(Suffix)
        # e.g. "(King)," -> Prefix="(", Core="King", Suffix="),"
        match = re.match(r"^([^\w]*)([\w\'-]+)([^\w]*)$", token)

        if match:
            prefix, core, suffix = match.groups()

            # Decode the core word
            # decode_greedy returns (best_word, score)
            normalized_core, _ = decoder.decode_greedy(core)

            # Reconstruct
            normalized_tokens.append(prefix + normalized_core + suffix)
        else:
            # If no word core found (e.g. just "..." or "??"), keep as is
            normalized_tokens.append(token)

    return " ".join(normalized_tokens)


def calculate_accuracy(pred, gold):
    """Calculate word-level similarity percentage."""
    # Split into words (ignore punctuation for scoring)
    pred_words = re.findall(r"\w+", pred.lower())
    gold_words = re.findall(r"\w+", gold.lower())

    if not pred_words and not gold_words:
        return 1.0

    matcher = difflib.SequenceMatcher(None, pred_words, gold_words)
    return matcher.ratio()


def evaluate_test_set(decoder):
    """Run evaluation on the test_parallel.txt file."""
    test_file = project_root / "data" / "test_parallel.txt"

    print(f"Evaluating on {test_file}")
    print("=" * 60)

    try:
        with open(test_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        total_acc = 0
        count = 0
        limit = min(40, len(lines))  # Show first 20 pairs (40 lines)

        for i in range(0, limit, 2):
            if i + 1 >= len(lines):
                break

            old = lines[i]
            gold = lines[i + 1]

            # Generate Prediction
            pred = normalize_sentence(decoder, old)

            # Metric
            acc = calculate_accuracy(pred, gold)
            total_acc += acc
            count += 1

            # Print output
            print(f"[{count}]")
            print(f"Old:  {old}")
            print(f"Pred: {pred}")
            print(f"Gold: {gold}")
            print(f"Accuracy: {acc:.1%}")
            print("-" * 40)

        if count > 0:
            print(f"\nAverage Accuracy on sample: {total_acc/count:.1%}")

    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file}")
        print("Run 01_build_dictionary.py (setup.py) first.")


def main():
    decoder = load_model()
    evaluate_test_set(decoder)


if __name__ == "__main__":
    main()
