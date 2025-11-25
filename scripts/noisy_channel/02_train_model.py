#!/usr/bin/env python3
"""Train noisy channel model from extracted word pairs."""

import sys
import pickle
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from scripts.noisy_channel.language_model import CharNgramLM
from scripts.noisy_channel.channel_model import ChannelModel
from scripts.noisy_channel.decoder import NoisyChannelDecoder


def load_word_pairs(csv_path):
    """Load word pairs from CSV file.

    Args:
        csv_path: Path to CSV file with 'old' and 'modern' columns

    Returns:
        List of (modern, old) tuples for training
    """
    print(f"Loading word pairs from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Convert to list of (modern, early) pairs
    pairs = list(zip(df["modern"].values, df["old"].values))

    print(f"Loaded {len(pairs)} word pairs")
    print(f"\nSample pairs:")
    for i, (modern, old) in enumerate(pairs[:10], 1):
        print(f"  {i}. {old:15s} → {modern}")

    return pairs


def train_from_word_pairs(csv_path, output_path=None):
    """
    Train noisy channel model from word pairs CSV.

    Args:
        csv_path: Path to CSV with word pairs
        output_path: Path to save trained model (default: models/word_pairs_model.pkl)
    """
    if output_path is None:
        output_path = Path(__file__).parents[2] / "models" / "word_pairs_model.pkl"

    print("=" * 70)
    print("Training Noisy Channel Model from Word Pairs")
    print("=" * 70)
    print()

    # Load word pairs
    pairs = load_word_pairs(csv_path)

    # Extract modern and early texts
    modern_texts = [modern for modern, _ in pairs]
    early_texts = [early for _, early in pairs]

    print(f"\n{'=' * 70}")
    print("Step 1: Training Language Model")
    print("=" * 70)

    # Train language model on modern texts
    lm = CharNgramLM(n=5, k=0.1)
    lm.train(modern_texts)

    # Compute perplexity
    sample_size = min(100, len(modern_texts))
    train_ppl = lm.perplexity(modern_texts[:sample_size])
    print(f"Training perplexity: {train_ppl:.2f}")

    print(f"\n{'=' * 70}")
    print("Step 2: Training Channel Model")
    print("=" * 70)

    # Train channel model
    channel = ChannelModel(alpha=1.0, use_word_features=True)
    channel.train(pairs)

    print(f"\n{'=' * 70}")
    print("Step 3: Creating Decoder")
    print("=" * 70)

    # Create decoder
    decoder = NoisyChannelDecoder(lm, channel, lm_weight=0.3)

    print(f"\n{'=' * 70}")
    print("Step 4: Saving Models")
    print("=" * 70)

    # Save models
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump({"lm": lm, "channel": channel, "decoder": decoder}, f)

    print(f"✓ Models saved to: {output_path}")

    print(f"\n{'=' * 70}")
    print("Step 5: Testing on Sample Data")
    print("=" * 70)

    # Test on a few examples
    test_samples = early_texts[:5]
    print("\nSample translations:")
    for i, early_word in enumerate(test_samples, 1):
        modern_word, score = decoder.decode_greedy(early_word)
        print(f"  {i}. {early_word:15s} → {modern_word:15s} (score: {score:.2f})")

    print("\n✓ Training complete!")
    return lm, channel, decoder


def main():
    """Train model from word pairs."""
    # Path to the combined word pairs CSV
    project_root = Path(__file__).parents[2]
    csv_path = (
        project_root / "data" / "processed" / "aligned_old_to_modern_combined.csv"
    )

    if not csv_path.exists():
        print(f"Error: Word pairs file not found at {csv_path}")
        print("Please run extract_word_pairs.py first")
        sys.exit(1)

    # Train the model
    train_from_word_pairs(csv_path)


if __name__ == "__main__":
    main()
