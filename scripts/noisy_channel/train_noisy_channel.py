"""Train noisy channel model."""

import sys
import pickle

sys.path.append(".")

from scripts.language_model import CharNgramLM
from scripts.channel_model import ChannelModel
from scripts.noisy_channel import NoisyChannelDecoder
from scripts.utils import load_parallel_data


def train_on_dataset(data_type="synthetic"):
    """
    Train models on specified dataset.

    Args:
        data_type: 'synthetic' or 'processed' (real data)
    """
    print(f"\n{'='*60}")
    print(f"Training on {data_type} data")
    print("=" * 60)

    # Load training data
    train_pairs = load_parallel_data(
        f"data/{data_type}/train_early.txt", f"data/{data_type}/train_modern.txt"
    )

    print(f"Loaded {len(train_pairs)} training pairs\n")

    # Train language model
    print("1. Training language model...")
    modern_texts = [modern for modern, _ in train_pairs]
    lm = CharNgramLM(n=5, k=0.1)
    lm.train(modern_texts)

    # Compute perplexity on training data
    train_ppl = lm.perplexity(modern_texts[:100])  # Sample for speed
    print(f"Training perplexity: {train_ppl:.2f}\n")

    # Train channel model
    print("2. Training channel model...")
    channel = ChannelModel(alpha=1.0)
    channel.train(train_pairs)
    print()

    # Create decoder
    print("3. Creating noisy channel decoder...")
    decoder = NoisyChannelDecoder(lm, channel, lm_weight=1.0)
    print("Done!\n")

    # Save models
    import os

    os.makedirs("models", exist_ok=True)

    model_path = f"models/{data_type}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"lm": lm, "channel": channel, "decoder": decoder}, f)

    print(f"Models saved to {model_path}")

    return lm, channel, decoder


def main():
    """Train on both synthetic and real data."""

    # Train on synthetic data (Step 3a)
    print("\n" + "=" * 60)
    print("STEP 3a: Training on SYNTHETIC data")
    print("=" * 60)
    train_on_dataset("synthetic")

    # Train on real data (Step 3b)
    print("\n" + "=" * 60)
    print("STEP 3b: Training on REAL data")
    print("=" * 60)

    import os

    if os.path.exists("data/processed/train_early.txt"):
        train_on_dataset("processed")
    else:
        print("Real data not found. Run preprocess_data.py first.")
        print("Skipping real data training for now.")


if __name__ == "__main__":
    main()
