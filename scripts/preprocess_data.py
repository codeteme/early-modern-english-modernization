"""Preprocess real Shakespeare data."""

import sys
import os

sys.path.append(".")

from scripts.utils import preprocess_text


def main():
    """
    Preprocess raw parallel Shakespeare data.

    Expected input format in data/raw/shakespeare_parallel.txt:
    EARLY: [early modern text]
    MODERN: [modern text]
    EARLY: [early modern text]
    MODERN: [modern text]
    ...
    """

    input_file = "data/raw/shakespeare_parallel.txt"

    if not os.path.exists(input_file):
        print(f"Please create {input_file} with parallel text.")
        print("Format:")
        print("EARLY: But soft what light through yonder window breaks")
        print("MODERN: But wait what's that light in the window over there")
        print("...")
        return

    early_lines = []
    modern_lines = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("EARLY:"):
                text = line.replace("EARLY:", "").strip()
                early_lines.append(preprocess_text(text))
            elif line.startswith("MODERN:"):
                text = line.replace("MODERN:", "").strip()
                modern_lines.append(preprocess_text(text))

    assert len(early_lines) == len(
        modern_lines
    ), "Mismatch in number of early and modern lines"

    # Split: 80% train, 10% dev, 10% test
    n = len(early_lines)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)

    os.makedirs("data/processed", exist_ok=True)

    # Train
    with open("data/processed/train_early.txt", "w") as f:
        f.write("\n".join(early_lines[:n_train]))
    with open("data/processed/train_modern.txt", "w") as f:
        f.write("\n".join(modern_lines[:n_train]))

    # Dev
    with open("data/processed/dev_early.txt", "w") as f:
        f.write("\n".join(early_lines[n_train : n_train + n_dev]))
    with open("data/processed/dev_modern.txt", "w") as f:
        f.write("\n".join(modern_lines[n_train : n_train + n_dev]))

    # Test
    with open("data/processed/test_early.txt", "w") as f:
        f.write("\n".join(early_lines[n_train + n_dev :]))
    with open("data/processed/test_modern.txt", "w") as f:
        f.write("\n".join(modern_lines[n_train + n_dev :]))

    print(f"Processed {n} parallel sentences:")
    print(f"  Train: {n_train}")
    print(f"  Dev: {n_dev}")
    print(f"  Test: {n - n_train - n_dev}")


if __name__ == "__main__":
    main()
