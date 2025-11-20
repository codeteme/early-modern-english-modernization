"""Generate synthetic parallel data."""

import random
import os


def generate_synthetic_pair():
    """Generate one synthetic (modern, early_modern) pair."""

    # Modern English templates
    templates = [
        "you are {adj}",
        "you have {noun}",
        "you go to {place}",
        "do you see {noun}",
        "what do you want",
        "where do you live",
        "you should {verb}",
        "i think you {verb}",
    ]

    # Vocabulary
    adjectives = ["wise", "kind", "brave", "strong", "clever"]
    nouns = ["honor", "courage", "wisdom", "strength", "grace"]
    places = ["london", "the castle", "the forest", "home", "court"]
    verbs = ["know", "understand", "believe", "trust", "hope"]

    # Generate modern sentence
    template = random.choice(templates)
    modern = template.format(
        adj=random.choice(adjectives),
        noun=random.choice(nouns),
        place=random.choice(places),
        verb=random.choice(verbs),
    )

    # Apply inverse transformations for early modern
    early = modern
    early = early.replace("you are", "thou art")
    early = early.replace("you have", "thou hast")
    early = early.replace("you go", "thou goest")
    early = early.replace("do you", "dost thou")
    early = early.replace("you should", "thou shouldst")
    early = early.replace("you ", "thou ")

    return modern, early


def main():
    """Generate synthetic dataset."""
    os.makedirs("data/synthetic", exist_ok=True)

    n_train = 800
    n_dev = 100
    n_test = 100

    print(f"Generating {n_train} training pairs...")
    train_pairs = [generate_synthetic_pair() for _ in range(n_train)]

    print(f"Generating {n_dev} dev pairs...")
    dev_pairs = [generate_synthetic_pair() for _ in range(n_dev)]

    print(f"Generating {n_test} test pairs...")
    test_pairs = [generate_synthetic_pair() for _ in range(n_test)]

    # Write train
    with open("data/synthetic/train_modern.txt", "w") as f:
        f.write("\n".join(p[0] for p in train_pairs))
    with open("data/synthetic/train_early.txt", "w") as f:
        f.write("\n".join(p[1] for p in train_pairs))

    # Write dev
    with open("data/synthetic/dev_modern.txt", "w") as f:
        f.write("\n".join(p[0] for p in dev_pairs))
    with open("data/synthetic/dev_early.txt", "w") as f:
        f.write("\n".join(p[1] for p in dev_pairs))

    # Write test
    with open("data/synthetic/test_modern.txt", "w") as f:
        f.write("\n".join(p[0] for p in test_pairs))
    with open("data/synthetic/test_early.txt", "w") as f:
        f.write("\n".join(p[1] for p in test_pairs))

    print("Synthetic data generated successfully!")
    print("\nExample pairs:")
    for i in range(5):
        print(f"Modern: {train_pairs[i][0]}")
        print(f"Early:  {train_pairs[i][1]}")
        print()


if __name__ == "__main__":
    main()
