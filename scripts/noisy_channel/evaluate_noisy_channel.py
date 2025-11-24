"""Evaluate trained models."""

import sys
import pickle

sys.path.append(".")

from scripts.utils import load_parallel_data, compute_metrics


def evaluate_model(data_type="synthetic", model_type="synthetic"):
    """
    Evaluate model on test set.

    Args:
        data_type: Dataset to evaluate on ('synthetic' or 'processed')
        model_type: Model to use ('synthetic' or 'processed')
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type} model on {data_type} test set")
    print("=" * 60)

    # Load model
    model_path = f"models/{model_type}_model.pkl"
    with open(model_path, "rb") as f:
        models = pickle.load(f)

    decoder = models["decoder"]

    # Load test data
    test_pairs = load_parallel_data(
        f"data/{data_type}/test_early.txt", f"data/{data_type}/test_modern.txt"
    )

    print(f"Loaded {len(test_pairs)} test pairs\n")

    # Generate predictions
    predictions = []
    references = []

    print("Generating predictions...")
    for modern_ref, early in test_pairs:
        # Generate candidates
        candidates = decoder.generate_candidates(early)

        # Decode
        prediction, score = decoder.decode_greedy(early, candidates)

        predictions.append(prediction)
        references.append(modern_ref)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, references)

    print(f"\nResults:")
    print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
    print(f"  Character Error Rate: {metrics['character_error_rate']:.4f}")
    print(f"  Exact Matches: {metrics['exact_matches']} / {metrics['total']}")

    # Show examples
    print(f"\n{'='*60}")
    print("Example Predictions:")
    print("=" * 60)

    for i in range(min(10, len(test_pairs))):
        modern_ref, early = test_pairs[i]
        pred = predictions[i]

        match = "✓" if pred == modern_ref else "✗"

        print(f"\n{match} Example {i+1}:")
        print(f"  Early:      {early}")
        print(f"  Predicted:  {pred}")
        print(f"  Reference:  {modern_ref}")

    # Save results
    import os

    os.makedirs("results", exist_ok=True)

    result_file = f"results/{model_type}_on_{data_type}.txt"
    with open(result_file, "w") as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Test Set: {data_type}\n")
        f.write(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}\n")
        f.write(f"Character Error Rate: {metrics['character_error_rate']:.4f}\n")
        f.write(f"\nExamples:\n")
        f.write("=" * 60 + "\n")

        for i in range(len(test_pairs)):
            modern_ref, early = test_pairs[i]
            pred = predictions[i]
            match = "✓" if pred == modern_ref else "✗"

            f.write(f"\n{match} Example {i+1}:\n")
            f.write(f"  Early:      {early}\n")
            f.write(f"  Predicted:  {pred}\n")
            f.write(f"  Reference:  {modern_ref}\n")

    print(f"\nResults saved to {result_file}")

    return metrics


def main():
    """Run all evaluations."""
    import os

    results = {}

    # Evaluate synthetic model on synthetic data
    if os.path.exists("models/synthetic_model.pkl"):
        results["synthetic_on_synthetic"] = evaluate_model("synthetic", "synthetic")

    # Evaluate real model on real data
    if os.path.exists("models/processed_model.pkl") and os.path.exists(
        "data/processed/test_early.txt"
    ):
        results["processed_on_processed"] = evaluate_model("processed", "processed")

    # Cross-evaluation: synthetic model on real data
    if os.path.exists("models/synthetic_model.pkl") and os.path.exists(
        "data/processed/test_early.txt"
    ):
        results["synthetic_on_processed"] = evaluate_model("processed", "synthetic")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL RESULTS")
    print("=" * 60)
    print(f"{'Model':<20} {'Test Set':<20} {'Exact Match':<15} {'CER':<10}")
    print("-" * 60)

    for key, metrics in results.items():
        model, test = key.split("_on_")
        print(
            f"{model:<20} {test:<20} "
            f"{metrics['exact_match_accuracy']:>14.2%} "
            f"{metrics['character_error_rate']:>9.4f}"
        )


if __name__ == "__main__":
    main()
