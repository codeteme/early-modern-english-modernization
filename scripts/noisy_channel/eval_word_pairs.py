#!/usr/bin/env python3
"""
Evaluation script for Noisy Channel spelling normalization model

This script evaluates the word-level Noisy Channel model on Old English text normalization.

Usage:
    python scripts/noisy_channel/eval_word_pairs.py --input "I cannot conceiue you"
    python scripts/noisy_channel/eval_word_pairs.py --input-file test_sentences.txt
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

# Add project root to path for pickle imports
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))


# ============================================================================
# Model Loading
# ============================================================================

def load_noisy_channel_model(model_path='models/word_pairs_model.pkl'):
    """Load the word pairs noisy channel model"""
    print(f"Loading Noisy Channel model from {model_path}...")

    # Resolve path
    if not Path(model_path).is_absolute():
        model_path = project_root / model_path

    with open(model_path, 'rb') as f:
        models = pickle.load(f)

    decoder = models['decoder']
    print("Model loaded successfully!")

    return decoder


# ============================================================================
# Sentence Processing
# ============================================================================

def tokenize_sentence(sentence):
    """Split sentence into words while preserving punctuation and spacing"""
    words = []
    spaces = []

    tokens = sentence.split()
    for i, token in enumerate(tokens):
        words.append(token)
        if i < len(tokens) - 1:
            spaces.append(' ')

    return words, spaces


def normalize_sentence_word_by_word(sentence, decoder):
    """Normalize a sentence by processing each word separately"""
    words, spaces = tokenize_sentence(sentence)
    normalized_words = []

    for word in words:
        # Separate punctuation
        match = re.match(r'^([^\w]*)([\w]+)([^\w]*)$', word)
        if match:
            prefix, core, suffix = match.groups()
            # Normalize the core word
            normalized_core, _ = decoder.decode_greedy(core.lower())
            # Preserve capitalization
            if core and core[0].isupper():
                normalized_core = normalized_core.capitalize()
            normalized_word = prefix + normalized_core + suffix
        else:
            # No word core found, keep as is
            normalized_word = word

        normalized_words.append(normalized_word)

    # Reconstruct sentence
    result = []
    for i, word in enumerate(normalized_words):
        result.append(word)
        if i < len(spaces):
            result.append(spaces[i])

    return ''.join(result)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Noisy Channel spelling normalization on sentences'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/word_pairs_model.pkl',
        help='Path to model (default: models/word_pairs_model.pkl)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input sentence to normalize'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='File with sentences to normalize (one per line)'
    )

    args = parser.parse_args()

    # Load model
    decoder = load_noisy_channel_model(args.model_path)

    print("=" * 80)

    # Get input sentences
    if args.input:
        sentences = [args.input]
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            sentences = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("\nInteractive mode - Enter sentences to normalize (Ctrl+C to exit)")
        print("=" * 80)
        sentences = []
        try:
            while True:
                sentence = input("\nOld English: ").strip()
                if sentence:
                    sentences.append(sentence)

                    # Process immediately in interactive mode
                    normalized = normalize_sentence_word_by_word(sentence, decoder)

                    print(f"Modern:      {normalized}")
                    sentences = []  # Clear after processing
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return

    # Process batch mode
    if sentences:
        print("\nProcessing sentences...")
        print("=" * 80)

        for i, sentence in enumerate(sentences, 1):
            print(f"\n[{i}/{len(sentences)}]")
            print(f"Old:    {sentence}")

            normalized = normalize_sentence_word_by_word(sentence, decoder)

            print(f"Modern: {normalized}")

        print("\n" + "=" * 80)
        print("Processing complete!")


if __name__ == '__main__':
    main()
