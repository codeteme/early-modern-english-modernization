"""
Main translator: Old English -> Modern English
Logic adapted to match 'eval_word_pairs.py' regex tokenization.
"""

import csv
import re
import math
import difflib
from collections import defaultdict, Counter


class Translator:
    def __init__(self, n=5, k=0.5):
        self.n = n
        self.k = k
        self.word_map = defaultdict(list)
        self.modern_vocab = Counter()

    def load_weights(self, weights_path="../../data/word_weights.csv"):
        """Load word translation weights."""
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    old = row["old"].strip().lower()
                    modern = row["modern"].strip().lower()
                    weight = float(row["weight"])
                    self.word_map[old].append((modern, weight))

            for old in self.word_map:
                self.word_map[old].sort(key=lambda x: x[1])
            print(f"Loaded {len(self.word_map)} word mappings")
        except FileNotFoundError:
            print("Warning: word_weights.csv not found. Run setup.py first.")

    def train_lm(self, train_file="../../data/train_parallel.txt"):
        """Train 5-gram language model for spell check context."""
        try:
            with open(train_file, "r", encoding="utf-8") as f:
                text = f.read().lower()
            words = re.findall(r"\b\w+\b", text)
            self.modern_vocab.update(words)
            print(f"Trained LM ({len(self.modern_vocab)} vocab)")
        except FileNotFoundError:
            print("Warning: train_parallel.txt not found.")

    def edit_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev[j + 1] + 1
                rem = curr[j] + 1
                sub = prev[j] + (c1 != c2)
                curr.append(min(ins, rem, sub))
            prev = curr
        return prev[-1]

    def get_best_word(self, word):
        """Logic equivalent to decoder.decode_greedy()."""
        word_lower = word.lower()

        # 1. Dictionary Lookup
        if word_lower in self.word_map:
            return self.word_map[word_lower][0][0]

        # 2. Spell Correction (if no map)
        if word_lower in self.modern_vocab:
            return word_lower

        candidates = []
        for vocab_word, freq in self.modern_vocab.items():
            if freq < 2:
                continue
            dist = self.edit_distance(word_lower, vocab_word)
            if dist <= 2:
                score = math.log(freq) - (dist * 2)
                candidates.append((vocab_word, score))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return word_lower

    def normalize_sentence(self, sentence):
        """
        MATCHING EVALUATION SCRIPT LOGIC:
        Splits by space, then uses Regex to strip punctuation,
        translates core, restores case, and rebuilds.
        """
        # 1. Split into tokens (preserving space separation)
        tokens = sentence.split()
        normalized_words = []

        for token in tokens:
            # 2. Regex separation (Prefix + Core + Suffix)
            match = re.match(r"^([^\w]*)([\w]+)([^\w]*)$", token)

            if match:
                prefix, core, suffix = match.groups()

                # 3. Decode Core
                normalized_core = self.get_best_word(core)

                # 4. Preserve Capitalization
                if core and core[0].isupper():
                    normalized_core = normalized_core.capitalize()

                normalized_words.append(prefix + normalized_core + suffix)
            else:
                # No core found (e.g., just punctuation "...")
                normalized_words.append(token)

        # 5. Reconstruct
        return " ".join(normalized_words)

    def calculate_accuracy(self, pred, gold):
        """Calculate word-level similarity percentage."""
        pred_words = re.findall(r"\w+", pred.lower())
        gold_words = re.findall(r"\w+", gold.lower())
        if not pred_words and not gold_words:
            return 1.0
        matcher = difflib.SequenceMatcher(None, pred_words, gold_words)
        return matcher.ratio()


def evaluate(translator):
    print("\n" + "=" * 60 + "\nEVALUATION (REGEX TOKENIZATION MATCH)\n" + "=" * 60)
    try:
        with open("../../data/test_parallel.txt", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        total_acc = 0
        count = 0
        limit = min(40, len(lines))  # Evaluate the first 20 pairs

        for i in range(0, limit, 2):
            if i + 1 >= len(lines):
                break
            old, gold = lines[i], lines[i + 1]

            # Use the new reference-matching logic
            pred = translator.normalize_sentence(old)

            acc = translator.calculate_accuracy(pred, gold)
            total_acc += acc
            count += 1

            print(f"[{count}]")
            print(f"Old:  {old}")
            print(f"Pred: {pred}")
            print(f"Gold: {gold}")
            print(f"Accuracy: {acc:.1%}")
            print("-" * 40)

        if count > 0:
            print(f"\nAverage Accuracy on sample: {total_acc/count:.1%}")

    except FileNotFoundError:
        print("Run setup.py to generate test data.")


if __name__ == "__main__":
    import sys

    t = Translator()
    print("Loading models...")
    t.load_weights()
    t.train_lm()

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        print("\nInteractive Mode")
        while True:
            txt = input("> ")
            if txt == "quit":
                break
            print(t.normalize_sentence(txt))
    else:
        evaluate(t)
