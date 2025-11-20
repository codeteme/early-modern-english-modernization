"""Channel model with learned edit costs."""

from collections import defaultdict, Counter
import Levenshtein as lev


class ChannelModel:
    """Character-level channel model with learned edit costs."""

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: Smoothing parameter for edit costs
        """
        self.alpha = alpha

        # Edit operation counts
        self.insert_counts = Counter()  # inserted_char -> count
        self.delete_counts = Counter()  # deleted_char -> count
        self.substitute_counts = Counter()  # (old_char, new_char) -> count

        self.total_inserts = 0
        self.total_deletes = 0
        self.total_substitutes = 0

        # For normalization
        self.char_vocab = set()

    def train(self, parallel_pairs):
        """
        Train on parallel (modern, early_modern) pairs.

        Args:
            parallel_pairs: List of (modern_text, early_modern_text) tuples
        """
        print("Training channel model...")

        for modern, early in parallel_pairs:
            self.char_vocab.update(modern)
            self.char_vocab.update(early)

            # Get edit operations
            ops = lev.editops(modern, early)

            for op, i, j in ops:
                if op == "insert":
                    char = early[j]
                    self.insert_counts[char] += 1
                    self.total_inserts += 1

                elif op == "delete":
                    char = modern[i]
                    self.delete_counts[char] += 1
                    self.total_deletes += 1

                elif op == "replace":
                    old_char = modern[i]
                    new_char = early[j]
                    self.substitute_counts[(old_char, new_char)] += 1
                    self.total_substitutes += 1

        print(
            f"Total edits - Inserts: {self.total_inserts}, "
            f"Deletes: {self.total_deletes}, Substitutes: {self.total_substitutes}"
        )

    def insert_cost(self, char):
        """Cost of inserting char (modern -> early)."""
        count = self.insert_counts[char]
        total = self.total_inserts
        vocab_size = len(self.char_vocab)

        # Add-alpha smoothing, convert to cost (negative log prob)
        prob = (count + self.alpha) / (total + self.alpha * vocab_size)
        return -math.log(prob) if prob > 0 else float("inf")

    def delete_cost(self, char):
        """Cost of deleting char (modern -> early)."""
        count = self.delete_counts[char]
        total = self.total_deletes
        vocab_size = len(self.char_vocab)

        prob = (count + self.alpha) / (total + self.alpha * vocab_size)
        return -math.log(prob) if prob > 0 else float("inf")

    def substitute_cost(self, old_char, new_char):
        """Cost of substituting old_char with new_char (modern -> early)."""
        if old_char == new_char:
            return 0.0  # No cost for identical

        count = self.substitute_counts[(old_char, new_char)]
        total = self.total_substitutes
        vocab_size = len(self.char_vocab) ** 2

        prob = (count + self.alpha) / (total + self.alpha * vocab_size)
        return -math.log(prob) if prob > 0 else float("inf")

    def edit_distance(self, s1, s2):
        """Compute weighted edit distance from s1 to s2."""
        m, n = len(s1), len(s2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        # Initialize
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + self.delete_cost(s1[i - 1])
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + self.insert_cost(s2[j - 1])

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    substitute = dp[i - 1][j - 1] + self.substitute_cost(
                        s1[i - 1], s2[j - 1]
                    )
                    delete = dp[i - 1][j] + self.delete_cost(s1[i - 1])
                    insert = dp[i][j - 1] + self.insert_cost(s2[j - 1])
                    dp[i][j] = min(substitute, delete, insert)

        return dp[m][n]

    def channel_prob(self, modern, early):
        """P(early | modern) based on edit distance."""
        distance = self.edit_distance(modern, early)
        return math.exp(-distance)


import math
