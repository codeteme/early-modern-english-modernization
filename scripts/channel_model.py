"""Enhanced channel model with word-level and character-level features."""

from collections import defaultdict, Counter
import Levenshtein as lev
import math
import re


class ChannelModel:
    """Enhanced channel model with multi-level edit costs."""

    def __init__(self, alpha=1.0, use_word_features=True):
        """
        Args:
            alpha: Smoothing parameter for edit costs
            use_word_features: Whether to learn word-level patterns
        """
        self.alpha = alpha
        self.use_word_features = use_word_features

        # Character-level edit operation counts
        self.insert_counts = Counter()
        self.delete_counts = Counter()
        self.substitute_counts = Counter()

        self.total_inserts = 0
        self.total_deletes = 0
        self.total_substitutes = 0

        # Word-level patterns (NEW!)
        self.word_mappings = Counter()  # (early_word, modern_word) → count

        # Context-sensitive edits (NEW!)
        # Track what comes before/after an edit
        self.substitute_context = defaultdict(
            Counter
        )  # (left, right) → {(old, new): count}

        self.char_vocab = set()

    def train(self, parallel_pairs):
        """
        Train on parallel (modern, early_modern) pairs.

        Args:
            parallel_pairs: List of (modern_text, early_modern_text) tuples
        """
        print("Training enhanced channel model...")

        for modern, early in parallel_pairs:
            self.char_vocab.update(modern)
            self.char_vocab.update(early)

            # Learn word-level mappings
            if self.use_word_features:
                self._learn_word_mappings(modern, early)

            # Get character-level edit operations
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

                    # Learn context (character before and after)
                    left = modern[i - 1] if i > 0 else "^"
                    right = modern[i + 1] if i < len(modern) - 1 else "$"
                    self.substitute_context[(left, right)][(old_char, new_char)] += 1

        print(f"Learned character edits:")
        print(f"  Inserts: {self.total_inserts}")
        print(f"  Deletes: {self.total_deletes}")
        print(f"  Substitutes: {self.total_substitutes}")

        if self.use_word_features:
            print(f"  Word mappings: {len(self.word_mappings)}")
            print(f"\nTop word mappings:")
            for (early, modern), count in self.word_mappings.most_common(10):
                print(f"    {early} → {modern}: {count}x")

    def _learn_word_mappings(self, modern, early):
        """Extract word-level mappings from aligned sentences."""
        # Tokenize
        modern_words = re.findall(r"\b\w+\b", modern)
        early_words = re.findall(r"\b\w+\b", early)

        # Simple alignment: if words at same position differ, record mapping
        for i, (m_word, e_word) in enumerate(zip(modern_words, early_words)):
            if m_word != e_word:
                self.word_mappings[(e_word, m_word)] += 1

    def word_replacement_bonus(self, modern_word, early_word):
        """
        Give bonus if this word replacement is common.

        Returns: negative cost (bonus) if mapping is frequent
        """
        if not self.use_word_features:
            return 0.0

        count = self.word_mappings.get((early_word, modern_word), 0)

        if count > 0:
            # Bonus proportional to frequency
            bonus = math.log(1 + count)
            return -bonus  # Negative cost = bonus

        return 0.0

    def insert_cost(self, char):
        """Cost of inserting char (modern -> early)."""
        count = self.insert_counts[char]
        total = self.total_inserts
        vocab_size = len(self.char_vocab)

        if total == 0:
            return 1.0  # Default cost if no training data

        prob = (count + self.alpha) / (total + self.alpha * vocab_size)
        return -math.log(prob) if prob > 0 else float("inf")

    def delete_cost(self, char):
        """Cost of deleting char (modern -> early)."""
        count = self.delete_counts[char]
        total = self.total_deletes
        vocab_size = len(self.char_vocab)

        if total == 0:
            return 1.0

        prob = (count + self.alpha) / (total + self.alpha * vocab_size)
        return -math.log(prob) if prob > 0 else float("inf")

    def substitute_cost(self, old_char, new_char, context=None):
        """
        Cost of substituting old_char with new_char.

        Args:
            old_char: character in modern text
            new_char: character in early modern text
            context: optional tuple (left_char, right_char) for context
        """
        if old_char == new_char:
            return 0.0

        # Try context-sensitive cost first
        if context and context in self.substitute_context:
            context_count = self.substitute_context[context][(old_char, new_char)]
            context_total = sum(self.substitute_context[context].values())

            if context_total > 0:
                prob = (context_count + self.alpha) / (
                    context_total + self.alpha * len(self.char_vocab)
                )
                return -math.log(prob)

        # Fall back to global substitution cost
        count = self.substitute_counts[(old_char, new_char)]
        total = self.total_substitutes
        vocab_size = len(self.char_vocab) ** 2

        if total == 0:
            return 1.0

        prob = (count + self.alpha) / (total + self.alpha * vocab_size)
        return -math.log(prob) if prob > 0 else float("inf")

    def edit_distance_with_word_bonus(self, modern, early):
        """
        Compute weighted edit distance with word-level bonuses.

        This is the key improvement: combines character and word features.
        """
        # First compute character-level edit distance
        char_distance = self.edit_distance(modern, early)

        # Add word-level bonuses
        modern_words = re.findall(r"\b\w+\b", modern)
        early_words = re.findall(r"\b\w+\b", early)

        word_bonus = 0.0
        for m_word, e_word in zip(modern_words, early_words):
            if m_word != e_word:
                word_bonus += self.word_replacement_bonus(m_word, e_word)

        total_distance = char_distance + word_bonus
        return max(0.0, total_distance)  # Don't go negative

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
                    # Context for substitution
                    left = s1[i - 2] if i > 1 else "^"
                    right = s1[i] if i < m else "$"
                    context = (left, right)

                    substitute = dp[i - 1][j - 1] + self.substitute_cost(
                        s1[i - 1], s2[j - 1], context
                    )
                    delete = dp[i - 1][j] + self.delete_cost(s1[i - 1])
                    insert = dp[i][j - 1] + self.insert_cost(s2[j - 1])
                    dp[i][j] = min(substitute, delete, insert)

        return dp[m][n]

    def channel_prob(self, modern, early):
        """P(early | modern) based on edit distance."""
        distance = self.edit_distance_with_word_bonus(modern, early)
        return math.exp(-distance)
