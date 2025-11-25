"""Character n-gram language model."""

from collections import defaultdict, Counter
import math


class CharNgramLM:
    """Character-level n-gram language model with add-k smoothing."""

    def __init__(self, n=5, k=0.1):
        self.n = n
        self.k = k
        self.counts = defaultdict(Counter)
        self.vocab = set()
        self.context_totals = defaultdict(int)
        # Fix: Track known words for the decoder
        self.known_words = set()

    def train(self, texts):
        """Train on list of text strings."""
        print(f"Training {self.n}-gram language model...")

        self.known_words.update(texts)

        for text in texts:
            # Add start/end markers
            text = "^" * (self.n - 1) + text + "$"

            for i in range(len(text) - self.n + 1):
                context = text[i : i + self.n - 1]
                char = text[i + self.n - 1]

                self.counts[context][char] += 1
                self.context_totals[context] += 1
                self.vocab.add(char)

        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Number of contexts: {len(self.counts)}")

    def prob(self, context, char):
        """Compute P(char | context) with add-k smoothing."""
        if len(context) != self.n - 1:
            context = ("^" * (self.n - 1) + context)[-(self.n - 1) :]

        count = self.counts[context][char]
        total = self.context_totals[context]
        vocab_size = len(self.vocab)

        return (count + self.k) / (total + self.k * vocab_size)

    def log_prob(self, text):
        """Compute log probability of entire text."""
        text = "^" * (self.n - 1) + text + "$"
        log_p = 0.0

        for i in range(len(text) - self.n + 1):
            context = text[i : i + self.n - 1]
            char = text[i + self.n - 1]
            log_p += math.log(self.prob(context, char))

        return log_p

    # Fix: Alias required by Decoder
    def score_word(self, word):
        return self.log_prob(word)

    def perplexity(self, texts):
        """Compute perplexity on a list of texts."""
        total_log_prob = 0.0
        total_chars = 0
        for text in texts:
            total_log_prob += self.log_prob(text)
            total_chars += len(text) + 1
        return math.exp(-total_log_prob / total_chars)
