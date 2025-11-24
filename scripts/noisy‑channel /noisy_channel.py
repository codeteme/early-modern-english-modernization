"""Enhanced noisy channel decoder with better candidate generation."""

import math
import re
from .language_model import CharNgramLM
from .channel_model import ChannelModel


class NoisyChannelDecoder:
    """Enhanced noisy channel model with smart candidate generation."""

    def __init__(self, language_model, channel_model, lm_weight=1.0):
        """
        Args:
            language_model: CharNgramLM instance
            channel_model: ChannelModel instance
            lm_weight: Weight for language model score
        """
        self.lm = language_model
        self.channel = channel_model
        self.lm_weight = lm_weight

        # Build common transformation rules from learned weights
        self._build_transformation_rules()

    def _build_transformation_rules(self):
        """Extract common word-level transformations from channel model."""
        self.word_rules = {}

        if hasattr(self.channel, "word_mappings"):
            # Get top transformations
            for (early, modern), count in self.channel.word_mappings.most_common(100):
                if count >= 2:  # Only if seen at least twice
                    self.word_rules[early] = modern

        print(f"Built {len(self.word_rules)} transformation rules")

    def score(self, modern, early):
        """Score a (modern, early) pair."""
        lm_score = self.lm.log_prob(modern)
        channel_score = math.log(self.channel.channel_prob(modern, early))

        return self.lm_weight * lm_score + channel_score

    def decode_greedy(self, early_text, candidates=None):
        """
        Greedy decoding: choose best candidate.

        Args:
            early_text: Early modern English input
            candidates: Optional list of candidates (auto-generated if None)

        Returns:
            best_candidate, best_score
        """
        if candidates is None:
            candidates = self.generate_candidates(early_text)

        best_candidate = None
        best_score = float("-inf")

        for candidate in candidates:
            score = self.score(candidate, early_text)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate, best_score

    def generate_candidates(self, early_text):
        """
        Generate candidate modernizations using learned transformations.

        This is much better than before!
        """
        candidates = []

        # Candidate 1: No change
        candidates.append(early_text)

        # Candidate 2: Apply learned word-level rules
        words = re.findall(r"\b\w+\b", early_text)
        transformed_words = []

        for word in words:
            if word in self.word_rules:
                transformed_words.append(self.word_rules[word])
            else:
                transformed_words.append(word)

        # Reconstruct sentence
        if transformed_words != words:
            # Simple reconstruction (preserves spaces/punctuation roughly)
            transformed = early_text
            for old, new in zip(words, transformed_words):
                if old != new:
                    transformed = re.sub(
                        r"\b" + re.escape(old) + r"\b", new, transformed, count=1
                    )
            candidates.append(transformed)

        # Candidate 3: Apply common character-level patterns
        # Look for common substitutions learned from data
        if hasattr(self.channel, "substitute_counts"):
            common_subs = self.channel.substitute_counts.most_common(20)

            variant = early_text
            for (old_char, new_char), count in common_subs:
                if count >= 5:  # Only frequent substitutions
                    # Try reversing the substitution
                    variant = variant.replace(new_char, old_char)

            if variant != early_text:
                candidates.append(variant)

        # Candidate 4: Lowercase version (handles capitalization)
        if early_text != early_text.lower():
            candidates.append(early_text.lower())

        # Candidate 5: Apply multiple rules together
        # Combine word and character transformations
        if len(candidates) > 2:
            # Apply rules to the word-transformed version
            combined = candidates[1] if len(candidates) > 1 else early_text

            for (old_char, new_char), count in common_subs[:10]:
                if count >= 5:
                    combined = combined.replace(new_char, old_char)

            if combined not in candidates:
                candidates.append(combined)

        return list(set(candidates))  # Remove duplicates


### Updated Data Format Instructions

# Create `data/raw/shakespeare_parallel.txt` with your format:
# ```
# In delivering my son from me, I bury a second husband.
# In saying goodbye to my son, it's like I'm losing another husband.
# Thou art a wise man and know'st well enough.
# You are a wise man and know well enough.
# What says my lady to my suit?
# What does my lady say about my proposal?
# I pray thee cease thy counsel which falls into mine ears.
# I beg you to stop your advice which falls on my ears.
# ...
