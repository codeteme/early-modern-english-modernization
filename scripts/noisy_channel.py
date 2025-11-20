"""Noisy channel decoder."""

import math
from .language_model import CharNgramLM
from .channel_model import ChannelModel


class NoisyChannelDecoder:
    """Noisy channel model: P(modern | early) ∝ P(early | modern) * P(modern)."""

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

    def score(self, modern, early):
        """
        Score a (modern, early) pair.

        Score = LM_weight * log P(modern) + log P(early | modern)
        """
        lm_score = self.lm.log_prob(modern)
        channel_score = math.log(self.channel.channel_prob(modern, early))

        return self.lm_weight * lm_score + channel_score

    def decode_greedy(self, early_text, candidates):
        """
        Greedy decoding: choose best candidate from list.

        Args:
            early_text: Early modern English input
            candidates: List of candidate modern English outputs

        Returns:
            best_candidate, best_score
        """
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
        Generate candidate modernizations (placeholder for now).

        For full implementation, you'd want:
        - Beam search
        - Edit-based candidate generation
        - Dictionary lookup

        For this project, we'll use simpler heuristics.
        """
        # Simple baseline: return input as-is as one candidate
        # You can expand this with:
        # - Apply common transformations (thou->you, etc.)
        # - Generate variants with different edit operations

        candidates = [early_text]  # Identity transform

        # Add some simple rule-based candidates
        simple_rules = {
            "thou": "you",
            "thee": "you",
            "thy": "your",
            "thine": "yours",
            "hath": "has",
            "doth": "does",
            "art": "are",
        }

        modified = early_text.lower()
        for old, new in simple_rules.items():
            modified = modified.replace(old, new)

        if modified != early_text.lower():
            candidates.append(modified)

        return candidates
