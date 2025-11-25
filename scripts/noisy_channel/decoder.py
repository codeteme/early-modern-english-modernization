"""
Noisy Channel Decoder.
Combines Language Model (P(M)) and Channel Model (P(E|M)) to find best correction.
"""

import math


class NoisyChannelDecoder:
    """Enhanced noisy channel model with smart candidate generation."""

    def __init__(self, lm, channel_model, lm_weight=1.5):
        self.lm = lm
        self.channel = channel_model
        self.lm_weight = lm_weight

        # Load the valid word list from the LM (if available)
        if hasattr(lm, "known_words"):
            self.vocab = lm.known_words
        else:
            self.vocab = set()

    def _edits1(self, word):
        """Generate all strings that are one edit distance away."""
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word):
        """Generate all strings that are two edits away."""
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))

    def generate_candidates(self, word):
        """Generate candidate corrections."""
        word_lower = word.lower()

        # 1. Generate 1-edit candidates
        candidates_1 = self._edits1(word_lower)

        # 2. Filter: Only keep candidates that are real words known to the LM
        valid_candidates = {word_lower}

        if self.vocab:
            # Add valid 1-edit words
            valid_candidates.update(w for w in candidates_1 if w in self.vocab)

            # Add valid 2-edit words only if needed (if very few 1-edit matches)
            if len(valid_candidates) < 5:
                candidates_2 = self._edits2(word_lower)
                valid_candidates.update(w for w in candidates_2 if w in self.vocab)
        else:
            # Fallback if no vocab: just return 1-edit distance strings
            valid_candidates.update(candidates_1)

        return list(valid_candidates)

    def score_candidate(self, old_word, candidate):
        """Score = log P(Modern) + log P(Old|Modern)"""
        # 1. Language Model Score (P(Modern))
        lm_score = self.lm.score_word(candidate)

        # 2. Channel Model Score (P(Old|Modern))
        prob = self.channel.channel_prob(candidate, old_word)
        channel_score = math.log(prob) if prob > 0 else -100.0

        return (self.lm_weight * lm_score) + channel_score

    def decode_greedy(self, word):
        """Find best candidate for a single word."""
        candidates = self.generate_candidates(word)

        best_word = word
        best_score = -float("inf")

        for cand in candidates:
            score = self.score_candidate(word, cand)

            # Boost exact dictionary matches slightly
            if cand in self.vocab:
                score += 0.5

            if score > best_score:
                best_score = score
                best_word = cand

        # Restore Case
        if word.istitle():
            return best_word.title(), best_score
        if word.isupper():
            return best_word.upper(), best_score

        return best_word, best_score


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
