"""Main translator: Old English → Modern English with 5-gram LM and add-k smoothing."""

import csv
import re
import math
from collections import defaultdict, Counter


class Translator:
    """Complete translator with word weights, 5-gram LM with add-k smoothing, spell correction, and context."""
    
    def __init__(self, n=5, k=0.5):
        self.n = n  # 5-gram model
        self.k = k  # Add-k smoothing parameter
        self.word_map = defaultdict(list)  # old -> [(modern, weight), ...]
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.modern_vocab = Counter()
        self.vocab_size = 0  # Size of character vocabulary for smoothing
        
        # Context-aware mappings: (prev_word, old_word) -> modern_word
        self.bigram_translations = defaultdict(Counter)
        self.trigram_translations = defaultdict(Counter)
        
    def load_weights(self, weights_path="../../data/word_weights.csv"):
        """Load word translation weights."""
        with open(weights_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                old = row['old'].strip().lower()
                modern = row['modern'].strip().lower()
                weight = float(row['weight'])
                self.word_map[old].append((modern, weight))
        
        for old in self.word_map:
            self.word_map[old].sort(key=lambda x: x[1])
        
        print(f"✓ Loaded {len(self.word_map)} word mappings")
    
    def train_lm(self, train_file="../../data/train_parallel.txt"):
        """Train 5-gram language model with add-k smoothing."""
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        modern_texts = [lines[i] for i in range(1, len(lines), 2)]
        
        # Collect character vocabulary
        char_vocab = set()
        
        for text in modern_texts:
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            self.modern_vocab.update(words)
            
            # Collect characters for vocabulary size
            char_vocab.update(text_lower)
            
            # Pad with n-1 start symbols and 1 end symbol
            padded = '^' * (self.n - 1) + text_lower + '$'
            
            # Count n-grams (5-grams)
            for i in range(len(padded) - self.n + 1):
                context = padded[i:i + self.n - 1]  # 4 characters of context
                char = padded[i + self.n - 1]       # 5th character
                self.context_counts[context] += 1
                self.ngram_counts[(context, char)] += 1
        
        # Set vocabulary size (all unique characters seen + special symbols)
        self.vocab_size = len(char_vocab) + 2  # +2 for ^ and $
        
        print(f"✓ Trained {self.n}-gram LM ({len(self.modern_vocab)} word vocab, {self.vocab_size} char vocab)")
        print(f"  Add-k smoothing: k={self.k}")
    
    def get_ngram_prob(self, context, char):
        """Get smoothed n-gram probability using add-k smoothing.
        
        P(char | context) = (count(context, char) + k) / (count(context) + k * V)
        where V is the vocabulary size.
        """
        numerator = self.ngram_counts[(context, char)] + self.k
        denominator = self.context_counts[context] + self.k * self.vocab_size
        
        if denominator == 0:
            return 1.0 / self.vocab_size  # Uniform distribution fallback
        
        return numerator / denominator
    
    def score_word(self, word):
        """Score a word using the 5-gram language model with add-k smoothing."""
        word_lower = word.lower()
        padded = '^' * (self.n - 1) + word_lower + '$'
        
        log_prob = 0.0
        for i in range(len(padded) - self.n + 1):
            context = padded[i:i + self.n - 1]
            char = padded[i + self.n - 1]
            prob = self.get_ngram_prob(context, char)
            log_prob += math.log(prob)
        
        return log_prob
    
    def edit_distance(self, s1, s2):
        """Compute edit distance."""
        if len(s1) < len(s2):
            return self.edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def spell_correct(self, word, max_dist=2):
        """Spell correct using edit distance + frequency + 5-gram LM."""
        word_lower = word.lower()
        
        # Already in vocab - no correction needed
        if word_lower in self.modern_vocab and self.modern_vocab[word_lower] > 3:
            return word_lower, False
        
        # Find candidates within edit distance
        candidates = []
        for vocab_word, freq in self.modern_vocab.items():
            # Skip if too infrequent
            if freq < 2:
                continue
                
            dist = self.edit_distance(word_lower, vocab_word)
            
            if dist == 0:
                continue  # Same word
            
            if dist <= max_dist:
                # Score components:
                # 1. Frequency score
                freq_score = math.log(freq + 1)
                
                # 2. Language model score (5-gram with add-k smoothing)
                lm_score = self.score_word(vocab_word)
                
                # 3. Distance penalty
                distance_penalty = dist * 3
                
                # Combined score: higher frequency + higher LM prob + lower distance = better
                score = freq_score + 0.5 * lm_score - distance_penalty
                candidates.append((vocab_word, score, dist, freq))
        
        if not candidates:
            return word_lower, False
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Get best candidate
        best_word, best_score, dist, freq = candidates[0]
        
        # Only correct if:
        # 1. Distance is 1-2
        # 2. Score is positive
        # 3. Candidate is much more frequent (at least 5x or 10+ occurrences)
        current_freq = self.modern_vocab.get(word_lower, 0)
        
        if dist <= 2 and best_score > 0 and (freq >= 10 or freq > current_freq * 5):
            return best_word, True
        
        return word_lower, False
    
    def translate_word(self, word):
        """Translate single word."""
        word_lower = word.lower()
        if word_lower in self.word_map:
            return self.word_map[word_lower][0][0]
        return word_lower
    
    def translate(self, text, spell_check=True, verbose=False):
        """Full translation pipeline."""
        words = re.findall(r'\b\w+\b', text)
        
        # Step 1: Translate
        translated = [self.translate_word(w) for w in words]
        
        if verbose:
            print(f"\nAfter translation: {' '.join(translated)}")
        
        # Step 2: Spell check EVERY word
        corrections = []
        if spell_check:
            corrected = []
            for word in translated:
                corrected_word, was_corrected = self.spell_correct(word)
                corrected.append(corrected_word)
                if was_corrected:
                    corrections.append(f"{word} → {corrected_word}")
                    if verbose:
                        print(f"  Correcting: {word} → {corrected_word}")
            translated = corrected
        
        result = ' '.join(translated)
        if result and text and text[0].isupper():
            result = result[0].upper() + result[1:]
        
        return result, corrections


def evaluate(translator, test_file="../../data/test_parallel.txt", n=20):
    """Evaluate on test set."""
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print("\n" + "="*70)
    print("EVALUATION (5-GRAM WITH ADD-K SMOOTHING)")
    print("="*70 + "\n")
    
    exact = 0
    total_corrections = 0
    
    for i in range(0, min(n*2, len(lines)), 2):
        if i+1 >= len(lines):
            break
        
        old_text = lines[i]
        gold = lines[i+1]
        
        predicted, corrections = translator.translate(old_text, spell_check=True)
        total_corrections += len(corrections)
        
        print(f"[{i//2 + 1}]")
        print(f"Old:  {old_text[:65]}")
        print(f"Pred: {predicted[:65]}")
        if corrections:
            print(f"🔧 Spell corrections: {', '.join(corrections)}")
        print(f"Gold: {gold[:65]}")
        
        if predicted.strip().lower() == gold.strip().lower():
            print("✓ EXACT MATCH")
            exact += 1
        else:
            # Word-level accuracy
            pred_words = predicted.lower().split()
            gold_words = gold.lower().split()
            matches = sum(1 for p, g in zip(pred_words, gold_words) if p == g)
            total_words = max(len(pred_words), len(gold_words))
            word_acc = matches / total_words if total_words > 0 else 0
            print(f"✗ Word accuracy: {word_acc*100:.1f}%")
        print()
    
    print("="*70)
    print(f"Exact matches: {exact}/{n} ({exact/n*100:.1f}%)")
    print(f"Total spell corrections: {total_corrections}")
    print("="*70 + "\n")


def interactive(translator):
    """Interactive mode."""
    print("\n" + "="*70)
    print("INTERACTIVE MODE (5-GRAM WITH ADD-K SMOOTHING)")
    print("="*70)
    print("Commands:")
    print("  <text>        - Translate text")
    print("  debug <text>  - Translate with debug info")
    print("  quit          - Exit")
    print("="*70 + "\n")
    
    while True:
        text = input("Old English > ").strip()
        if not text or text.lower() == 'quit':
            break
        
        verbose = False
        if text.startswith('debug '):
            verbose = True
            text = text[6:]
        
        translation, corrections = translator.translate(text, spell_check=True, verbose=verbose)
        print(f"Modern: {translation}")
        if corrections:
            print(f"🔧 Corrections ({len(corrections)}): {', '.join(corrections)}")
        print()


if __name__ == "__main__":
    import sys
    
    print("Loading translator with 5-gram LM and add-k smoothing...")
    t = Translator(n=5, k=0.5)
    t.load_weights()
    t.train_lm()
    print()
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    
    if mode == "eval":
        evaluate(t, n=20)
    else:
        interactive(t)