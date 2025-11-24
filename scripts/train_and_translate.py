"""Step 3: Train noisy channel model and translate old to modern English."""

import csv
import re
import math
from collections import Counter, defaultdict


class NoisyChannelTranslator:
    """Simple noisy channel model with n-gram LM and weighted translations."""
    
    def __init__(self, n=3):
        self.n = n
        
        # Language model (n-gram)
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        
        # Translation weights
        self.word_weights = defaultdict(list)  # old -> [(modern, weight), ...]
        
    def load_weights(self, weights_file="data/word_weights.csv"):
        """Load precomputed word translation weights."""
        print(f"Loading weights from {weights_file}...")
        
        with open(weights_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                old = row['old']
                modern = row['modern']
                weight = float(row['weight'])
                self.word_weights[old].append((modern, weight))
        
        # Sort by weight (best first)
        for old in self.word_weights:
            self.word_weights[old].sort(key=lambda x: x[1])
        
        print(f"✓ Loaded weights for {len(self.word_weights)} words\n")
    
    def train_language_model(self, modern_texts):
        """Train n-gram language model on modern English."""
        print(f"Training {self.n}-gram language model...")
        
        for text in modern_texts:
            text = text.lower()
            # Pad with start/end markers
            padded = '^' * (self.n - 1) + text + '$'
            
            for i in range(len(padded) - self.n + 1):
                context = padded[i:i + self.n - 1]
                char = padded[i + self.n - 1]
                
                self.context_counts[context] += 1
                self.ngram_counts[(context, char)] += 1
        
        print(f"✓ Learned {len(self.ngram_counts)} n-grams\n")
    
    def language_model_score(self, text):
        """Score text using n-gram language model."""
        text = text.lower()
        padded = '^' * (self.n - 1) + text + '$'
        
        log_prob = 0.0
        alpha = 0.01  # Smoothing
        vocab_size = 100  # Approximate
        
        for i in range(len(padded) - self.n + 1):
            context = padded[i:i + self.n - 1]
            char = padded[i + self.n - 1]
            
            count = self.ngram_counts.get((context, char), 0)
            context_total = self.context_counts.get(context, 0)
            
            prob = (count + alpha) / (context_total + alpha * vocab_size)
            log_prob += math.log(prob) if prob > 0 else -20
        
        return log_prob
    
    def translate_word(self, old_word):
        """Get best modern translation for old word."""
        old_lower = old_word.lower()
        
        if old_lower in self.word_weights:
            # Return best translation (lowest weight)
            return self.word_weights[old_lower][0][0]
        
        # No translation found, return as-is
        return old_lower
    
    def translate(self, old_text):
        """Translate old English text to modern English."""
        # Extract words
        words = re.findall(r'\b\w+\b', old_text)
        
        # Translate each word
        translated = [self.translate_word(w) for w in words]
        
        return ' '.join(translated)
    
    def translate_with_score(self, old_text):
        """Translate and return with language model score."""
        translation = self.translate(old_text)
        lm_score = self.language_model_score(translation)
        return translation, lm_score


def load_parallel_pairs(file_path):
    """Load parallel pairs from alternating line format."""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            old = lines[i]
            modern = lines[i + 1]
            pairs.append((old, modern))
    
    return pairs


def evaluate(translator, test_pairs, num_samples=20):
    """Evaluate translator on test set."""
    
    print("="*60)
    print("EVALUATION ON TEST SET")
    print("="*60 + "\n")
    
    exact_matches = 0
    
    for i, (old_text, gold_modern) in enumerate(test_pairs[:num_samples]):
        print(f"[{i+1}/{num_samples}]")
        print("─"*60)
        print(f"📜 Old English:")
        print(f"   {old_text}")
        
        predicted, score = translator.translate_with_score(old_text)
        
        print(f"\n✨ Predicted:")
        print(f"   {predicted}")
        print(f"   [LM Score: {score:.2f}]")
        
        print(f"\n✅ Gold:")
        print(f"   {gold_modern}")
        
        # Check match
        if predicted.strip().lower() == gold_modern.strip().lower():
            print(f"\n🎯 EXACT MATCH!")
            exact_matches += 1
        else:
            print(f"\n❌ Different")
        
        print()
    
    accuracy = exact_matches / num_samples * 100
    print("="*60)
    print(f"ACCURACY: {exact_matches}/{num_samples} ({accuracy:.1f}%)")
    print("="*60 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and translate with noisy channel')
    parser.add_argument('--train', default='data/train_parallel.txt')
    parser.add_argument('--test', default='data/test_parallel.txt')
    parser.add_argument('--weights', default='data/word_weights.csv')
    parser.add_argument('--samples', type=int, default=20, help='Test samples')
    parser.add_argument('--mode', choices=['train', 'interactive'], default='train')
    
    args = parser.parse_args()
    
    print("="*60)
    print("STEP 3: TRAINING & TRANSLATION")
    print("="*60 + "\n")
    
    # Create translator
    translator = NoisyChannelTranslator(n=3)
    
    # Load weights
    translator.load_weights(args.weights)
    
    # Load training data
    print(f"Loading training data from {args.train}...")
    train_pairs = load_parallel_pairs(args.train)
    print(f"✓ Loaded {len(train_pairs)} training pairs\n")
    
    # Train language model on modern English
    modern_texts = [modern for _, modern in train_pairs]
    translator.train_language_model(modern_texts)
    
    if args.mode == 'train':
        # Load test data
        print(f"Loading test data from {args.test}...")
        test_pairs = load_parallel_pairs(args.test)
        print(f"✓ Loaded {len(test_pairs)} test pairs\n")
        
        # Evaluate
        evaluate(translator, test_pairs, num_samples=args.samples)
    
    else:
        # Interactive mode
        print("="*60)
        print("INTERACTIVE TRANSLATION MODE")
        print("="*60)
        print("Type old English text (or 'quit' to exit)\n")
        
        while True:
            text = input("Old English > ").strip()
            if not text or text.lower() == 'quit':
                break
            
            translation, score = translator.translate_with_score(text)
            print(f"Modern:     {translation}")
            print(f"LM Score:   {score:.2f}\n")


if __name__ == "__main__":
    main()