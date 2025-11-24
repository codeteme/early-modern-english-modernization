"""Setup: Create parallel corpus and compute word weights (RUN ONCE)."""

import os
import csv
import re
import math
from collections import Counter


def create_parallel_corpus(
    old_file="../../data/old_merged.txt",
    new_file="../../data/new_merged.txt",
    train_split=0.8
):
    """Step 1: Create parallel corpus and train/test split."""
    
    print("="*60)
    print("STEP 1: CREATING PARALLEL CORPUS")
    print("="*60 + "\n")
    
    with open(old_file, 'r', encoding='utf-8', errors='ignore') as f:
        old_lines = [line.strip() for line in f if line.strip()]
    
    with open(new_file, 'r', encoding='utf-8', errors='ignore') as f:
        new_lines = [line.strip() for line in f if line.strip()]
    
    min_len = min(len(old_lines), len(new_lines))
    print(f"Creating {min_len} aligned pairs")
    
    # Split
    split_idx = int(min_len * train_split)
    
    # Train
    with open('../../data/train_parallel.txt', 'w', encoding='utf-8') as f:
        for i in range(split_idx):
            f.write(old_lines[i] + '\n')
            f.write(new_lines[i] + '\n')
    
    # Test
    with open('../../data/test_parallel.txt', 'w', encoding='utf-8') as f:
        for i in range(split_idx, min_len):
            f.write(old_lines[i] + '\n')
            f.write(new_lines[i] + '\n')
    
    print(f"✓ Train: {split_idx} pairs")
    print(f"✓ Test: {min_len - split_idx} pairs\n")


def compute_word_weights(
    aligned_csv="../../scripts/aligned_old_to_modern.csv",
    old_corpus="../../data/old_merged.txt",
    new_corpus="../../data/new_merged.txt",
    train_file="../../data/train_parallel.txt"
):
    """Step 2: Compute word translation weights."""
    
    print("="*60)
    print("STEP 2: COMPUTING WORD WEIGHTS")
    print("="*60 + "\n")
    
    # Load corpora
    with open(old_corpus, 'r', encoding='utf-8', errors='ignore') as f:
        old_text = f.read().lower()
    with open(new_corpus, 'r', encoding='utf-8', errors='ignore') as f:
        new_text = f.read().lower()
    
    old_words = re.findall(r'\b\w+\b', old_text)
    new_words = re.findall(r'\b\w+\b', new_text)
    
    old_freq = Counter(old_words)
    new_freq = Counter(new_words)
    
    print(f"Old corpus: {len(old_words):,} words")
    print(f"New corpus: {len(new_words):,} words\n")
    
    # Load aligned pairs
    pair_counts = Counter()
    old_word_total = Counter()
    
    # Try to load pre-existing alignment CSV, otherwise extract from parallel corpus
    if os.path.exists(aligned_csv):
        print(f"Loading alignments from: {aligned_csv}")
        with open(aligned_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                old = row['old'].strip().lower()
                modern = row['modern'].strip().lower()
                pair_counts[(old, modern)] += 1
                old_word_total[old] += 1
        print(f"Loaded {len(pair_counts)} word pair mappings\n")
    else:
        print(f"No pre-existing alignment file found at: {aligned_csv}")
        print("Extracting word alignments from parallel corpus...\n")
        
        # Extract alignments from parallel corpus
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        for i in range(0, len(lines), 2):
            if i+1 >= len(lines):
                break
            
            old_line = lines[i].lower()
            modern_line = lines[i+1].lower()
            
            old_line_words = re.findall(r'\b\w+\b', old_line)
            modern_line_words = re.findall(r'\b\w+\b', modern_line)
            
            # Simple word-by-word alignment (assumes similar word order)
            for old_word, modern_word in zip(old_line_words, modern_line_words):
                pair_counts[(old_word, modern_word)] += 1
                old_word_total[old_word] += 1
        
        print(f"Extracted {len(pair_counts)} word pair mappings from parallel corpus\n")
    
    # Compute weights
    weighted_pairs = []
    
    for (old, modern), pair_count in pair_counts.items():
        old_corpus_freq = old_freq.get(old, 0)
        modern_corpus_freq = new_freq.get(modern, 0)
        
        # P(modern|old) - translation probability
        p_trans = pair_count / old_word_total[old]
        
        # Channel cost: -log(P(modern|old))
        channel_cost = -math.log(p_trans) if p_trans > 0 else 100
        
        # Language model cost: prefer common modern words
        if modern_corpus_freq > 0:
            lm_cost = -math.log(modern_corpus_freq / len(new_words))
        else:
            lm_cost = 10
        
        # Combined weight (lower = better)
        weight = 0.7 * channel_cost + 0.3 * lm_cost
        
        weighted_pairs.append({
            'old': old,
            'modern': modern,
            'pair_count': pair_count,
            'old_corpus_freq': old_corpus_freq,
            'modern_corpus_freq': modern_corpus_freq,
            'p_translation': p_trans,
            'weight': weight
        })
    
    # Sort by weight (best first)
    weighted_pairs.sort(key=lambda x: x['weight'])
    
    # Save
    with open('../../data/word_weights.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'old', 'modern', 'pair_count', 'old_corpus_freq', 
            'modern_corpus_freq', 'p_translation', 'weight'
        ])
        writer.writeheader()
        writer.writerows(weighted_pairs)
    
    print(f"✓ Saved word_weights.csv\n")
    
    # Show top translations
    print("Top 15 most confident translations:")
    for i, pair in enumerate(weighted_pairs[:15], 1):
        print(f"{i:2d}. {pair['old']:12s} → {pair['modern']:12s} (weight: {pair['weight']:.2f})")


if __name__ == "__main__":
    create_parallel_corpus()
    compute_word_weights()
    print("\n✓ Setup complete! Now run: python translate.py")