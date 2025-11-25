"""Diagnose alignment and data quality issues."""

import re

def check_parallel_corpus():
    """Check the parallel corpus alignment quality."""
    
    print("="*70)
    print("CHECKING PARALLEL CORPUS ALIGNMENT")
    print("="*70 + "\n")
    
    with open('../../data/parallel_corpus.txt', 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Total lines in parallel corpus: {len(lines)}")
    print(f"Expected pairs: {len(lines) // 2}\n")
    
    # Check first 10 pairs
    print("First 10 sentence pairs:")
    print("-"*70)
    for i in range(0, min(20, len(lines)), 2):
        if i+1 >= len(lines):
            break
        old = lines[i]
        new = lines[i+1]
        
        # Check if they look similar (good alignment indicator)
        old_words = set(re.findall(r'\b\w+\b', old.lower()))
        new_words = set(re.findall(r'\b\w+\b', new.lower()))
        overlap = len(old_words & new_words)
        total = len(old_words | new_words)
        similarity = overlap / total if total > 0 else 0
        
        print(f"\n[Pair {i//2 + 1}] Similarity: {similarity:.2%}")
        print(f"Old: {old[:65]}")
        print(f"New: {new[:65]}")
    
    print("\n" + "="*70 + "\n")


def check_merged_files():
    """Check if merged files contain metadata/junk."""
    
    print("="*70)
    print("CHECKING MERGED FILES FOR METADATA")
    print("="*70 + "\n")
    
    with open('../../data/old_merged.txt', 'r', encoding='utf-8', errors='ignore') as f:
        old_lines = [line.strip() for line in f if line.strip()]
    
    with open('../../data/new_merged.txt', 'r', encoding='utf-8', errors='ignore') as f:
        new_lines = [line.strip() for line in f if line.strip()]
    
    print(f"Old merged: {len(old_lines)} lines")
    print(f"New merged: {len(new_lines)} lines\n")
    
    # Check for metadata patterns
    metadata_patterns = [
        r'={3,}',  # Multiple equal signs
        r'FILE:',
        r'Title:',
        r'Version:',
        r'Source:',
        r'https?://',
        r'\.txt$',
        r'^\d+$',  # Just numbers
    ]
    
    print("Checking for metadata in old_merged.txt (first 50 lines):")
    metadata_count = 0
    for i, line in enumerate(old_lines[:50]):
        is_metadata = any(re.search(pattern, line) for pattern in metadata_patterns)
        if is_metadata:
            metadata_count += 1
            print(f"  Line {i+1}: {line[:60]}")
    
    print(f"\nMetadata lines found: {metadata_count}/50")
    print("\n" + "="*70 + "\n")


def analyze_alignment_quality():
    """Analyze how well the parallel corpus is aligned."""
    
    print("="*70)
    print("ALIGNMENT QUALITY ANALYSIS")
    print("="*70 + "\n")
    
    with open('../../data/parallel_corpus.txt', 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    similarities = []
    length_ratios = []
    
    for i in range(0, min(200, len(lines)), 2):
        if i+1 >= len(lines):
            break
        
        old = lines[i]
        new = lines[i+1]
        
        # Word overlap similarity
        old_words = set(re.findall(r'\b\w+\b', old.lower()))
        new_words = set(re.findall(r'\b\w+\b', new.lower()))
        overlap = len(old_words & new_words)
        total = len(old_words | new_words)
        similarity = overlap / total if total > 0 else 0
        similarities.append(similarity)
        
        # Length ratio
        old_len = len(old_words)
        new_len = len(new_words)
        if old_len > 0 and new_len > 0:
            ratio = min(old_len, new_len) / max(old_len, new_len)
            length_ratios.append(ratio)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    avg_length_ratio = sum(length_ratios) / len(length_ratios) if length_ratios else 0
    
    print(f"Average word overlap: {avg_similarity:.2%}")
    print(f"Average length ratio: {avg_length_ratio:.2%}")
    print(f"\nInterpretation:")
    if avg_similarity > 0.6:
        print("✓ GOOD alignment (high word overlap)")
    elif avg_similarity > 0.3:
        print("⚠ MODERATE alignment (medium word overlap)")
    else:
        print("✗ POOR alignment (low word overlap)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    check_parallel_corpus()
    check_merged_files()
    analyze_alignment_quality()