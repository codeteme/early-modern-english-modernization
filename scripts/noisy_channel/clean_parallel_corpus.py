"""Clean parallel corpus by removing metadata and non-content lines."""

import re

def is_metadata_or_junk(line):
    """Check if a line is metadata, stage direction, or junk."""
    
    line = line.strip()
    
    # Empty lines
    if not line:
        return True
    
    # Metadata patterns
    metadata_patterns = [
        r'^={3,}',  # Lines of equal signs
        r'^FILE:',
        r'^Title:',
        r'^Version:',
        r'^Source:',
        r'^https?://',
        r'\.txt$',
        r'^\d+\s*$',  # Just numbers
        r'^Page \d+',
        r'^\[.*\]$',  # Lines entirely in brackets
    ]
    
    for pattern in metadata_patterns:
        if re.search(pattern, line):
            return True
    
    # Very short lines (likely not sentences)
    if len(line) < 3:
        return True
    
    # Stage directions (common in Shakespeare)
    if line.startswith('Enter ') or line.startswith('Exit') or line.startswith('Exeunt'):
        return True
    
    # Character names (all caps, short)
    if line.isupper() and len(line.split()) <= 3:
        return True
    
    return False


def is_sentence_pair_good(old_line, new_line):
    """Check if a sentence pair is worth keeping."""
    
    # Both should have reasonable length
    old_words = re.findall(r'\b\w+\b', old_line.lower())
    new_words = re.findall(r'\b\w+\b', new_line.lower())
    
    if len(old_words) < 3 or len(new_words) < 3:
        return False
    
    # Check word overlap (aligned pairs should share some words)
    old_set = set(old_words)
    new_set = set(new_words)
    overlap = len(old_set & new_set)
    total = len(old_set | new_set)
    
    if total > 0:
        similarity = overlap / total
        # Require at least 20% word overlap
        if similarity < 0.2:
            return False
    
    # Check length ratio (shouldn't be too different)
    ratio = min(len(old_words), len(new_words)) / max(len(old_words), len(new_words))
    if ratio < 0.3:  # One is 3x longer than the other
        return False
    
    return True


def clean_parallel_corpus(
    input_file="../../data/parallel_corpus.txt",
    output_file="../../data/parallel_corpus_clean.txt"
):
    """Clean the parallel corpus."""
    
    print("="*70)
    print("CLEANING PARALLEL CORPUS")
    print("="*70 + "\n")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Original: {len(lines)} lines ({len(lines)//2} pairs)")
    
    # Process pairs
    clean_pairs = []
    metadata_removed = 0
    quality_filtered = 0
    
    for i in range(0, len(lines), 2):
        if i+1 >= len(lines):
            break
        
        old_line = lines[i]
        new_line = lines[i+1]
        
        # Check for metadata
        if is_metadata_or_junk(old_line) or is_metadata_or_junk(new_line):
            metadata_removed += 1
            continue
        
        # Check pair quality
        if not is_sentence_pair_good(old_line, new_line):
            quality_filtered += 1
            continue
        
        clean_pairs.append((old_line, new_line))
    
    # Save cleaned corpus
    with open(output_file, 'w', encoding='utf-8') as f:
        for old_line, new_line in clean_pairs:
            f.write(old_line + '\n')
            f.write(new_line + '\n')
    
    print(f"✓ Cleaned: {len(clean_pairs)} pairs")
    print(f"  Removed {metadata_removed} pairs (metadata/junk)")
    print(f"  Removed {quality_filtered} pairs (quality filter)")
    print(f"  Kept {len(clean_pairs)}/{len(lines)//2} pairs "
          f"({len(clean_pairs)/(len(lines)//2)*100:.1f}%)\n")
    
    # Show some examples
    print("Sample cleaned pairs:")
    print("-"*70)
    for i, (old, new) in enumerate(clean_pairs[:10]):
        print(f"\n[{i+1}]")
        print(f"Old: {old[:65]}")
        print(f"New: {new[:65]}")
    
    print("\n" + "="*70)
    print(f"✓ Saved to: {output_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    clean_parallel_corpus()
    print("\nNext steps:")
    print("1. Check the cleaned corpus looks good")
    print("2. Update setup_v2.py to use 'parallel_corpus_clean.txt'")
    print("3. Run: python setup_v2.py")