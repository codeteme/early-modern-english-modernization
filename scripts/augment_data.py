#!/usr/bin/env python3
"""
Data Augmentation for Early Modern English Spelling Normalization

This script analyzes existing aligned pairs and generates synthetic training
examples based on common spelling transformation patterns.
"""

import pandas as pd
import re
from collections import defaultdict, Counter
from pathlib import Path

# Common Early Modern English to Modern English patterns
SPELLING_PATTERNS = [
    # Pattern: (old_pattern, modern_pattern, context)
    (r'ie$', 'y', 'suffix'),           # dutie → duty, pitty → pity
    (r'ue$', '', 'suffix'),            # haue → have, proue → prove
    (r'ee$', 'e$', 'suffix'),          # beene → been
    (r'ee', 'e', 'middle'),            # heere → here, yeere → year
    (r'oo', 'o', 'middle'),            # doore → door, booke → book
    (r'^vn', 'un', 'prefix'),          # vntender → untender, vnworthy → unworthy
    (r'v([aeiou])', r'u\1', 'middle'), # loue → love, haue → have
    (r'([aeiou])v([aeiou])', r'\1v\2', 'middle'), # diuine → divine
    (r"'d$", 'ed', 'suffix'),          # chang'd → changed, wash'd → washed
    (r'lly$', 'ly', 'suffix'),         # grossely → grossly (keep the ll)
    (r'our$', 'or', 'suffix'),         # honor → honour (bidirectional)
    (r'([^e])s$', r'\1s', 'suffix'),   # generall pattern for plural/singular
]

# Common word-ending transformations
ENDING_TRANSFORMS = {
    'ie': ['y'],
    'ue': ['e', ''],
    'es': ['s'],
    'ed': ["'d"],
    'ly': ['lie'],
    'our': ['or'],
    'or': ['our'],
}

def analyze_patterns(df):
    """Analyze existing pairs to find transformation patterns"""
    patterns = defaultdict(int)
    char_substitutions = Counter()

    for _, row in df.iterrows():
        old = str(row['old']).lower()
        modern = str(row['modern']).lower()

        # Skip if too different in length
        if abs(len(old) - len(modern)) > 3:
            continue

        # Character-level analysis
        if len(old) == len(modern):
            for i, (o, m) in enumerate(zip(old, modern)):
                if o != m:
                    context = 'start' if i == 0 else ('end' if i == len(old)-1 else 'middle')
                    char_substitutions[(o, m, context)] += 1

        # Ending analysis
        for old_end, mod_ends in ENDING_TRANSFORMS.items():
            if old.endswith(old_end):
                for mod_end in mod_ends:
                    if modern.endswith(mod_end):
                        patterns[(old_end, mod_end)] += 1

    return patterns, char_substitutions

def generate_synthetic_pairs(base_words, patterns, char_subs, count=500):
    """Generate synthetic training pairs using learned patterns"""
    synthetic = []

    # Common Early Modern English words to augment
    modern_words = [
        # Common verbs with 'ove/ave/ive' endings
        'have', 'love', 'give', 'move', 'prove', 'live', 'above', 'strive',
        'arrive', 'survive', 'forgive', 'receive', 'achieve', 'believe',
        'behave', 'enslave', 'forgave', 'gave', 'leave', 'weave', 'grieve',

        # Words ending in 'y' → 'ie'
        'duty', 'pity', 'city', 'party', 'beauty', 'plenty', 'twenty', 'thirty',
        'glory', 'story', 'history', 'victory', 'memory', 'company', 'family',
        'truly', 'holy', 'lovely', 'lively', 'friendly', 'deadly', 'goodly',

        # Words with 'here/there/where' patterns (double e)
        'here', 'there', 'where', 'were', 'been', 'seen', 'green', 'queen',
        'between', 'screen', 'keen',

        # Words with 'ear/eer' patterns
        'year', 'dear', 'near', 'fear', 'hear', 'clear', 'appear', 'tear',
        'bear', 'wear', 'swear', 'spear', 'shear', 'smear',

        # Words with 'oor' patterns
        'door', 'poor', 'floor', 'moor',

        # Words with 'un' prefix → 'vn'
        'unto', 'upon', 'under', 'until', 'unlike', 'unfold', 'undo', 'unless',
        'unfair', 'unknown', 'unwrap', 'untrue', 'unsafe', 'unhappy',

        # Past tense 'ed' → "'d"
        'changed', 'moved', 'loved', 'lived', 'proved', 'believed', 'received',
        'washed', 'pushed', 'wished', 'finished', 'promised', 'surprised',

        # Adverbs 'ly'
        'greatly', 'truly', 'merely', 'barely', 'fairly', 'rarely', 'nearly',
        'clearly', 'dearly', 'yearly',

        # Words with 'our' → 'or'
        'honour', 'favour', 'labour', 'neighbour', 'colour', 'harbour', 'rumour',
        'vigour', 'humour', 'behaviour', 'flavour', 'splendour',
        'know', 'knew', 'knowledge', 'knight', 'knave', 'kneel',
        'speak', 'break', 'great', 'meat', 'seat', 'beat', 'heat',
        'devil', 'evil', 'civil',
        'come', 'some', 'become', 'welcome', 'overcome',
        'news', 'views', 'jews',
        'majesty', 'honesty', 'modesty',
        'control', 'console', 'continue',
        'service', 'surface', 'practice', 'notice', 'justice',
        'jeweller', 'traveller', 'counsellor',
        'moved', 'loved', 'proved', 'believed', 'received',
        'people', 'purple', 'simple', 'example', 'temple',
        'heart', 'earth', 'heard',
        'emperor', 'error', 'terror',
        'promise', 'premise', 'compromise',
        'worthy', 'earthy', 'wealthy', 'healthy',
        'flowers', 'towers', 'powers', 'showers',
        'cousin', 'dozen', 'reason', 'season', 'treason',
        'virtue', 'torture', 'capture', 'nature', 'creature',
        'January', 'February', 'library', 'ordinary', 'necessary',
        'judgment', 'argument', 'instrument', 'monument',
        'beauty', 'bounty', 'county', 'plenty',
        'yoke', 'spoke', 'broke', 'stroke',
        'falls', 'calls', 'walls', 'halls', 'balls',
        'near', 'fear', 'year', 'clear', 'dear',
        'means', 'beans', 'cleans', 'leans',
        'afford', 'accord', 'record',
        'conserve', 'reserve', 'preserve', 'deserve', 'observe',
        'household', 'threshold',
        'younger', 'longer', 'stronger',
        'days', 'ways', 'says', 'plays', 'stays',
        'answered', 'considered', 'wondered', 'ordered',
        'ears', 'years', 'fears', 'tears', 'appears',
        'knaves', 'slaves', 'graves', 'waves',
        'resolved', 'dissolved', 'involved', 'revolved',
        'covetous', 'grievous', 'previous',
        'midst', 'midst',
        'replied', 'applied', 'supplied', 'implied',
        'arms', 'farms', 'harms', 'charms', 'alarms',
        'subjection', 'objection', 'rejection', 'projection',
        'mortality', 'morality', 'reality', 'quality',
        'living', 'giving', 'loving', 'moving',
        'virginity', 'divinity', 'trinity', 'affinity',
        'advantage', 'disadvantage',
        'years', 'tears', 'fears', 'hears',
        'young', 'tongue', 'lung',
        'prove', 'move', 'love', 'above',
        'shepherd', 'leopard',
        'pains', 'gains', 'chains', 'remains', 'trains',
        'delivered', 'considered', 'discovered',
        'delivery', 'discovery', 'recovery',
        'widow', 'window', 'shadow', 'meadow',
        'conference', 'reference', 'preference', 'difference',
        'reckoned', 'beckoned', 'weakened',
        'beloved', 'removed', 'improved',
        'cities', 'duties', 'beauties', 'parties',
    ]

    for word in modern_words:
        # Apply "u" → "v" transformation in middle/beginning (most common EME pattern)
        # But NOT at the end (to avoid "havue" instead of "haue")
        if 'u' in word[:-1]:  # Only replace 'u' not in the last position
            old_form = word[:-1].replace('u', 'v') + word[-1]
            if old_form != word and len(word) > 3:
                synthetic.append((old_form, word))

        # Apply "y" → "ie" transformation for endings
        if word.endswith('y') and len(word) > 3:
            old_form = word[:-1] + 'ie'
            synthetic.append((old_form, word))

        # Apply "ove/ave/ive" → "oue/aue/iue" transformation for verbs
        # This is a separate pattern from the 'u' → 'v' replacement
        if word.endswith(('ove', 'ave', 'ive')) and len(word) > 3:
            old_form = word[:-2] + 'ue'  # e.g., love → loue, have → haue
            synthetic.append((old_form, word))

        # Apply "un" → "vn" prefix transformation
        if word.startswith('un') and len(word) > 4:
            old_form = 'vn' + word[2:]
            synthetic.append((old_form, word))

        # Apply "ee" additions (only in common positions like "ere", "ear", "eet")
        if word.endswith('ere') or word.endswith('ear') or word.endswith('eet'):
            # here → heere, year → yeere, meet → meete
            pos = word.rfind('e', 0, -2)  # Find 'e' before the last 2 chars
            if pos > 0:
                old_form = word[:pos] + 'ee' + word[pos+1:]
                synthetic.append((old_form, word))
        # been pattern
        if word == 'been':
            synthetic.append(('beene', word))

        # Apply "'d" → "ed" transformation
        if word.endswith('ed') and len(word) > 4:
            old_form = word[:-2] + "'d"
            synthetic.append((old_form, word))

        # Apply "our" ↔ "or" transformation
        if 'our' in word:
            old_form = word.replace('our', 'or')
            if old_form != word:
                synthetic.append((word, old_form))  # Reverse too
        if 'or' in word and not 'our' in word:
            old_form = word.replace('or', 'our')
            if old_form != word and len(word) > 4:
                synthetic.append((old_form, word))

        # Apply "ll" simplification patterns
        if 'll' in word:
            # jeweller → jeweler pattern
            if word.endswith('ller'):
                old_form = word[:-2] + 'r'
                synthetic.append((word, old_form))

    # Deduplicate and quality filter
    seen = set()
    unique_synthetic = []
    for old, modern in synthetic:
        old_lower = old.lower()
        mod_lower = modern.lower()

        # Skip if identical
        if old_lower == mod_lower:
            continue

        # Skip if too short
        if len(old_lower) < 3 or len(mod_lower) < 3:
            continue

        # Skip if length difference too large
        if abs(len(old_lower) - len(mod_lower)) > 3:
            continue

        # Quality check: reject malformed patterns
        # e.g., "haveue" has "ue" suffix when it shouldn't
        if old_lower.endswith('ueue') or old_lower.endswith('ieie'):
            continue

        # Quality check: ensure words look reasonable (no triple letters except common ones)
        for triple in [old_lower[i:i+3] for i in range(len(old_lower)-2)]:
            if len(set(triple)) == 1 and triple[0] not in ['e', 's', 'l']:
                continue

        key = (old_lower, mod_lower)
        if key not in seen:
            seen.add(key)
            unique_synthetic.append((old, modern))

    return unique_synthetic[:count]

def main():
    """Main augmentation pipeline"""
    script_dir = Path(__file__).parent
    input_csv = script_dir / 'aligned_old_to_modern.csv'
    output_csv = script_dir / 'aligned_old_to_modern_augmented.csv'

    print("="*80)
    print("DATA AUGMENTATION FOR SPELLING NORMALIZATION")
    print("="*80)

    # Load existing data
    print(f"\nLoading existing data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Original dataset: {len(df)} pairs")

    # Analyze patterns
    print("\nAnalyzing transformation patterns...")
    patterns, char_subs = analyze_patterns(df)

    print(f"\nFound {len(patterns)} ending patterns:")
    for (old_end, mod_end), count in sorted(patterns.items(), key=lambda x: -x[1])[:10]:
        print(f"  {old_end:15} → {mod_end:15} ({count} occurrences)")

    print(f"\nFound {len(char_subs)} character substitutions:")
    for (old_c, mod_c, ctx), count in sorted(char_subs.items(), key=lambda x: -x[1])[:15]:
        print(f"  {old_c} → {mod_c} ({ctx:10}) ({count} occurrences)")

    # Generate synthetic pairs
    print("\nGenerating synthetic training pairs...")
    base_words = df['modern'].unique().tolist()
    synthetic_pairs = generate_synthetic_pairs(base_words, patterns, char_subs, count=800)

    print(f"Generated {len(synthetic_pairs)} synthetic pairs")

    # Combine original and synthetic
    original_pairs = [(row['old'], row['modern']) for _, row in df.iterrows()]
    all_pairs = original_pairs + synthetic_pairs

    # Create augmented dataframe
    augmented_df = pd.DataFrame(all_pairs, columns=['old', 'modern'])

    # Remove exact duplicates
    augmented_df = augmented_df.drop_duplicates()

    # Save augmented data
    augmented_df.to_csv(output_csv, index=False)

    print(f"\n{'='*80}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*80}")
    print(f"Original pairs:   {len(df)}")
    print(f"Synthetic pairs:  {len(synthetic_pairs)}")
    print(f"Total pairs:      {len(augmented_df)}")
    print(f"Output file:      {output_csv}")

    # Show sample synthetic pairs
    print(f"\nSample synthetic pairs:")
    for i, (old, modern) in enumerate(synthetic_pairs[:20], 1):
        print(f"  {i:2}. {old:20} → {modern}")

    print(f"\n{'='*80}")

if __name__ == '__main__':
    main()
