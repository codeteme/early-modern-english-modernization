# Early Modern English to Modern English Modernization

A probabilistic noisy channel model that learns to translate Shakespearean English to modern English from parallel data.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Generate synthetic data (Step 3a)
python scripts/generate_synthetic_data.py

# Add your real data to data/raw/shakespeare_parallel.txt (Step 3b)
# Format: alternating lines of early/modern English

# Preprocess real data
python scripts/preprocess_data.py

# Train models
python scripts/train.py

# Evaluate
python scripts/evaluate.py
```

## Data Format

Create `data/raw/shakespeare_parallel.txt`:
```
In delivering my son from me, I bury a second husband.
In saying goodbye to my son, it's like I'm losing another husband.
Thou art a wise man and know'st well enough.
You are a wise man and know well enough.
```

**Format:** Alternating lines (line 1 = early, line 2 = modern, line 3 = early, etc.)

**Minimum:** 100+ sentence pairs

## Model

**Noisy Channel:** `P(modern | early) ∝ P(modern) × P(early | modern)`

- **Language Model:** 5-gram character model for fluent modern English
- **Channel Model:** Learned edit costs from parallel data
  - Character-level: insert/delete/substitute costs
  - Word-level: direct word mappings (thou→you)
  - Context-sensitive: surrounding character patterns

We have two primarily files for our noisy channel model, `setup.py` and `translate.py`. Run the following to get the parallel corpus, train and test sets, and word pairs and frequencies based on the corpora.

The model will actually run for the second script and we can see output for a random test set in the terminal. There is also an interactive option to put in your own sentences.

```bash
python scripts/setup.py
python scripts/translate.py
```

**Novel Contribution:** Learns transformation patterns from data rather than using fixed rules.

## Expected Results

| Dataset | Exact Match | Character Error Rate |
|---------|-------------|---------------------|
| Synthetic | 70-90% | 0.05-0.15 |
| Real | 20-40% | 0.3-0.5 |

## Project Structure

```
├── src/
│   ├── language_model.py    # Character n-gram LM
│   ├── channel_model.py     # Edit cost learning
│   ├── noisy_channel.py     # Decoder
│   └── utils.py
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── preprocess_data.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   ├── raw/shakespeare_parallel.txt    # Your data here
│   ├── synthetic/                       # Auto-generated
│   └── processed/                       # Auto-generated
└── results/                             # Evaluation outputs
```

## Requirements

```
numpy>=1.21.0
python-Levenshtein>=0.12.2
matplotlib>=3.4.0
pandas>=1.3.0
tqdm>=4.62.0
```

## Example

**Input:** `thou art wise`

**Output:** `you are wise`

**How:** Model learned from data that "thou→you" and "art→are" are common transformations with low cost.