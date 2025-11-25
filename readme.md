# Early Modern English to Modern English Modernization

A comparative study of three approaches for normalizing Early Modern English (Shakespearean) spelling to modern conventions: a probabilistic noisy channel model, an LSTM-based sequence-to-sequence model, and a fine-tuned T5 transformer.

## Overview

This project explores different methods for automatically modernizing Early Modern English text while preserving semantic meaning. The system is trained on parallel texts from Shakespeare's works (First Folio vs. modern editions).

### Models Implemented

1. **T5 Transformer** - Fine-tuned pre-trained transformer model (best performance)
2. **LSTM with Attention** - Character-level encoder-decoder with attention mechanism
3. **Noisy Channel Model** - Probabilistic model using word-level transformations

## Quick Start

### Prerequisites

- Python 3.8+
- GPU recommended (Apple Silicon MPS, CUDA, or CPU fallback supported)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd early-modern-english-modernization

# Install dependencies
pip install -r requirements.txt
```

### Requirements

The project requires the following packages (see [requirements.txt](requirements.txt)):

```
numpy>=1.21.0
python-Levenshtein>=0.12.2
matplotlib>=3.4.0
pandas>=1.3.0
tqdm>=4.62.0
torch>=2.0.0
beautifulsoup4>=4.14.2
torchvision
requests>=2.32.0
scikit-learn>=1.7.2
transformers>=4.57.2
sentencepiece>=0.2.1
```

## Data Collection

The project uses parallel texts from Shakespeare's works scraped from ShakespearesWords.com.

### Scrape Data

```bash
# Scrape Shakespeare texts from ShakespearesWords.com
# Work IDs range from 1-30 for different plays
cd scripts/data_generators
python scraper.py
```

The scraper saves texts to `data/raw/ShakespearesWordscom_<work_id>/`:
- `old_text.txt` - First Folio (original spelling)
- `new_text.txt` - Modern edition

### Data Processing

```bash
# Extract word-level alignments from scraped texts
cd scripts/data_generators
python extract_word_pairs.py
```

This creates aligned word pairs in `data/processed/aligned_old_to_modern_combined.csv`.

## Usage

### 1. T5 Transformer

The T5 model provides the best performance for most inputs.

**Training:**
```bash
cd scripts/seq2seq
python finetune_t5.py
```

**Evaluation:**
```bash
cd scripts/seq2seq

# Test with a single sentence
python eval_seq2seq.py --model-type t5 --input "Thou art most heartily welcome"

# Test with a file of sentences
python eval_seq2seq.py --model-type t5 --input-file ../../test_sentences.txt
```

**Outputs:**
- Model saved to: `models/best_t5_model/`
- Results saved to: `results/t5_results.txt`

### 2. LSTM with Attention

Character-level sequence-to-sequence model with attention mechanism.

**Training:**
```bash
cd scripts/seq2seq
python train_LSTM.py
```

**Evaluation:**
```bash
cd scripts/seq2seq

# Test with a single sentence
python eval_seq2seq.py --model-type lstm --input "Thou art most heartily welcome"

# Test with a file
python eval_seq2seq.py --model-type lstm --input-file ../../test_sentences.txt
```

**Outputs:**
- Model saved to: `models/best_model.pt`
- Results saved to: `results/lstm_results.txt`

### 3. Noisy Channel Model

Word-level probabilistic model combining a language model and channel model.

**Setup (one-time):**
```bash
cd scripts/noisy_channel
python setup.py
```

This creates:
- `data/train_parallel.txt` - Training set
- `data/test_parallel.txt` - Test set
- Word pair statistics for the channel model

**Training:**
```bash
cd scripts/noisy_channel
python train_from_word_pairs.py
```

**Translation:**
```bash
cd scripts/noisy_channel
python translate.py
```

The translate script offers:
- Batch translation of test set
- Interactive mode for custom sentences

**Outputs:**
- Model saved to: `models/word_pairs_model.pkl`

## Project Structure

```
early-modern-english-modernization/
├── data/
│   ├── raw/                          # Scraped Shakespeare texts
│   │   └── ShakespearesWordscom_*/   # Individual play directories
│   ├── processed/                    # Word alignments and pairs
│   ├── old_merged.txt                # Combined First Folio text
│   ├── new_merged.txt                # Combined modern text
│   ├── train_parallel.txt            # Training data
│   └── test_parallel.txt             # Test data
├── models/                           # Saved model checkpoints
│   ├── best_t5_model/               # Fine-tuned T5 model
│   ├── best_model.pt                # LSTM model
│   └── word_pairs_model.pkl         # Noisy channel model
├── scripts/
│   ├── data_generators/
│   │   ├── scraper.py               # Web scraper for Shakespeare texts
│   │   └── extract_word_pairs.py   # Extract word alignments
│   ├── seq2seq/
│   │   ├── finetune_t5.py          # Train T5 model
│   │   ├── train_LSTM.py           # Train LSTM model
│   │   └── eval_seq2seq.py         # Evaluate both seq2seq models
│   └── noisy_channel/
│       ├── setup.py                 # Setup parallel corpus
│       ├── train_from_word_pairs.py # Train channel model
│       ├── translate.py             # Interactive translation
│       ├── language_model.py        # Character n-gram LM
│       ├── channel_model.py         # Edit cost model
│       └── decoder.py               # Noisy channel decoder
├── results/                          # Evaluation outputs
├── test_sentences.txt                # Test sentences
├── requirements.txt                  # Python dependencies
└── model_comparison.md              # Detailed model comparison
```

## Example Transformations

The models learn to normalize common Early Modern English spelling patterns:

| Pattern | Old Form | Modern Form |
|---------|----------|-------------|
| u/v interchange | haue, vpon, vs | have, upon, us |
| Double vowels | seene, owne | seen, own |
| Archaic verbs | doth, hath | does, has |
| i/j variations | iourney | journey |
| Spelling | wisedome, euening | wisdom, evening |

**Example:**

**Input:** `I haue seene the truth with mine owne eyes.`

**T5 Output:** `I have seen the truth with mine own eyes.`

## Model Comparison

See [model_comparison.md](model_comparison.md) for detailed performance comparison on test sentences.

### Summary

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **T5** | Best overall accuracy, handles most spelling patterns | Occasional hallucinations on rare words |
| **LSTM** | Good at character-level patterns | Capitalization errors, mode collapse issues |
| **Noisy Channel** | Interpretable, fast | Limited pattern learning, requires extensive word pairs |

### Performance Highlights

- **T5**: Successfully normalizes most spelling variations (haue→have, seene→seen, owne→own)
- **LSTM**: Struggles with consistency, produces artifacts like "WITTOL", "Yhe"
- **Noisy Channel**: Learned basic patterns (Hee→He) but misses most transformations

## Testing with Custom Text

Create a text file with Early Modern English sentences (one per line):

```bash
# Create test file
cat > my_test.txt << EOF
Thou art most heartily welcome to this place.
I cannot conceiue why thou wouldst refuse.
The storme shall come vpon vs this euening.
EOF

# Test with T5
python scripts/seq2seq/eval_seq2seq.py --model-type t5 --input-file my_test.txt

# Test with LSTM
python scripts/seq2seq/eval_seq2seq.py --model-type lstm --input-file my_test.txt
```

## Development Notes

### GPU Acceleration

The training scripts automatically detect and use available GPU acceleration:
- **Apple Silicon**: MPS (Metal Performance Shaders)
- **NVIDIA**: CUDA
- **Fallback**: CPU

### Training Time

Approximate training times on Apple M1:
- **T5**: ~20-30 minutes (depends on dataset size)
- **LSTM**: ~15-25 minutes
- **Noisy Channel**: ~5-10 minutes

### Model Size

- **T5 model**: ~500MB (tokenizer + weights)
- **LSTM model**: ~9MB
- **Noisy Channel**: ~200KB

## References

- Data source: [Shakespeare's Words](https://www.shakespeareswords.com)
- T5 Paper: [Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)
- Noisy Channel approach: Probabilistic models for spelling correction

## License

This project is for educational purposes as part of ECE 684 coursework.
