#!/usr/bin/env python3
"""
Evaluation script for spelling normalization models

This script evaluates both T5 and LSTM models on Old English text normalization.
All model classes are embedded to avoid import issues.

Usage:
    python scripts/eval.py --model-type lstm --input "I cannot conceiue you"
    python scripts/eval.py --model-type t5 --input-file test_sentences.txt
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import re


# ============================================================================
# LSTM Model Classes (embedded to avoid import issues)
# ============================================================================

class Encoder(nn.Module):
    """LSTM Encoder"""
    def __init__(self, input_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0,
                           bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    """Attention mechanism"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    """LSTM Decoder with Attention"""
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + embedding_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(x))
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """Sequence-to-Sequence model"""
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device


# ============================================================================
# T5 Model Loading
# ============================================================================

def load_t5_model(model_path='models/best_t5_model'):
    """Load a fine-tuned T5 model"""
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    print(f"Loading T5 model from {model_path}...")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()
    return model, tokenizer, device


def normalize_with_t5(text, model, tokenizer, device, max_length=128):
    """Normalize text using T5 model"""
    input_text = f"normalize spelling: {text}"
    input_encoding = tokenizer(
        input_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = input_encoding['input_ids'].to(device)
    attention_mask = input_encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            temperature=0.8
        )

    normalized = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return normalized


# ============================================================================
# LSTM Model Loading
# ============================================================================

def load_lstm_model(model_path='models/spelling_normalizer_pytorch.pt'):
    """Load a trained LSTM model"""
    print(f"Loading LSTM model from {model_path}...")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Extract checkpoint data
    input_vocab = checkpoint['input_vocab']
    target_vocab = checkpoint['target_vocab']
    reverse_target_vocab = checkpoint['reverse_target_vocab']
    max_len = checkpoint['max_len']
    embedding_dim = checkpoint['embedding_dim']
    hidden_dim = checkpoint['hidden_dim']
    n_layers = checkpoint['n_layers']

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("="*80)
        print("DEVICE CONFIGURATION")
        print("="*80)
        print("✓ MPS (Metal Performance Shaders) is available!")
        print("  Using Apple Silicon GPU acceleration")
        print(f"  Device: {device}")
        print("="*80 + "\n")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Build model using embedded classes
    encoder = Encoder(len(input_vocab), embedding_dim, hidden_dim, n_layers)
    decoder = Decoder(len(target_vocab), embedding_dim, hidden_dim, n_layers)
    model = Seq2Seq(encoder, decoder, device).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, input_vocab, target_vocab, reverse_target_vocab, max_len, device


def normalize_with_lstm(text, model, input_vocab, target_vocab, reverse_target_vocab, max_len, device):
    """Normalize text using LSTM model"""
    # Encode input
    input_seq = [input_vocab.get(char, 0) for char in text]
    input_seq = input_seq[:max_len] + [0] * (max_len - len(input_seq))
    input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(input_tensor)
        input_token = torch.tensor([[target_vocab['<START>']]]).to(device)

        decoded = []
        for _ in range(max_len):
            output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
            pred_token = output.argmax(1).item()

            if pred_token == target_vocab['<END>']:
                break

            char = reverse_target_vocab.get(pred_token, '')
            if char not in ['<PAD>', '<START>', '<END>']:
                decoded.append(char)

            input_token = torch.tensor([[pred_token]]).to(device)

    return ''.join(decoded)


# ============================================================================
# Sentence Processing
# ============================================================================

def tokenize_sentence(sentence):
    """Split sentence into words while preserving punctuation and spacing"""
    words = []
    spaces = []

    tokens = sentence.split()
    for i, token in enumerate(tokens):
        words.append(token)
        if i < len(tokens) - 1:
            spaces.append(' ')

    return words, spaces


def normalize_sentence_word_by_word(sentence, normalize_func):
    """Normalize a sentence by processing each word separately"""
    words, spaces = tokenize_sentence(sentence)
    normalized_words = []

    for word in words:
        # Separate punctuation
        match = re.match(r'^([^\w]*)([\w]+)([^\w]*)$', word)
        if match:
            prefix, core, suffix = match.groups()
            normalized_core = normalize_func(core)
            normalized_word = prefix + normalized_core + suffix
        else:
            normalized_word = normalize_func(word)

        normalized_words.append(normalized_word)

    # Reconstruct sentence
    result = []
    for i, word in enumerate(normalized_words):
        result.append(word)
        if i < len(spaces):
            result.append(spaces[i])

    return ''.join(result)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evalu3ate spelling normalization on sentences'
    )
    parser.add_argument(
        '--model-type',
        choices=['t5', 'lstm'],
        default='t5',
        help='Type of model to use (default: t5)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model (default: models/best_t5_model or models/spelling_normalizer_pytorch.pt)'
    )
    parser.add_argument(
        '--mode',
        choices=['word', 'sentence'],
        default='word',
        help='Process word-by-word or whole sentence (default: word)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input sentence to normalize'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='File with sentences to normalize (one per line)'
    )

    args = parser.parse_args()

    # Determine model path
    if args.model_path is None:
        args.model_path = 'models/best_t5_model' if args.model_type == 't5' else 'models/spelling_normalizer_pytorch.pt'

    # Load model
    if args.model_type == 't5':
        model, tokenizer, device = load_t5_model(args.model_path)
        normalize_func = lambda text: normalize_with_t5(text, model, tokenizer, device)
    else:
        model, input_vocab, target_vocab, reverse_target_vocab, max_len, device = load_lstm_model(args.model_path)
        normalize_func = lambda text: normalize_with_lstm(
            text, model, input_vocab, target_vocab, reverse_target_vocab, max_len, device
        )

    print(f"Model loaded successfully!")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print("=" * 80)

    # Get input sentences
    if args.input:
        sentences = [args.input]
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            sentences = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("\nInteractive mode - Enter sentences to normalize (Ctrl+C to exit)")
        print("=" * 80)
        sentences = []
        try:
            while True:
                sentence = input("\nOld English: ").strip()
                if sentence:
                    sentences.append(sentence)

                    # Process immediately in interactive mode
                    if args.mode == 'word':
                        normalized = normalize_sentence_word_by_word(sentence, normalize_func)
                    else:
                        normalized = normalize_func(sentence)

                    print(f"Modern:      {normalized}")
                    sentences = []  # Clear after processing
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return

    # Process batch mode
    if sentences:
        print("\nProcessing sentences...")
        print("=" * 80)

        for i, sentence in enumerate(sentences, 1):
            print(f"\n[{i}/{len(sentences)}]")
            print(f"Old:    {sentence}")

            if args.mode == 'word':
                normalized = normalize_sentence_word_by_word(sentence, normalize_func)
            else:
                normalized = normalize_func(sentence)

            print(f"Modern: {normalized}")

        print("\n" + "=" * 80)
        print("Processing complete!")


if __name__ == '__main__':
    main()
