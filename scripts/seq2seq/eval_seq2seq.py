#!/usr/bin/env python3
"""
Evaluation script for Seq2Seq models (T5 & LSTM).
Tests on ../../data/test_parallel.txt with Accuracy % reporting.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import re
import difflib
import sys
import os

# ============================================================================
# LSTM Model Classes (Must match training script exactly)
# ============================================================================


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim + embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(x))
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device


# ============================================================================
# Model Loading & Inference
# ============================================================================


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_t5_model(model_path):
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    # RESOLVE PATH: Fixes the "Repository Not Found" error
    abs_path = Path(model_path).resolve()
    if not abs_path.exists():
        # Try look relative to project root
        project_root = Path(__file__).parents[2]
        abs_path = (project_root / model_path).resolve()

    print(f"Loading T5 model from {abs_path}...")

    try:
        tokenizer = T5Tokenizer.from_pretrained(str(abs_path), local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(
            str(abs_path), local_files_only=True
        )
    except Exception as e:
        print(f"Error loading local T5: {e}")
        print("Using default 't5-small' weights instead...")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

    device = get_device()
    model.to(device)
    model.eval()
    return model, tokenizer, device


def normalize_with_t5(text, model, tokenizer, device):
    input_text = f"normalize spelling: {text}"
    input_encoding = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_encoding.input_ids, max_length=128, num_beams=4, early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_lstm_model(model_path):
    # Resolve path
    abs_path = Path(model_path).resolve()
    if not abs_path.exists():
        project_root = Path(__file__).parents[2]
        abs_path = (project_root / model_path).resolve()

    print(f"Loading LSTM model from {abs_path}...")
    checkpoint = torch.load(abs_path, map_location="cpu")

    device = get_device()
    encoder = Encoder(
        len(checkpoint["input_vocab"]),
        checkpoint["embedding_dim"],
        checkpoint["hidden_dim"],
        checkpoint["n_layers"],
    )
    decoder = Decoder(
        len(checkpoint["target_vocab"]),
        checkpoint["embedding_dim"],
        checkpoint["hidden_dim"],
        checkpoint["n_layers"],
    )
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, device


def normalize_with_lstm(text, model, checkpoint, device):
    input_vocab = checkpoint["input_vocab"]
    target_vocab = checkpoint["target_vocab"]
    reverse_target_vocab = checkpoint["reverse_target_vocab"]
    max_len = checkpoint["max_len"]

    input_seq = [input_vocab.get(c, 0) for c in text]
    input_seq = input_seq[:max_len] + [0] * (max_len - len(input_seq))
    input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(input_tensor)
        input_token = torch.tensor([[target_vocab["<START>"]]]).to(device)
        decoded = []
        for _ in range(max_len):
            output, hidden, cell, _ = model.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            pred_token = output.argmax(1).item()
            if pred_token == target_vocab["<END>"]:
                break
            char = reverse_target_vocab.get(pred_token, "")
            if char not in ["<PAD>", "<START>", "<END>"]:
                decoded.append(char)
            input_token = torch.tensor([[pred_token]]).to(device)

    return "".join(decoded)


# ============================================================================
# Evaluation Logic
# ============================================================================


def normalize_sentence_word_by_word(sentence, normalize_func):
    """Normalize a sentence word-by-word using regex tokenization."""
    tokens = sentence.split()
    normalized_words = []

    for token in tokens:
        match = re.match(r"^([^\w]*)([\w]+)([^\w]*)$", token)
        if match:
            prefix, core, suffix = match.groups()
            normalized_core = normalize_func(core)
            normalized_words.append(prefix + normalized_core + suffix)
        else:
            normalized_words.append(normalize_func(token))

    return " ".join(normalized_words)


def calculate_accuracy(pred, gold):
    pred_words = re.findall(r"\w+", pred.lower())
    gold_words = re.findall(r"\w+", gold.lower())
    if not pred_words and not gold_words:
        return 1.0
    return difflib.SequenceMatcher(None, pred_words, gold_words).ratio()


def evaluate_test_set(normalize_func):
    project_root = Path(__file__).parents[2]
    test_file = project_root / "data" / "test_parallel.txt"

    print(f"Evaluating on {test_file}")
    print("=" * 60)

    try:
        with open(test_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        total_acc = 0
        count = 0
        limit = min(
            1_770, len(lines)
        )  # Run test on the entire 885 pairs or 1,770 lines

        for i in range(0, limit, 2):
            if i + 1 >= len(lines):
                break
            old = lines[i]
            gold = lines[i + 1]

            # Use word-by-word normalization for consistency with Noisy Channel
            pred = normalize_sentence_word_by_word(old, normalize_func)

            acc = calculate_accuracy(pred, gold)
            total_acc += acc
            count += 1

            print(f"[{count}]")
            print(f"Old:  {old}")
            print(f"Pred: {pred}")
            print(f"Gold: {gold}")
            print(f"Accuracy: {acc:.1%}")
            print("-" * 40)

        if count > 0:
            print(f"\nAverage Accuracy on sample: {total_acc/count:.1%}")

    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["t5", "lstm"], default="t5")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = (
            "models/best_t5_model"
            if args.model_type == "t5"
            else "models/spelling_normalizer_pytorch.pt"
        )

    # Set up normalization function based on model type
    if args.model_type == "t5":
        model, tokenizer, device = load_t5_model(args.model_path)
        normalize_func = lambda text: normalize_with_t5(text, model, tokenizer, device)
    else:
        model, checkpoint, device = load_lstm_model(args.model_path)
        normalize_func = lambda text: normalize_with_lstm(
            text, model, checkpoint, device
        )

    print(f"Model loaded successfully! Device: {device}")
    evaluate_test_set(normalize_func)


if __name__ == "__main__":
    main()
