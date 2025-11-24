#!/usr/bin/env python3
"""
Spelling Normalization using PyTorch Encoder-Decoder Network with MPS Support

This script trains a character-level encoder-decoder model with attention
to normalize old English spelling to modern spelling.
Works seamlessly with Apple Silicon MPS acceleration.

ANTI-MODE-COLLAPSE MEASURES:
1. Data Augmentation: 3x dataset size with identity mappings and variations
2. Label Smoothing: 0.15 to prevent overconfidence
3. Strong Teacher Forcing: Maintains 70%+ throughout training
4. Proper Weight Initialization: Xavier/Glorot initialization
5. Lower Weight Decay: 0.005 to allow learning
6. Larger Batch Size: 32 for better gradient estimates
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
print("=" * 80)
print("DEVICE CONFIGURATION")
print("=" * 80)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ MPS (Metal Performance Shaders) is available!")
    print("  Using Apple Silicon GPU acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ CUDA is available!")
    print("  Using NVIDIA GPU acceleration")
else:
    device = torch.device("cpu")
    print("✗ No GPU acceleration available")
    print("  Using CPU")
print(f"  Device: {device}")
print("=" * 80 + "\n")


class SpellingDataset(Dataset):
    """Dataset for spelling normalization"""

    def __init__(self, input_texts, target_texts, input_vocab, target_vocab, max_len):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        # Encode input
        input_seq = [self.input_vocab.get(char, 0) for char in input_text]
        input_seq = input_seq[: self.max_len] + [0] * (self.max_len - len(input_seq))

        # Encode target (with start and end tokens)
        target_seq = (
            [self.target_vocab["<START>"]]
            + [self.target_vocab.get(char, 0) for char in target_text]
            + [self.target_vocab["<END>"]]
        )
        target_seq = target_seq[: self.max_len] + [0] * (self.max_len - len(target_seq))

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
        )


class Encoder(nn.Module):
    """LSTM Encoder - Optimized for word-level learning"""

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
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
        )  # Keep unidirectional for memory
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)
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
        # hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, seq_len, hidden_dim)

        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        # Repeat hidden state seq_len times
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    """LSTM Decoder with Attention - Optimized for word-level learning"""

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
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights properly to prevent mode collapse
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if "weight" in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch, 1)
        embedded = self.dropout(self.embedding(x))

        # Calculate attention
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)

        # Apply attention to encoder outputs
        context = torch.bmm(attn_weights, encoder_outputs)

        # Concatenate embedded input and context
        lstm_input = torch.cat((embedded, context), dim=2)

        # LSTM step
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # Prediction
        prediction = self.fc(output.squeeze(1))

        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """Sequence-to-Sequence model"""

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        # First input is start token
        input_token = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            outputs[:, t] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs


class SpellingNormalizer:
    """Spelling normalization model manager"""

    def __init__(self, embedding_dim=128, hidden_dim=256, n_layers=2):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.model = None

    def prepare_data(self, csv_path, test_size=0.15, val_size=0.15, augment=True):
        """Load and prepare data with optional augmentation

        Preserves case for better generalization
        """
        print("Loading data from CSV...")
        df = pd.read_csv(csv_path)

        # Preserve case for better generalization
        input_texts = df["old"].astype(str).tolist()
        target_texts = df["modern"].astype(str).tolist()

        print(f"Loaded {len(input_texts)} text pairs (case-preserved)")

        # Data augmentation to increase dataset size
        if augment:
            print("Applying data augmentation...")
            aug_input = []
            aug_target = []

            # 1. Add identity mappings (modern -> modern) to teach copying
            for text in target_texts:
                aug_input.append(text)
                aug_target.append(text)

            # 2. Add character-level noise for robustness
            import random

            for old, modern in zip(
                input_texts[:100], target_texts[:100]
            ):  # Limit to avoid too much noise
                # Randomly duplicate a character
                if len(old) > 2 and random.random() < 0.3:
                    pos = random.randint(1, len(old) - 1)
                    noisy = old[:pos] + old[pos] + old[pos:]
                    aug_input.append(noisy)
                    aug_target.append(modern)

            # 3. Add reverse identity mappings (helps prevent mode collapse)
            # This teaches the model that different inputs should produce different outputs
            for modern in set(target_texts):
                if modern not in input_texts:
                    aug_input.append(modern)
                    aug_target.append(modern)

            input_texts.extend(aug_input)
            target_texts.extend(aug_target)

            print(f"After augmentation: {len(input_texts)} text pairs")

        # Build vocabularies
        input_chars = set()
        target_chars = set()

        for text in input_texts:
            input_chars.update(text)
        for text in target_texts:
            target_chars.update(text)

        # Create vocabulary mappings
        self.input_vocab = {
            char: idx + 1 for idx, char in enumerate(sorted(input_chars))
        }
        self.input_vocab["<PAD>"] = 0

        self.target_vocab = {
            char: idx + 3 for idx, char in enumerate(sorted(target_chars))
        }
        self.target_vocab["<PAD>"] = 0
        self.target_vocab["<START>"] = 1
        self.target_vocab["<END>"] = 2

        self.reverse_target_vocab = {v: k for k, v in self.target_vocab.items()}

        self.max_len = (
            max(max(len(t) for t in input_texts), max(len(t) for t in target_texts)) + 2
        )

        print(f"Input vocabulary size: {len(self.input_vocab)}")
        print(f"Target vocabulary size: {len(self.target_vocab)}")
        print(f"Max sequence length: {self.max_len}")

        # Split data
        train_input, test_input, train_target, test_target = train_test_split(
            input_texts, target_texts, test_size=test_size, random_state=42
        )

        train_input, val_input, train_target, val_target = train_test_split(
            train_input, train_target, test_size=val_size, random_state=42
        )

        print(f"Training samples: {len(train_input)}")
        print(f"Validation samples: {len(val_input)}")
        print(f"Test samples: {len(test_input)}")

        # Create datasets
        self.train_dataset = SpellingDataset(
            train_input, train_target, self.input_vocab, self.target_vocab, self.max_len
        )
        self.val_dataset = SpellingDataset(
            val_input, val_target, self.input_vocab, self.target_vocab, self.max_len
        )
        self.test_input = test_input
        self.test_target = test_target

    def build_model(self):
        """Build the encoder-decoder model"""
        print("\nBuilding encoder-decoder model...")

        encoder = Encoder(
            len(self.input_vocab), self.embedding_dim, self.hidden_dim, self.n_layers
        )
        decoder = Decoder(
            len(self.target_vocab), self.embedding_dim, self.hidden_dim, self.n_layers
        )

        self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

    def train(
        self, epochs=50, batch_size=16, lr=0.003, accumulation_steps=4, warmup_epochs=5
    ):
        """Train the model with memory-efficient strategies

        Args:
            epochs: Number of training epochs
            batch_size: Actual batch size (reduced for MPS memory constraints)
            lr: Learning rate (increased for better convergence)
            accumulation_steps: Gradient accumulation steps (effective_batch_size = batch_size * accumulation_steps)
            warmup_epochs: Number of epochs for learning rate warmup
        """
        print(f"\nTraining for {epochs} epochs...")
        print(
            f"Batch size: {batch_size}, Gradient accumulation steps: {accumulation_steps}"
        )
        print(f"Effective batch size: {batch_size * accumulation_steps}")
        print(f"Initial learning rate: {lr} (with {warmup_epochs} epoch warmup)")

        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        # Use label smoothing to prevent overconfidence and mode collapse
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.15)
        # Lower weight decay to allow model to learn better
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.005)

        # Cosine annealing with warmup for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10  # Increased patience for better training

        for epoch in range(epochs):
            # Learning rate warmup
            if epoch < warmup_epochs:
                warmup_lr = lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr
                current_lr = warmup_lr
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            optimizer.zero_grad()

            for batch_idx, (src, trg) in enumerate(pbar):
                src, trg = src.to(self.device), trg.to(self.device)

                # More aggressive teacher forcing schedule to prevent mode collapse
                # Start high and decay slowly
                teacher_forcing_ratio = max(0.7, 1.0 - (epoch / (epochs * 2)))
                output = self.model(
                    src, trg, teacher_forcing_ratio=teacher_forcing_ratio
                )

                # Reshape for loss calculation
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = criterion(output, trg)

                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()

                # Only update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(
                    train_loader
                ):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Clear MPS cache to prevent memory buildup
                    if self.device.type == "mps":
                        torch.mps.empty_cache()

                train_loss += loss.item() * accumulation_steps

                # Calculate accuracy
                pred = output.argmax(1)
                mask = trg != 0
                train_correct += ((pred == trg) & mask).sum().item()
                train_total += mask.sum().item()

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item() * accumulation_steps:.4f}",
                        "lr": f"{current_lr:.6f}",
                    }
                )

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0

            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for src, trg in val_loader:
                    src, trg = src.to(self.device), trg.to(self.device)
                    output = self.model(src, trg, teacher_forcing_ratio=0)

                    output_dim = output.shape[-1]
                    output = output[:, 1:].reshape(-1, output_dim)
                    trg = trg[:, 1:].reshape(-1)

                    loss = criterion(output, trg)
                    val_loss += loss.item()

                    pred = output.argmax(1)
                    mask = trg != 0
                    val_correct += ((pred == trg) & mask).sum().item()
                    val_total += mask.sum().item()

                    # Clear MPS cache during validation too
                    if self.device.type == "mps":
                        torch.mps.empty_cache()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total if val_total > 0 else 0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            print(
                f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}"
            )

            # Step scheduler after warmup
            if epoch >= warmup_epochs:
                scheduler.step()

            # Early stopping with improvement threshold
            if val_loss < best_val_loss - 0.001:  # Require meaningful improvement
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")
                patience_counter = 0
                print(f"  → New best model saved! (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load("best_model.pt"))
        self._plot_history(history)

        return history

    def _plot_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Model Loss")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(history["train_acc"], label="Train")
        plt.plot(history["val_acc"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Model Accuracy")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(history["lr"])
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150)
        print("\nTraining history saved as 'training_history.png'")
        plt.close()

    def normalize_text(self, text):
        """Normalize spelling of input text"""
        self.model.eval()

        # Encode input
        input_seq = [self.input_vocab.get(char, 0) for char in text]
        input_seq = input_seq[: self.max_len] + [0] * (self.max_len - len(input_seq))
        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(self.device)

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.model.encoder(input_tensor)

            # Start with <START> token
            input_token = torch.tensor([[self.target_vocab["<START>"]]]).to(self.device)

            decoded = []
            for _ in range(self.max_len):
                output, hidden, cell, _ = self.model.decoder(
                    input_token, hidden, cell, encoder_outputs
                )
                pred_token = output.argmax(1).item()

                if pred_token == self.target_vocab["<END>"]:
                    break

                char = self.reverse_target_vocab.get(pred_token, "")
                if char not in ["<PAD>", "<START>", "<END>"]:
                    decoded.append(char)

                input_token = torch.tensor([[pred_token]]).to(self.device)

        return "".join(decoded)

    def evaluate_on_test_set(self, num_examples=10):
        """Evaluate model on test set"""
        print(f"\n{'='*80}")
        print("EVALUATION ON TEST SET")
        print(f"{'='*80}\n")

        for i in range(min(num_examples, len(self.test_input))):
            input_text = self.test_input[i]
            target_text = self.test_target[i]
            predicted_text = self.normalize_text(input_text)

            print(f"Example {i+1}:")
            print(f"  Input (old):  {input_text}")
            print(f"  Target:       {target_text}")
            print(f"  Predicted:    {predicted_text}")
            print()

    def save_model(self, filepath="spelling_normalizer_pytorch.pt"):
        """Save model and metadata"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_vocab": self.input_vocab,
                "target_vocab": self.target_vocab,
                "reverse_target_vocab": self.reverse_target_vocab,
                "max_len": self.max_len,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
            },
            filepath,
        )
        print(f"\nModel saved to {filepath}")


def main():
    """Main training script"""
    CSV_PATH = "data/processed/aligned_old_to_modern_combined.csv"

    # Optimized for word-level aligned data with larger dataset (2468 pairs)
    # Larger model for better capacity
    # embedding_dim: 128, hidden_dim: 256, n_layers: 2 for better learning
    normalizer = SpellingNormalizer(embedding_dim=128, hidden_dim=256, n_layers=2)
    normalizer.prepare_data(CSV_PATH, augment=True)
    normalizer.build_model()

    # Optimized training for larger dataset
    # Higher learning rate for faster convergence with more data
    # More epochs to fully leverage larger dataset
    normalizer.train(
        epochs=60,
        batch_size=32,  # Larger batch for better gradient estimates
        lr=0.002,  # Higher LR for faster convergence with more data
        accumulation_steps=2,
        warmup_epochs=5,
    )
    normalizer.evaluate_on_test_set(num_examples=15)
    normalizer.save_model()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
