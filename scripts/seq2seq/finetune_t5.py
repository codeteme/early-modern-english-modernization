#!/usr/bin/env python3
"""
Fine-tune T5 for Early Modern English to Modern English Spelling Normalization

This script uses a pre-trained T5 model and fine-tunes it on the aligned corpus.
T5 is a sequence-to-sequence transformer model that works well for text transformation tasks.
"""

import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

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


class SpellingNormalizationDataset(Dataset):
    """Dataset for T5 fine-tuning"""

    def __init__(self, old_texts, modern_texts, tokenizer, max_length=128):
        self.old_texts = old_texts
        self.modern_texts = modern_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.old_texts)

    def __getitem__(self, idx):
        old_text = str(self.old_texts[idx])
        modern_text = str(self.modern_texts[idx])

        # Add task prefix for T5
        input_text = f"normalize spelling: {old_text}"

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            modern_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze()
        # Replace padding token id with -100 so it's ignored by loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


class T5SpellingNormalizer:
    """T5-based spelling normalizer"""

    def __init__(self, model_name="t5-small"):
        """
        Initialize with a pre-trained T5 model

        Args:
            model_name: HuggingFace model name (t5-small, t5-base, etc.)
        """
        print(f"Loading pre-trained model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def prepare_data(self, csv_path, test_size=0.15, val_size=0.15, max_samples=None):
        """Load and prepare data"""
        print(f"\nLoading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        if max_samples:
            df = df.head(max_samples)
            print(f"Limited to {max_samples} samples")

        old_texts = df["old"].astype(str).tolist()
        modern_texts = df["modern"].astype(str).tolist()

        print(f"Loaded {len(old_texts)} text pairs")

        # Split data
        train_old, test_old, train_modern, test_modern = train_test_split(
            old_texts, modern_texts, test_size=test_size, random_state=42
        )

        train_old, val_old, train_modern, val_modern = train_test_split(
            train_old, train_modern, test_size=val_size, random_state=42
        )

        print(f"Training samples: {len(train_old)}")
        print(f"Validation samples: {len(val_old)}")
        print(f"Test samples: {len(test_old)}")

        # Create datasets
        self.train_dataset = SpellingNormalizationDataset(
            train_old, train_modern, self.tokenizer
        )
        self.val_dataset = SpellingNormalizationDataset(
            val_old, val_modern, self.tokenizer
        )
        self.test_old = test_old
        self.test_modern = test_modern

    def train(self, epochs=10, batch_size=8, lr=5e-5, warmup_steps=500):
        """Fine-tune the model"""
        print(f"\nFine-tuning for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Warmup steps: {warmup_steps}")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Important for MPS
        )
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, num_workers=0)

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        history = {"train_loss": [], "val_loss": [], "lr": []}

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 5

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Clear MPS cache
                if device.type == "mps":
                    torch.mps.empty_cache()

                train_loss += loss.item()
                current_lr = optimizer.param_groups[0]["lr"]

                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"}
                )

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    val_loss += outputs.loss.item()

                    # Clear MPS cache
                    if device.type == "mps":
                        torch.mps.empty_cache()

            val_loss /= len(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            print(
                f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, LR={current_lr:.6f}"
            )

            # Early stopping
            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                self.model.save_pretrained("best_t5_model")
                self.tokenizer.save_pretrained("best_t5_model")
                patience_counter = 0
                print(f"  → New best model saved! (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        print("\nLoading best model...")
        self.model = T5ForConditionalGeneration.from_pretrained("best_t5_model")
        self.model.to(device)

        self._plot_history(history)
        return history

    def _plot_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Model Loss")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history["lr"])
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("t5_training_history.png", dpi=150)
        print("\nTraining history saved as 't5_training_history.png'")
        plt.close()

    def normalize_text(self, text, max_length=128):
        """Normalize spelling of input text"""
        self.model.eval()

        input_text = f"normalize spelling: {text}"

        input_encoding = self.tokenizer(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = input_encoding["input_ids"].to(device)
        attention_mask = input_encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )

        normalized = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return normalized

    def evaluate_on_test_set(self, num_examples=10):
        """Evaluate model on test set"""
        print(f"\n{'='*80}")
        print("EVALUATION ON TEST SET")
        print(f"{'='*80}\n")

        for i in range(min(num_examples, len(self.test_old))):
            old_text = self.test_old[i]
            modern_text = self.test_modern[i]
            predicted_text = self.normalize_text(old_text)

            print(f"Example {i+1}:")
            print(f"  Input (old):  {old_text}")
            print(f"  Target:       {modern_text}")
            print(f"  Predicted:    {predicted_text}")

            # Simple accuracy check
            match = (
                "✓"
                if predicted_text.strip().lower() == modern_text.strip().lower()
                else "✗"
            )
            print(f"  Match: {match}")
            print()

    def save_model(self, path="t5_spelling_normalizer"):
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"\nModel saved to {path}")


def main():
    """Main training script"""
    CSV_PATH = "../data/processed/aligned_old_to_modern_extracted.csv"  # Using larger dataset (2,468 pairs)

    # Use t5-small for memory efficiency
    # Can upgrade to t5-base for better quality if you have more memory
    normalizer = T5SpellingNormalizer(model_name="t5-small")
    normalizer.prepare_data(CSV_PATH)

    # Fine-tune with conservative settings
    normalizer.train(
        epochs=20, batch_size=8, lr=3e-4, warmup_steps=200  # Higher LR for fine-tuning
    )

    normalizer.evaluate_on_test_set(num_examples=15)
    normalizer.save_model()

    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETE!")
    print("=" * 80)

    # Demo
    print("\n" + "=" * 80)
    print("INTERACTIVE DEMO")
    print("=" * 80 + "\n")

    demo_texts = [
        "Proceed Solinus to procure my fall",
        "Yet this my comfort, when your words are done",
        "I cannot conceiue you",
        "He hath bin out nine yeares",
        "haue",
        "yeares",
        "beene",
    ]

    for text in demo_texts:
        normalized = normalizer.normalize_text(text)
        print(f"Old:    {text}")
        print(f"Modern: {normalized}\n")


if __name__ == "__main__":
    main()
