"""Train a seq2seq Transformer for Early Modern English modernization."""

import argparse
import math
import os
import random
import sys
from collections import Counter
from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(".")
from scripts.utils import compute_metrics, load_parallel_data  # noqa: E402


class CharVocab:
    """Character-level vocabulary with BOS/EOS/PAD tokens."""

    def __init__(self, texts: Sequence[str]):
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        counts = Counter("".join(texts))

        self.itos = list(specials) + sorted(counts)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    def encode(self, text: str) -> List[int]:
        ids = [self.bos_id]
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in text)
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        chars = []
        for idx in ids:
            if idx in {self.pad_id, self.bos_id}:
                continue
            if idx == self.eos_id:
                break
            chars.append(self.itos[idx] if idx < len(self.itos) else "")
        return "".join(chars)


class ParallelDataset(Dataset):
    """Early (src) / modern (tgt) pairs."""

    def __init__(
        self, pairs: Sequence[Tuple[str, str]], src_vocab: CharVocab, tgt_vocab: CharVocab, max_length: int
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length

        self.examples = []
        for early, modern in pairs:
            if max_length and (len(early) + 2 > max_length or len(modern) + 2 > max_length):
                continue
            self.examples.append((early, modern))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        early, modern = self.examples[idx]
        return self.src_vocab.encode(early), self.tgt_vocab.encode(modern)


def collate_batch(
    batch: Sequence[Tuple[List[int], List[int]]],
    src_pad_id: int,
    tgt_pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length examples."""
    src_seqs, tgt_seqs = zip(*batch)
    src_max = max(len(seq) for seq in src_seqs)
    tgt_max = max(len(seq) for seq in tgt_seqs)

    src_batch = torch.full((len(batch), src_max), src_pad_id, dtype=torch.long)
    tgt_batch = torch.full((len(batch), tgt_max), tgt_pad_id, dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        src_batch[i, : len(src)] = torch.tensor(src, dtype=torch.long)
        tgt_batch[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)

    return src_batch, tgt_batch


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """Minimal Transformer encoder-decoder."""

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pad_id = pad_id
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        # enable_nested_tensor=False avoids ops not implemented on MPS.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.generator(output)


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)
    return mask


def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
    max_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    device: torch.device,
) -> List[int]:
    model.eval()
    ys = torch.tensor([[bos_id]], device=device)

    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        tgt_padding_mask = ys == pad_id

        out = model(src, ys, src_key_padding_mask, tgt_padding_mask, tgt_mask)
        next_token = out[:, -1, :].argmax(dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == eos_id:
            break

    return ys.squeeze(0).tolist()


def train_epoch(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    pad_id: int,
) -> float:
    model.train()
    total_loss = 0.0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        targets = tgt[:, 1:]

        src_key_padding_mask = src == pad_id
        tgt_key_padding_mask = tgt_input == pad_id
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        logits = model(src, tgt_input, src_key_padding_mask, tgt_key_padding_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    src_vocab: CharVocab,
    tgt_vocab: CharVocab,
    device: torch.device,
    max_len: int,
) -> dict:
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            src_padding_mask = src == src_vocab.pad_id

            for i in range(src.size(0)):
                src_seq = src[i].unsqueeze(0)
                src_mask = src_padding_mask[i].unsqueeze(0)
                pred_ids = greedy_decode(
                    model,
                    src_seq,
                    src_mask,
                    max_len=max_len,
                    bos_id=tgt_vocab.bos_id,
                    eos_id=tgt_vocab.eos_id,
                    pad_id=tgt_vocab.pad_id,
                    device=device,
                )
                pred_text = tgt_vocab.decode(pred_ids)
                ref_text = tgt_vocab.decode(tgt[i].tolist())

                predictions.append(pred_text)
                references.append(ref_text)

    return compute_metrics(predictions, references)


def build_dataloaders(
    train_pairs: Sequence[Tuple[str, str]],
    dev_pairs: Sequence[Tuple[str, str]],
    batch_size: int,
    max_length: int,
):
    src_vocab = CharVocab([src for src, _ in train_pairs])
    tgt_vocab = CharVocab([tgt for _, tgt in train_pairs])

    train_dataset = ParallelDataset(train_pairs, src_vocab, tgt_vocab, max_length=max_length)
    dev_dataset = ParallelDataset(dev_pairs, src_vocab, tgt_vocab, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, src_vocab.pad_id, tgt_vocab.pad_id),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, src_vocab.pad_id, tgt_vocab.pad_id),
    )

    return train_loader, dev_loader, src_vocab, tgt_vocab


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(use_mps_fallback: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if use_mps_fallback and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("Enabled PYTORCH_ENABLE_MPS_FALLBACK=1 to avoid missing ops on MPS.")
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer on the modernization task.")
    parser.add_argument("--data", default="synthetic", choices=["synthetic", "processed"], help="Dataset split type.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--ff", type=int, default=512, help="Feed-forward hidden size.")
    parser.add_argument("--layers", type=int, default=3, help="Number of encoder/decoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=200, help="Max sequence length (including BOS/EOS).")
    parser.add_argument("--save-path", default=None, help="Optional path to save the trained model.")
    parser.add_argument(
        "--no-mps-fallback",
        action="store_true",
        help="Disable automatic PYTORCH_ENABLE_MPS_FALLBACK=1 when using MPS.",
    )
    args = parser.parse_args()

    set_seed()

    train_file_early = f"data/{args.data}/train_early.txt"
    train_file_modern = f"data/{args.data}/train_modern.txt"
    dev_file_early = f"data/{args.data}/dev_early.txt"
    dev_file_modern = f"data/{args.data}/dev_modern.txt"

    train_raw = load_parallel_data(train_file_early, train_file_modern)
    dev_raw = load_parallel_data(dev_file_early, dev_file_modern)

    # load_parallel_data returns (modern, early)
    train_pairs = [(early, modern) for modern, early in train_raw]
    dev_pairs = [(early, modern) for modern, early in dev_raw]

    train_loader, dev_loader, src_vocab, tgt_vocab = build_dataloaders(
        train_pairs, dev_pairs, batch_size=args.batch_size, max_length=args.max_length
    )

    device = get_device(use_mps_fallback=not args.no_mps_fallback)
    print(f"Using device: {device}")

    model = Seq2SeqTransformer(
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.ff,
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        dropout=args.dropout,
        pad_id=src_vocab.pad_id,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_cer = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, pad_id=src_vocab.pad_id)
        metrics = evaluate(model, dev_loader, src_vocab, tgt_vocab, device, max_len=args.max_length)

        print(
            f"Epoch {epoch:02d} | Train loss {train_loss:.4f} | "
            f"Dev EM {metrics['exact_match_accuracy']:.2%} | Dev CER {metrics['character_error_rate']:.4f}"
        )

        if metrics["character_error_rate"] < best_cer:
            best_cer = metrics["character_error_rate"]
            best_state = {
                "model_state": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "metrics": metrics,
                "config": vars(args),
            }

    if best_state:
        os.makedirs("models", exist_ok=True)
        save_path = args.save_path or f"models/transformer_{args.data}.pt"
        torch.save(best_state, save_path)
        print(f"Best model saved to {save_path} (CER={best_cer:.4f})")
    else:
        print("No model state to save; check training data.")


if __name__ == "__main__":
    main()
