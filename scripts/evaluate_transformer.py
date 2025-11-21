"""Evaluate a trained Transformer modernization model."""

import argparse
import sys
import os
from pathlib import Path
from typing import Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(".")
from scripts.train_transformer import (  # noqa: E402
    CharVocab,
    ParallelDataset,
    Seq2SeqTransformer,
    collate_batch,
    evaluate,
    get_device,
    greedy_decode,
)
from scripts.utils import load_parallel_data, compute_metrics  # noqa: E402


class EvalDataset(Dataset):
    """Dataset wrapper to reuse saved vocabs."""

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        src_vocab: CharVocab,
        tgt_vocab: CharVocab,
        max_length: int,
    ):
        self.inner = ParallelDataset(pairs, src_vocab, tgt_vocab, max_length)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self.inner[idx]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved Transformer model.")
    parser.add_argument("--checkpoint", default="models/transformer_synthetic.pt")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--data", default=None, help="Dataset type: synthetic or processed (default: from checkpoint).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None, help="Override max sequence length.")
    parser.add_argument(
        "--no-mps-fallback",
        action="store_true",
        help="Disable automatic PYTORCH_ENABLE_MPS_FALLBACK=1 when using MPS.",
    )
    parser.add_argument("--examples", type=int, default=5, help="Number of examples to print.")
    args = parser.parse_args()

    # Allow CharVocab objects inside checkpoints (trusted source).
    try:
        torch.serialization.add_safe_globals([CharVocab])
    except Exception:
        pass

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        # For older torch versions without weights_only
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})

    data_type = args.data or cfg.get("data", "synthetic")
    batch_size = args.batch_size or cfg.get("batch_size", 64)
    max_len = args.max_length or cfg.get("max_length", 200)

    src_vocab = ckpt.get("src_vocab")
    tgt_vocab = ckpt.get("tgt_vocab")
    if src_vocab is None or tgt_vocab is None:
        # Fallback: rebuild vocab from training data if missing in checkpoint.
        train_raw = load_parallel_data(f"data/{data_type}/train_early.txt", f"data/{data_type}/train_modern.txt")
        train_pairs = [(e, m) for m, e in train_raw]
        src_vocab = CharVocab([src for src, _ in train_pairs])
        tgt_vocab = CharVocab([tgt for _, tgt in train_pairs])

    file_prefix = f"data/{data_type}/{args.split}"
    raw_pairs = load_parallel_data(f"{file_prefix}_early.txt", f"{file_prefix}_modern.txt")
    pairs = [(e, m) for m, e in raw_pairs]

    dataset = EvalDataset(pairs, src_vocab, tgt_vocab, max_length=max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, src_vocab.pad_id, tgt_vocab.pad_id),
    )

    device = get_device(use_mps_fallback=not args.no_mps_fallback)
    print(f"Using device: {device}")

    model = Seq2SeqTransformer(
        num_encoder_layers=cfg.get("layers", 3),
        num_decoder_layers=cfg.get("layers", 3),
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 4),
        dim_feedforward=cfg.get("ff", 512),
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        dropout=cfg.get("dropout", 0.1),
        pad_id=src_vocab.pad_id,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # Decode and collect predictions
    decoded_examples = []
    references = []
    predictions = []
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = src == src_vocab.pad_id
            for i in range(src.size(0)):
                pred_ids = greedy_decode(
                    model,
                    src[i].unsqueeze(0),
                    src_mask[i].unsqueeze(0),
                    max_len=max_len,
                    bos_id=tgt_vocab.bos_id,
                    eos_id=tgt_vocab.eos_id,
                    pad_id=tgt_vocab.pad_id,
                    device=device,
                )
                early = src_vocab.decode(src[i].tolist())
                pred = tgt_vocab.decode(pred_ids)
                gold = tgt_vocab.decode(tgt[i].tolist())
                predictions.append(pred)
                references.append(gold)
                decoded_examples.append((early, pred, gold))

    metrics = compute_metrics(predictions, references)
    print(f"Results on {args.split} ({data_type}):")
    print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
    print(f"  Character Error Rate: {metrics['character_error_rate']:.4f}")

    # Show a few examples to stdout
    print("\nExamples:")
    for idx, (early, pred, gold) in enumerate(decoded_examples[: args.examples]):
        print(f"\nExample {idx + 1}:")
        print(f"  Early: {early}")
        print(f"  Pred : {pred}")
        print(f"  Gold : {gold}")

    # Save full results to file (matching existing format)
    os.makedirs("results", exist_ok=True)
    model_label = Path(args.checkpoint).stem
    result_path = Path("results") / f"{model_label}_on_{data_type}_{args.split}.txt"
    with open(result_path, "w") as f:
        f.write(f"Model: {model_label}\n")
        f.write(f"Test Set: {data_type} ({args.split})\n")
        f.write(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}\n")
        f.write(f"Character Error Rate: {metrics['character_error_rate']:.4f}\n")
        f.write("\nExamples:\n")
        f.write("=" * 60 + "\n")
        for idx, (early, pred, gold) in enumerate(decoded_examples):
            match = "✓" if pred == gold else "✗"
            f.write(f"\n{match} Example {idx + 1}:\n")
            f.write(f"  Early:      {early}\n")
            f.write(f"  Predicted:  {pred}\n")
            f.write(f"  Reference:  {gold}\n")

    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
