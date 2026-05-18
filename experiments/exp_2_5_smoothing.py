"""
W&B Report 2.5 - Decoder sensitivity to label smoothing.

Train two models with eps=0.1 vs eps=0.0 and (post-hoc) plot the
softmax probability assigned to the *correct* target token on the
validation set.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from train import (
    run_training_experiment, DEFAULT_CONFIG, build_model, load_checkpoint,
)
from model import make_src_mask, make_tgt_mask
from dataset import get_dataloaders


def confidence_histogram(model, loader, device, pad_idx, max_batches=20):
    model.eval()
    probs = []
    with torch.no_grad():
        for i, (src, tgt) in enumerate(loader):
            if i >= max_batches:
                break
            src = src.to(device); tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]; tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx)
            logits = model(src, tgt_in, src_mask, tgt_mask)
            p = F.softmax(logits, dim=-1)              # [B, T, V]
            gathered = p.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
            mask = (tgt_out != pad_idx)
            probs.extend(gathered[mask].cpu().tolist())
    return probs


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Train with smoothing ---
    run_training_experiment(
        cfg={**DEFAULT_CONFIG, "epochs": 12, "smoothing": 0.1},
        run_name="2.5_smoothing_0.1",
        ckpt_path="ckpt_2_5_smooth.pt",
        extra_log={"smoothing": 0.1},
    )

    # --- Train without smoothing ---
    run_training_experiment(
        cfg={**DEFAULT_CONFIG, "epochs": 12, "smoothing": 0.0},
        run_name="2.5_smoothing_0.0",
        ckpt_path="ckpt_2_5_no_smooth.pt",
        extra_log={"smoothing": 0.0},
    )

    # --- Confidence histograms ---
    _, val_loader, _, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=DEFAULT_CONFIG["batch_size"],
        min_freq=DEFAULT_CONFIG["min_freq"],
        max_len=DEFAULT_CONFIG["max_len"],
    )

    results = {}
    for label, ckpt in [("eps=0.1", "ckpt_2_5_smooth.pt"),
                        ("eps=0.0", "ckpt_2_5_no_smooth.pt")]:
        model = build_model(DEFAULT_CONFIG, len(src_vocab), len(tgt_vocab), device)
        load_checkpoint(ckpt, model)
        results[label] = confidence_histogram(
            model, val_loader, device, DEFAULT_CONFIG["pad_idx"]
        )

    plt.figure(figsize=(8, 5))
    for label, probs in results.items():
        plt.hist(probs, bins=50, alpha=0.5, label=label, density=True)
    plt.xlabel("Softmax prob of correct token")
    plt.ylabel("Density")
    plt.title("Decoder prediction confidence (validation set)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "2_5_confidence_hist.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    try:
        import wandb
        wandb.init(project="da6401-a3", name="2.5_confidence_plot", reinit=True)
        wandb.log({"confidence_hist": wandb.Image(out_path)})
        wandb.finish()
    except Exception as e:
        print(f"[wandb disabled] {e}")


if __name__ == "__main__":
    main()
