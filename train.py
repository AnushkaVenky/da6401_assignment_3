"""
train.py - Training pipeline, inference and evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (signatures preserved):
  greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol) -> [1, out_len]
  evaluate_bleu(model, test_dataloader, tgt_vocab, device) -> float
  save_checkpoint(model, optimizer, scheduler, epoch, path) -> None
  load_checkpoint(path, model, optimizer, scheduler)        -> int
"""

import argparse
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler


# ============================================================
# Label-smoothing loss
# ============================================================

class LabelSmoothingLoss(nn.Module):
    """
    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)
    where the pad index receives zero probability.

    Loss is the KL divergence between log-softmax(logits) and the
    smoothed targets.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        # KL accepts log-probs as input, probs as target.
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [N, V]  target: [N]
        if self.smoothing == 0.0:
            return F.cross_entropy(logits, target, ignore_index=self.pad_idx)

        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.full_like(
                log_probs, self.smoothing / (self.vocab_size - 2)
            )                                                # -2: exclude true class + pad
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[:, self.pad_idx] = 0.0
            mask = (target == self.pad_idx).unsqueeze(1)
            true_dist.masked_fill_(mask, 0.0)

        # Average over non-pad tokens.
        n_tokens = (target != self.pad_idx).sum().clamp(min=1)
        loss = -(true_dist * log_probs).sum() / n_tokens
        return loss


# ============================================================
# Training / eval loop
# ============================================================

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    log_grad_norms: bool = False,
    grad_norm_layer: int = 0,
    grad_norm_history: Optional[list] = None,
    max_grad_norm_steps: Optional[int] = None,
) -> float:
    """One pass over ``data_iter``. Returns mean per-token loss."""
    model.train(is_train)

    total_loss = 0.0
    total_tokens = 0
    pad_idx = model.pad_idx

    for batch_idx, (src, tgt) in enumerate(data_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher-forcing: shift target.
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
        )

        if is_train:
            optimizer.zero_grad()
            loss.backward()

            if log_grad_norms and grad_norm_history is not None:
                # Grad norms of the first encoder layer's W_q / W_k.
                enc0 = model.encoder.layers[grad_norm_layer]
                wq = enc0.self_attn.W_q.weight.grad
                wk = enc0.self_attn.W_k.weight.grad
                grad_norm_history.append({
                    "step": scheduler.last_epoch + 1 if scheduler is not None
                            else batch_idx,
                    "W_q_grad_norm": float(wq.norm().item()) if wq is not None else 0.0,
                    "W_k_grad_norm": float(wk.norm().item()) if wk is not None else 0.0,
                })
                if max_grad_norm_steps is not None and len(grad_norm_history) >= max_grad_norm_steps:
                    log_grad_norms = False  # stop logging

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        n_tokens = (tgt_out != pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(1, total_tokens)


# ============================================================
# Greedy decoding
# ============================================================

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Token-by-token greedy decoding.  Returns [1, out_len]."""
    model.eval()
    memory = model.encode(src, src_mask)
    ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, model.pad_idx)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == end_symbol:
            break

    return ys


# ============================================================
# BLEU evaluation
# ============================================================

def _decode_tokens(ids, vocab, eos_idx: int, pad_idx: int, sos_idx: int):
    out = []
    for i in ids:
        i = int(i)
        if i == eos_idx:
            break
        if i in (pad_idx, sos_idx):
            continue
        if hasattr(vocab, "itos"):
            tok = vocab.itos.get(i, "<unk>")
        elif hasattr(vocab, "lookup_token"):
            tok = vocab.lookup_token(i)
        else:
            tok = "<unk>"
        out.append(tok)
    return out


def _corpus_bleu(hyps, refs) -> float:
    """Corpus BLEU on the 0-100 scale; tries sacrebleu, then nltk."""
    try:
        import sacrebleu
        hyp_strs = [" ".join(h) for h in hyps]
        ref_strs = [[" ".join(r) for r in refs]]
        return float(sacrebleu.corpus_bleu(hyp_strs, ref_strs).score)
    except Exception:
        pass

    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        # nltk expects each reference wrapped in a list.
        refs_wrapped = [[r] for r in refs]
        return 100.0 * corpus_bleu(refs_wrapped, hyps, smoothing_function=smooth)
    except Exception:
        return 0.0


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """Corpus-level BLEU on the 0-100 scale."""
    model.eval()

    sos_idx = tgt_vocab["<sos>"]
    eos_idx = tgt_vocab["<eos>"]
    pad_idx = tgt_vocab["<pad>"]

    hyps, refs = [], []
    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            for i in range(src.size(0)):
                src_i = src[i : i + 1]
                src_mask = make_src_mask(src_i, model.pad_idx)
                out_ids = greedy_decode(
                    model, src_i, src_mask, max_len,
                    sos_idx, eos_idx, device,
                )[0].tolist()
                hyp = _decode_tokens(out_ids, tgt_vocab, eos_idx, pad_idx, sos_idx)
                ref = _decode_tokens(tgt[i].tolist(), tgt_vocab, eos_idx, pad_idx, sos_idx)
                hyps.append(hyp)
                refs.append(ref)
    return _corpus_bleu(hyps, refs)


# ============================================================
# Checkpoint utilities
# ============================================================

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    model_config = {
        "src_vocab_size": model.src_vocab_size,
        "tgt_vocab_size": model.tgt_vocab_size,
        "d_model": model.d_model,
        "N": len(model.encoder.layers),
        "num_heads": model.encoder.layers[0].self_attn.num_heads,
        "d_ff": model.encoder.layers[0].ffn.linear1.out_features,
        "dropout": model.encoder.layers[0].dropout1.p,
        "pad_idx": model.pad_idx,
    }
    # Embed vocabs so a single gdown download is enough for the
    # autograder to reconstruct everything.
    src_vocab = getattr(model, "src_vocab", None)
    tgt_vocab = getattr(model, "tgt_vocab", None)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model_config,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt.get("epoch", 0))


# ============================================================
# Default hyper-parameters
# ============================================================

DEFAULT_CONFIG = {
    "d_model": 256,        # smaller than 512 - fits comfortably on one GPU
    "N": 3,
    "num_heads": 8,
    "d_ff": 1024,
    "dropout": 0.1,
    "batch_size": 128,
    "epochs": 15,
    "warmup_steps": 4000,
    "smoothing": 0.1,
    "min_freq": 2,
    "max_len": 100,
    "pos_enc": "sinusoidal",
    "attn_scale": True,
    "fixed_lr": None,       # None = use Noam; float = constant lr
    "pad_idx": 1,
}


def build_model(cfg, src_vocab_size, tgt_vocab_size, device):
    # checkpoint_path=None during training - we are CREATING the
    # checkpoint, not loading one.
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg["d_model"],
        N=cfg["N"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"] + 50,
        pad_idx=cfg["pad_idx"],
        pos_enc=cfg["pos_enc"],
        attn_scale=cfg["attn_scale"],
        checkpoint_path=None,
    ).to(device)
    return model


# ============================================================
# Experiment runner
# ============================================================

def run_training_experiment(
    cfg: Optional[dict] = None,
    wandb_project: str = "da6401-a3",
    run_name: Optional[str] = None,
    extra_log: Optional[dict] = None,
    ckpt_path: str = "checkpoint.pt",
    log_grad_norms: bool = False,
    grad_norm_steps: int = 1000,
) -> dict:
    """
    Train + evaluate a single run.  Returns a metrics dict.
    """
    from dataset import get_dataloaders

    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- W&B (optional, never required) ----
    use_wandb = False
    try:
        import wandb
        wandb.init(project=wandb_project, name=run_name, config=cfg,
                   reinit=True)
        use_wandb = True
    except Exception as e:
        print(f"[wandb disabled] {e}")

    print(f"[device] {device}")
    print(f"[config] {cfg}")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=cfg["batch_size"],
        min_freq=cfg["min_freq"],
        max_len=cfg["max_len"],
    )
    print(f"[data] src_vocab={len(src_vocab)}  tgt_vocab={len(tgt_vocab)}")

    model = build_model(cfg, len(src_vocab), len(tgt_vocab), device)

    # Stash vocabs on the model so save_checkpoint() can embed them.
    # `Transformer.__init__` will rehydrate these (plus the spaCy
    # tokenizer) on autograder load.
    model.src_vocab = src_vocab.stoi
    model.tgt_vocab = tgt_vocab.stoi

    # Optimiser / scheduler / loss.
    base_lr = cfg["fixed_lr"] if cfg["fixed_lr"] is not None else 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr,
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = None
    if cfg["fixed_lr"] is None:
        scheduler = NoamScheduler(optimizer, d_model=cfg["d_model"],
                                  warmup_steps=cfg["warmup_steps"])
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=cfg["pad_idx"],
        smoothing=cfg["smoothing"],
    )

    grad_norm_history: list = [] if log_grad_norms else None

    best_val = float("inf")
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
            log_grad_norms=log_grad_norms,
            grad_norm_history=grad_norm_history,
            max_grad_norm_steps=grad_norm_steps,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )
        val_ppl = math.exp(min(val_loss, 20))
        dt = time.time() - t0

        print(f"[epoch {epoch:02d}] train={train_loss:.4f}  "
              f"val={val_loss:.4f}  ppl={val_ppl:.2f}  ({dt:.1f}s)")

        if use_wandb:
            log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if extra_log:
                log.update(extra_log)
            wandb.log(log)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path)

    # ---- Final BLEU on the test set ----
    load_checkpoint(ckpt_path, model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device,
                         max_len=cfg["max_len"])
    print(f"[test] corpus-BLEU = {bleu:.2f}")

    metrics = {
        "best_val_loss": best_val,
        "test_bleu": bleu,
        "grad_norm_history": grad_norm_history,
    }
    if use_wandb:
        wandb.log({"test_bleu": bleu})
        wandb.finish()
    return metrics


# ============================================================
# CLI
# ============================================================

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--d_model", type=int, default=DEFAULT_CONFIG["d_model"])
    p.add_argument("--N", type=int, default=DEFAULT_CONFIG["N"])
    p.add_argument("--num_heads", type=int, default=DEFAULT_CONFIG["num_heads"])
    p.add_argument("--d_ff", type=int, default=DEFAULT_CONFIG["d_ff"])
    p.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--smoothing", type=float, default=DEFAULT_CONFIG["smoothing"])
    p.add_argument("--warmup_steps", type=int, default=DEFAULT_CONFIG["warmup_steps"])
    p.add_argument("--fixed_lr", type=float, default=None,
                   help="If set, use a constant LR instead of Noam.")
    p.add_argument("--pos_enc", type=str, default="sinusoidal",
                   choices=["sinusoidal", "learned"])
    p.add_argument("--attn_scale", type=lambda s: s.lower() != "false",
                   default=True, help="Apply 1/sqrt(d_k) scaling.")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--ckpt_path", type=str, default="checkpoint.pt")
    p.add_argument("--log_grad_norms", action="store_true")
    p.add_argument("--grad_norm_steps", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = {**DEFAULT_CONFIG, **vars(args)}
    cfg.pop("run_name", None)
    cfg.pop("ckpt_path", None)
    cfg.pop("log_grad_norms", None)
    cfg.pop("grad_norm_steps", None)
    run_training_experiment(
        cfg=cfg,
        run_name=args.run_name,
        ckpt_path=args.ckpt_path,
        log_grad_norms=args.log_grad_norms,
        grad_norm_steps=args.grad_norm_steps,
    )
