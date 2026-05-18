"""
W&B Report 2.3 - Attention rollout / head specialisation.

Loads the best checkpoint from the canonical run (Sec. 2.1 Noam) and
visualises the per-head attention weights from the *last encoder layer*
for a single English sentence (we pass an English sentence through the
encoder, as requested by the assignment text).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import get_dataloaders, _load_spacy_models, SOS_IDX, EOS_IDX, UNK_IDX
from model import Transformer, make_src_mask
from train import load_checkpoint, DEFAULT_CONFIG, build_model


SENTENCE_EN = "A man wearing a red hat is walking his dog in the park."


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    # Reload everything that produced the checkpoint.
    _, _, _, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=DEFAULT_CONFIG["batch_size"],
        min_freq=DEFAULT_CONFIG["min_freq"],
        max_len=DEFAULT_CONFIG["max_len"],
    )
    cfg = DEFAULT_CONFIG
    model = build_model(cfg, len(src_vocab), len(tgt_vocab), device)
    load_checkpoint("ckpt_2_1_noam.pt", model)
    model.eval()

    _, nlp_en = _load_spacy_models()
    tokens = [t.text.lower() for t in nlp_en(SENTENCE_EN)]
    # The encoder vocab is German - to still show *something* with no
    # German sentence the assignment asks for the last encoder layer's
    # heatmap for a given English sentence.  We push the English tokens
    # through the (German) vocab and treat OOVs as <unk>; this is fine
    # for visualisation because we only care about the attention shape.
    ids = [SOS_IDX] + [src_vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
    src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = make_src_mask(src, model.pad_idx)

    with torch.no_grad():
        _ = model.encode(src, src_mask)

    last_layer = model.encoder.layers[-1]
    attn = last_layer.self_attn.attn_weights         # [1, h, L, L]
    attn = attn[0].cpu().numpy()
    labels = ["<sos>"] + tokens + ["<eos>"]

    h = attn.shape[0]
    cols = 4
    rows = (h + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i in range(h):
        ax = axes[i]
        im = ax.imshow(attn[i], cmap="viridis")
        ax.set_title(f"Head {i}")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(h, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "2_3_attention_heads.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # Try to log to W&B.
    try:
        import wandb
        wandb.init(project="da6401-a3", name="2.3_attention_heads", reinit=True)
        wandb.log({"attention_heads": wandb.Image(out_path),
                   "sentence": SENTENCE_EN})
        wandb.finish()
    except Exception as e:
        print(f"[wandb disabled] {e}")


if __name__ == "__main__":
    main()
