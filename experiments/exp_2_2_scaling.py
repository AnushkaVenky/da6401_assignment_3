"""
W&B Report 2.2 - Ablation of the 1 / sqrt(d_k) scaling factor.

Train two models that differ only in whether attention scores are
divided by sqrt(d_k).  Log gradient norms of Q / K projections for the
first 1000 steps so we can demonstrate the "vanishing gradient" effect
discussed in section 3.2.1 of the paper.
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import run_training_experiment, DEFAULT_CONFIG


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    # With scaling (canonical)
    m1 = run_training_experiment(
        cfg={**DEFAULT_CONFIG, "epochs": 5, "attn_scale": True},
        run_name="2.2_scaled",
        ckpt_path="ckpt_2_2_scaled.pt",
        extra_log={"variant": "scaled"},
        log_grad_norms=True,
        grad_norm_steps=1000,
    )
    with open(os.path.join(out_dir, "2_2_scaled_grad_norms.json"), "w") as f:
        json.dump(m1["grad_norm_history"], f)

    # Without scaling
    m2 = run_training_experiment(
        cfg={**DEFAULT_CONFIG, "epochs": 5, "attn_scale": False},
        run_name="2.2_unscaled",
        ckpt_path="ckpt_2_2_unscaled.pt",
        extra_log={"variant": "unscaled"},
        log_grad_norms=True,
        grad_norm_steps=1000,
    )
    with open(os.path.join(out_dir, "2_2_unscaled_grad_norms.json"), "w") as f:
        json.dump(m2["grad_norm_history"], f)


if __name__ == "__main__":
    main()
