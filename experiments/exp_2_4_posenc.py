"""
W&B Report 2.4 - Sinusoidal vs learned positional encoding.

Train two otherwise-identical models and compare validation BLEU.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import run_training_experiment, DEFAULT_CONFIG


def main():
    run_training_experiment(
        cfg={**DEFAULT_CONFIG, "epochs": 12, "pos_enc": "sinusoidal"},
        run_name="2.4_sinusoidal",
        ckpt_path="ckpt_2_4_sin.pt",
        extra_log={"pos_enc": "sinusoidal"},
    )
    run_training_experiment(
        cfg={**DEFAULT_CONFIG, "epochs": 12, "pos_enc": "learned"},
        run_name="2.4_learned",
        ckpt_path="ckpt_2_4_learned.pt",
        extra_log={"pos_enc": "learned"},
    )


if __name__ == "__main__":
    main()
