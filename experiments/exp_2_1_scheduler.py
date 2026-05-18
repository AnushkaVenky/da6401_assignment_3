"""
W&B Report 2.1 - Necessity of the Noam Scheduler.

Train one model with the Noam warm-up schedule and a second with a
constant learning rate of 1e-4.  Both runs log train_loss and val_loss
so the W&B UI can overlay the curves.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import run_training_experiment, DEFAULT_CONFIG


def main():
    # Noam (default)
    run_training_experiment(
        cfg={**DEFAULT_CONFIG, "epochs": 15},
        run_name="2.1_noam_scheduler",
        ckpt_path="ckpt_2_1_noam.pt",
        extra_log={"variant": "noam"},
    )

    # Constant LR = 1e-4
    run_training_experiment(
        cfg={**DEFAULT_CONFIG, "epochs": 15, "fixed_lr": 1e-4},
        run_name="2.1_fixed_lr_1e-4",
        ckpt_path="ckpt_2_1_fixed.pt",
        extra_log={"variant": "fixed_lr"},
    )


if __name__ == "__main__":
    main()
