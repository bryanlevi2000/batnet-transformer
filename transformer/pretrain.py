"""Self-supervised encoder pretraining on PRETRAIN_SHARD_DIR (mvts task "imputation").
Run from the project root: python transformer/pretrain.py

Random segments of each feature are set to 0 and the encoder learns to reconstruct them.
The loss is the MSE over the masked real values. Labels in the shards are ignored.
"""

import sys
from functools import partial
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from dataset import ImputationDataset, collate_unsupervised
from model import TSTransformerEncoder
from training import backward_step, encoder_args, fit, l2_penalty, load_splits, seed_epoch


def run_epoch(model, loader, device, optimizer=None):
    """Train when an optimizer is given, otherwise evaluate. Returns {"loss": masked MSE per value}."""

    training = optimizer is not None
    model.train(training)
    total_loss = total_values = 0

    for x, targets, target_masks, padding_masks in tqdm(loader, leave=False, desc="train" if training else "val"):

        x, targets = x.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        target_masks = target_masks.to(device, non_blocking=True)
        padding_masks = padding_masks.to(device, non_blocking=True)

        predictions = model(x, padding_masks)
        squared_error = torch.square(predictions - targets)[target_masks]
        loss = squared_error.mean()

        if training:
            penalty = l2_penalty(model)
            backward_step(model, optimizer, loss if penalty is None else loss + penalty)

        total_loss += squared_error.sum().item()
        total_values += squared_error.numel()

    return {"loss": float(total_loss / total_values)}


def main():

    if not 0 < config.MASKING_RATIO < 1:
        raise ValueError(f"MASKING_RATIO must be between 0 and 1, got {config.MASKING_RATIO}.")

    if config.MEAN_MASK_LENGTH < 1:
        raise ValueError(f"MEAN_MASK_LENGTH must be at least 1, got {config.MEAN_MASK_LENGTH}.")

    if config.SEED is not None:
        seed_epoch(config.SEED, 0, torch.Generator())  # model initialization

    datasets = load_splits(
        config.PRETRAIN_SHARD_DIR,
        partial(ImputationDataset, masking_ratio=config.MASKING_RATIO, mean_mask_length=config.MEAN_MASK_LENGTH),
    )

    model_args = encoder_args()
    model = TSTransformerEncoder(**model_args)

    fit(
        "pretrain", model, model_args, datasets, collate_unsupervised, run_epoch,
        config.PRETRAIN_EPOCHS, "loss", extra={},
        log_lines=[
            f"PRETRAIN_SHARD_DIR: {config.PRETRAIN_SHARD_DIR}",
            f"PRETRAIN_EPOCHS: {config.PRETRAIN_EPOCHS}",
            f"MASKING_RATIO: {config.MASKING_RATIO}",
            f"MEAN_MASK_LENGTH: {config.MEAN_MASK_LENGTH}",
        ],
    )


if __name__ == "__main__":
    main()
