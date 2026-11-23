"""Train the pulse classifier on FINETUNE_SHARD_DIR, optionally starting from a pretrained encoder.
Run from the project root: python transformer/finetune.py

Loss is softmax cross-entropy (mvts task "classification").
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from dataset import PulseDataset, collate_supervised
from model import TSTransformerClassifier
from training import backward_step, encoder_args, fit, l2_penalty, load_splits, seed_epoch


def classification_metrics(predicted, targets, n_classes):
    """Accuracy, and precision averaged over classes weighted by their support,
    using only classes that were predicted at least once (mvts Analyzer.get_avg_prec_recall)."""

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)

    predicted_per_class = confusion.sum(axis=0)
    support = confusion.sum(axis=1)
    included = predicted_per_class > 0
    accuracy = float(np.trace(confusion) / len(targets))

    if support[included].sum() == 0:  # only classes without true samples were predicted
        return accuracy, 0.0

    weights = support[included] / support[included].sum()
    precision = np.diag(confusion)[included] / predicted_per_class[included]

    return accuracy, float(np.dot(weights, precision))


def make_run_epoch(n_classes):

    def run_epoch(model, loader, device, optimizer=None):
        """Train when an optimizer is given, otherwise evaluate. Returns loss, accuracy, precision."""

        training = optimizer is not None
        model.train(training)
        total_loss, predicted, targets = 0.0, [], []

        for x, labels, padding_masks in tqdm(loader, leave=False, desc="train" if training else "val"):

            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            padding_masks = padding_masks.to(device, non_blocking=True)

            logits = model(x, padding_masks)
            loss = F.cross_entropy(logits, labels)

            if training:
                penalty = l2_penalty(model)
                backward_step(model, optimizer, loss if penalty is None else loss + penalty)

            total_loss += loss.item() * len(labels)
            predicted.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())

        predicted, targets = np.concatenate(predicted), np.concatenate(targets)
        accuracy, precision = classification_metrics(predicted, targets, n_classes)

        return {"loss": float(total_loss / len(targets)), "accuracy": accuracy, "precision": precision}

    return run_epoch


def load_pretrained_encoder(model, model_args, path):
    """Copy every weight except the output head from a pretraining best_model.pt (mvts change_output)."""

    state = torch.load(path, map_location="cpu")
    pretrained_args = {k: v for k, v in state["model_args"].items() if k != "freeze"}
    current_args = {k: model_args[k] for k in pretrained_args}

    if state.get("phase") != "pretrain":
        raise ValueError(f"PRETRAINED_ENCODER_PATH {path} is not a pretraining checkpoint.")

    if pretrained_args != current_args or state["feature_columns"] != list(config.FEATURE_COLUMNS):
        raise ValueError(
            f"PRETRAINED_ENCODER_PATH {path} was trained with model settings {pretrained_args} and "
            f"features {state['feature_columns']}, but the current config has {current_args} and "
            f"{list(config.FEATURE_COLUMNS)}."
        )

    weights = {k: v for k, v in state["model_state"].items() if not k.startswith("output_")}
    result = model.load_state_dict(weights, strict=False)

    if result.unexpected_keys or any(not k.startswith("output_") for k in result.missing_keys):
        raise RuntimeError(f"Encoder weights didn't match: {result}")

    return state["epoch"]


def main():

    if config.CLASSIFIER_HEAD not in ("flatten", "mean", "attention"):
        raise ValueError(f"CLASSIFIER_HEAD must be flatten, mean or attention, got {config.CLASSIFIER_HEAD!r}.")

    if config.KEY_METRIC not in ("loss", "accuracy", "precision"):
        raise ValueError(f"KEY_METRIC must be loss, accuracy or precision, got {config.KEY_METRIC!r}.")

    if config.FREEZE_ENCODER and not config.PRETRAINED_ENCODER_PATH:
        raise ValueError("FREEZE_ENCODER needs PRETRAINED_ENCODER_PATH.")

    if config.SEED is not None:
        seed_epoch(config.SEED, 0, torch.Generator())  # model initialization

    with open(Path(config.FINETUNE_SHARD_DIR) / "class_mapping.json", encoding="utf-8") as file:
        class_mapping = json.load(file)

    datasets = load_splits(config.FINETUNE_SHARD_DIR, PulseDataset)

    model_args = {
        **encoder_args(freeze=config.FREEZE_ENCODER),
        "num_classes": len(class_mapping),
        "head": config.CLASSIFIER_HEAD,
    }
    model = TSTransformerClassifier(**model_args)

    log_lines = [
        f"FINETUNE_SHARD_DIR: {config.FINETUNE_SHARD_DIR}",
        f"FINETUNE_EPOCHS: {config.FINETUNE_EPOCHS}",
        f"CLASSIFIER_HEAD: {config.CLASSIFIER_HEAD}",
        f"PRETRAINED_ENCODER_PATH: {config.PRETRAINED_ENCODER_PATH}",
        f"FREEZE_ENCODER: {config.FREEZE_ENCODER}",
        f"classes: {len(class_mapping)}",
    ]

    if config.PRETRAINED_ENCODER_PATH and not config.RESUME_FROM:
        epoch = load_pretrained_encoder(model, model_args, config.PRETRAINED_ENCODER_PATH)
        log_lines.append(f"loaded pretrained encoder from epoch {epoch}")

    if config.FREEZE_ENCODER:  # mvts --freeze: train the output head only
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("output_")

    fit(
        "finetune", model, model_args, datasets, collate_supervised, make_run_epoch(len(class_mapping)),
        config.FINETUNE_EPOCHS, config.KEY_METRIC,
        extra={"class_mapping": class_mapping, "pretrained_encoder_path": config.PRETRAINED_ENCODER_PATH},
        log_lines=log_lines,
    )


if __name__ == "__main__":
    main()
