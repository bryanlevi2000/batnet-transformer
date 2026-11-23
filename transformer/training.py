"""Training loop shared by pretrain.py and finetune.py. Follows mvts_transformer src/main.py."""

import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import load_split

NEG_METRICS = {"loss"}  # metrics where lower is better

MODEL_SETTINGS = [
    "FEATURE_COLUMNS", "MAX_SEQUENCE_LENGTH", "D_MODEL", "NUM_HEADS", "NUM_LAYERS", "DIM_FEEDFORWARD",
    "DROPOUT", "POS_ENCODING", "ACTIVATION", "NORMALIZATION_LAYER",
]
TRAINING_SETTINGS = [
    "BATCH_SIZE", "OPTIMIZER", "LEARNING_RATE", "LR_STEPS", "LR_FACTORS", "L2_REG", "GLOBAL_REG",
    "GRAD_CLIP_NORM", "VAL_INTERVAL", "SEED", "SAVE_EVERY_EPOCH", "RESUME_FROM",
]


def check_settings():
    """Raise ValueError for config values the training code can't use."""

    if len(config.LR_STEPS) != len(config.LR_FACTORS):
        raise ValueError("LR_STEPS and LR_FACTORS must have the same length.")

    if config.OPTIMIZER not in ("Adam", "AdamW", "RAdam"):
        raise ValueError(f"OPTIMIZER must be Adam, AdamW or RAdam, got {config.OPTIMIZER!r}.")

    if config.VAL_INTERVAL < 1:
        raise ValueError("VAL_INTERVAL must be at least 1.")

    if config.D_MODEL % config.NUM_HEADS:
        raise ValueError("D_MODEL must be divisible by NUM_HEADS.")


def seed_epoch(seed, epoch, generator):
    """Reseed every epoch, so a resumed run sees the same shuffle order, masks and dropout."""

    random.seed(seed + epoch)
    np.random.seed(seed + epoch)
    torch.manual_seed(seed + epoch)
    generator.manual_seed(seed + epoch)


def encoder_args(freeze=False):
    """Constructor kwargs shared by TSTransformerEncoder and TSTransformerClassifier."""

    return {
        "feat_dim": len(config.FEATURE_COLUMNS),
        "max_len": config.MAX_SEQUENCE_LENGTH,
        "d_model": config.D_MODEL,
        "n_heads": config.NUM_HEADS,
        "num_layers": config.NUM_LAYERS,
        "dim_feedforward": config.DIM_FEEDFORWARD,
        "dropout": config.DROPOUT,
        "pos_encoding": config.POS_ENCODING,
        "activation": config.ACTIVATION,
        "norm": config.NORMALIZATION_LAYER,
        "freeze": freeze,
    }


def load_splits(shard_dir, make_dataset):
    """Load the train and validation splits of shard_dir into RAM once."""

    datasets = {}

    for split in ("train", "validation"):

        features, offsets, labels, _ = load_split(
            shard_dir, split, config.FEATURE_COLUMNS, config.MAX_SEQUENCE_LENGTH
        )

        if len(labels) == 0:
            raise RuntimeError(
                f"The '{split}' split in {shard_dir} has no samples. "
                "Training needs train and validation splits (build shards with VAL_RATIO > 0)."
            )

        datasets[split] = make_dataset(features, offsets, labels)

    return datasets


def make_optimizer(model):
    """mvts: with GLOBAL_REG the L2 penalty is weight decay; otherwise it's added to the loss
    for the output layer only (see l2_penalty)."""

    optim_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "RAdam": torch.optim.RAdam}
    weight_decay = config.L2_REG if config.GLOBAL_REG else 0

    return optim_class[config.OPTIMIZER](model.parameters(), lr=config.LEARNING_RATE, weight_decay=weight_decay)


def l2_penalty(model):
    """Output-layer L2 term added to the loss, or None (mvts l2_reg_loss)."""

    if config.GLOBAL_REG or not config.L2_REG:
        return None

    return config.L2_REG * torch.sum(torch.square(model.output_layer.weight))


def lr_for_epoch(epoch):
    """mvts step schedule: the rate is multiplied by LR_FACTORS[i] after epoch LR_STEPS[i]."""

    lr = config.LEARNING_RATE

    for step, factor in zip(config.LR_STEPS, config.LR_FACTORS):
        if epoch > step:
            lr *= factor

    return lr


def backward_step(model, optimizer, loss):
    """Backward pass, gradient clipping as in mvts (max norm 4.0 by default), optimizer step."""

    optimizer.zero_grad()
    loss.backward()

    if config.GRAD_CLIP_NORM:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)

    optimizer.step()


def fit(phase, model, model_args, datasets, collate_fn, run_epoch, num_epochs, key_metric, extra, log_lines=()):
    """Train `model` and save best_model.pt (by key_metric), last_model.pt, and optional epoch files.

    run_epoch(model, loader, device, optimizer=None) returns a dict of metrics including "loss".
    `extra` is added to every checkpoint (e.g. class_mapping). Returns the run folder.
    """

    check_settings()

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA, but no CUDA device is available.")

    device = torch.device("cuda")
    generator = None

    if config.SEED is not None:
        generator = torch.Generator()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def loader(split):
        return DataLoader(
            datasets[split],
            batch_size=config.BATCH_SIZE,
            shuffle=split == "train",
            collate_fn=lambda batch: collate_fn(batch, config.MAX_SEQUENCE_LENGTH),
            num_workers=0,  # on Windows, each worker would copy the in-RAM dataset
            pin_memory=True,
            generator=generator if split == "train" else None,
        )

    loaders = {split: loader(split) for split in datasets}

    model.to(device)
    optimizer = make_optimizer(model)

    better = (lambda a, b: a < b) if key_metric in NEG_METRICS else (lambda a, b: a > b)
    best_value = float("inf") if key_metric in NEG_METRICS else float("-inf")
    start_epoch = 1

    if config.RESUME_FROM:
        state = torch.load(config.RESUME_FROM, map_location=device)
        if state["phase"] != phase or state["model_args"] != model_args:
            raise ValueError(
                f"RESUME_FROM {config.RESUME_FROM} is a '{state['phase']}' checkpoint with model_args "
                f"{state['model_args']}, which doesn't match this {phase} run ({model_args})."
            )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = state["epoch"] + 1
        best_value = state["best_value"]

    run_dir = Path(config.TRANSFORMER_OUTPUT_DIR) / datetime.now().strftime(f"{phase}_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    epoch_dir = run_dir / "epochs"
    if config.SAVE_EVERY_EPOCH:
        epoch_dir.mkdir()

    def log(line=""):
        print(line)
        with open(run_dir / "training_log.txt", "a", encoding="utf-8") as file:
            file.write(line + "\n")

    for name in MODEL_SETTINGS + TRAINING_SETTINGS:
        log(f"{name}: {getattr(config, name)}")
    for line in log_lines:
        log(line)
    log(f"key metric: {key_metric}")
    log(f"train samples: {len(datasets['train']):,}")
    log(f"validation samples: {len(datasets['validation']):,}")
    log(f"parameters: {sum(p.numel() for p in model.parameters()):,} "
        f"(trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,})")
    if config.RESUME_FROM:
        log(f"resumed after epoch {start_epoch - 1}, best val {key_metric} so far {best_value:.4f}")
    log()

    def checkpoint(epoch, val_metrics, with_optimizer):
        state = {
            "phase": phase,
            "model_state": model.state_dict(),
            "model_args": model_args,
            "feature_columns": list(config.FEATURE_COLUMNS),
            "max_sequence_length": config.MAX_SEQUENCE_LENGTH,
            "epoch": epoch,
            "val_loss": val_metrics.get("loss") if val_metrics else None,
            "val_metrics": val_metrics,
            "key_metric": key_metric,
            **extra,
        }
        if with_optimizer:
            state.update(optimizer_state=optimizer.state_dict(), best_value=best_value)
        return state

    def validate(epoch):
        nonlocal best_value

        with torch.no_grad():
            metrics = run_epoch(model, loaders["validation"], device)

        log(f"epoch {epoch} validation | " + " | ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

        if better(metrics[key_metric], best_value):
            best_value = metrics[key_metric]
            torch.save(checkpoint(epoch, metrics, False), run_dir / "best_model.pt")
            log(f"saved best model (val {key_metric} {best_value:.4f})")

        return metrics

    val_metrics = None

    if start_epoch == 1:  # mvts evaluates once before training
        if config.SEED is not None:
            seed_epoch(config.SEED, 0, generator)
        val_metrics = validate(0)

    for epoch in range(start_epoch, num_epochs + 1):

        if config.SEED is not None:
            seed_epoch(config.SEED, epoch, generator)

        lr = lr_for_epoch(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        started = time.time()
        train_metrics = run_epoch(model, loaders["train"], device, optimizer)
        log(f"epoch {epoch}/{num_epochs} train | lr {lr:.2e} | "
            + " | ".join(f"{k} {v:.4f}" for k, v in train_metrics.items())
            + f" | {time.time() - started:.0f} s")

        if epoch == num_epochs or epoch == start_epoch or epoch % config.VAL_INTERVAL == 0:
            val_metrics = validate(epoch)

        state = checkpoint(epoch, val_metrics, True)
        torch.save(state, run_dir / "last_model.pt")
        if config.SAVE_EVERY_EPOCH:
            torch.save(state, epoch_dir / f"epoch_{epoch:03d}.pt")

    log(f"best val {key_metric} {best_value:.4f}, checkpoint: {run_dir / 'best_model.pt'}")

    return run_dir
