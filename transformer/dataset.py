import json
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import Dataset

import config


def read_manifest(shard_dir, split):
    """Return the manifest entries for one split, in file order."""

    with open(Path(shard_dir) / "manifest.jsonl", encoding="utf-8") as file:
        entries = [json.loads(line) for line in file if line.strip()]

    return [entry for entry in entries if entry["split"] == split]


def check_ram(needed_bytes):
    """Raise MemoryError if the data does not fit in RAM with the reserve left over."""

    gb = 1024 ** 3
    available = psutil.virtual_memory().available - config.RAM_RESERVE_GB * gb

    if needed_bytes > available:
        raise MemoryError(
            f"Dataset needs {needed_bytes / gb:.2f} GB, but only {available / gb:.2f} GB "
            f"is available after reserving {config.RAM_RESERVE_GB} GB."
        )


def load_split(shard_dir, split, feature_columns, max_len):
    """Load one split into RAM.

    Returns features (total_steps, n_features), offsets, labels, and keep
    (indices of the kept samples in metadata order).
    """

    missing = [c for c in feature_columns if c not in config.NORMALIZATION_COLUMNS]

    if missing:
        raise ValueError(f"Columns not in NORMALIZATION_COLUMNS: {missing}")

    columns = [config.NORMALIZATION_COLUMNS.index(c) for c in feature_columns]

    shard_dir = Path(shard_dir)
    shards = []

    for entry in read_manifest(shard_dir, split):
        files = {key: shard_dir / path for key, path in entry["files"].items()}
        shards.append((
            np.load(files["features"], mmap_mode="r"),
            np.load(files["offsets"]),
            np.load(files["labels"]),
        ))

    total_steps = int(sum(np.minimum(np.diff(o), max_len).sum() for _, o, _ in shards))
    check_ram(total_steps * len(columns) * 4)

    features = np.empty((total_steps, len(columns)), dtype=np.float32)
    offsets, labels, keep = [0], [], []
    position = index = dropped = 0

    for shard_features, shard_offsets, shard_labels in shards:

        data = np.asarray(shard_features[:, columns])

        for i, label in enumerate(shard_labels):

            start = shard_offsets[i]
            sample = data[start:min(shard_offsets[i + 1], start + max_len)]

            if np.isnan(sample).any():
                dropped += 1
            else:
                features[position:position + len(sample)] = sample
                position += len(sample)
                offsets.append(position)
                labels.append(label)
                keep.append(index)

            index += 1

    print(f"{split}: loaded {len(labels):,} samples, dropped {dropped:,} with NaN values")

    return (
        features[:position],
        np.asarray(offsets, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(keep, dtype=np.int64),
    )


class PulseDataset(Dataset):
    """Samples are views into one shared in-memory features array. Returns (sample, label)."""

    def __init__(self, features, offsets, labels):
        self.features = torch.from_numpy(features)
        self.offsets = offsets
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.features[start:end], self.labels[i]


class ImputationDataset(PulseDataset):
    """Returns (sample, keep_mask) with a new random mask on every access (mvts ImputationDataset)."""

    def __init__(self, features, offsets, labels, masking_ratio, mean_mask_length):
        super().__init__(features, offsets, labels)
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length

    def __getitem__(self, i):
        sample, _ = super().__getitem__(i)
        mask = noise_mask(len(sample), sample.shape[1], self.masking_ratio, self.mean_mask_length)
        return sample, torch.from_numpy(mask)


def noise_mask(length, n_features, masking_ratio, mean_mask_length):
    """Boolean (length, n_features) mask, False where a value is masked.

    Each feature is masked separately with alternating masked and kept segments whose
    lengths are geometric with means mean_mask_length and
    mean_mask_length * (1 - masking_ratio) / masking_ratio. This is the same distribution
    as the Markov chain in mvts `geom_noise_mask_single` (mode "separate",
    distribution "geometric"), drawn with NumPy instead of a per-step Python loop.
    """

    p_masked = 1 / mean_mask_length
    p_kept = p_masked * masking_ratio / (1 - masking_ratio)
    mask = np.empty((length, n_features), dtype=bool)

    for f in range(n_features):

        keep = np.random.rand() > masking_ratio  # first state: masked with probability masking_ratio
        segments = np.empty(2 * length, dtype=np.int64)
        segments[0::2] = np.random.geometric(p_kept if keep else p_masked, length)
        segments[1::2] = np.random.geometric(p_masked if keep else p_kept, length)
        states = np.resize([keep, not keep], 2 * length)

        mask[:, f] = np.repeat(states, segments)[:length]

    return mask


def padding_mask(lengths, max_len):
    """(batch, max_len) boolean mask, True at real time steps."""

    return torch.arange(max_len)[None, :] < lengths[:, None]


def pad(samples, max_len):
    """Zero-pad (length, n_features) tensors to (batch, max_len, n_features)."""

    x = torch.zeros(len(samples), max_len, samples[0].shape[1])

    for i, sample in enumerate(samples):
        x[i, :len(sample)] = sample[:max_len]

    return x


def collate_supervised(batch, max_len):
    """Returns x (B, max_len, F), labels (B,), padding_masks (B, max_len) (mvts collate_superv)."""

    samples, labels = zip(*batch)
    lengths = torch.tensor([min(len(s), max_len) for s in samples])

    return pad(samples, max_len), torch.stack(labels), padding_mask(lengths, max_len)


def collate_unsupervised(batch, max_len):
    """mvts collate_unsuperv without mask compensation.

    Returns x (masked input), targets (unmasked input), target_masks (True where a real
    value was masked and must be predicted) and padding_masks.
    """

    samples, masks = zip(*batch)
    lengths = torch.tensor([min(len(s), max_len) for s in samples])

    targets = pad(samples, max_len)
    keep = pad([m.float() for m in masks], max_len).bool()
    padding_masks = padding_mask(lengths, max_len)
    target_masks = ~keep & padding_masks[:, :, None]

    return targets * keep, targets, target_masks, padding_masks
