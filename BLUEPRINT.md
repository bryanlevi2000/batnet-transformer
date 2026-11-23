# batnet-transformer: Blueprint

Goal: keep batnet-lstm stages 1 to 3 (ZC extraction, preprocessing, shard building) and replace the stage 4 LSTM with the transformer encoder and classifier from mvts_transformer (Zerveas et al., KDD 2021).

Status: first version built (2026-11-16). Decisions are in section 5. Current behavior is described in .kiro/steering/.

## 1. What each codebase provides

### batnet-lstm (stages 1 to 3, reused)

Measured on `C:\Users\Students\Desktop\Working\model_training_110926V1\shards`:

| | train | validation |
|---|---|---|
| Samples (pulses) | 462,474 | 82,285 |
| Time steps | 25.9 M | 4.6 M |
| Features | 4: `freq_hz`, `amplitude`, `slope`, `position` | same |
| Classes | 13 (`BIGEAR` ... `PEISUB`) | same |
| NaN rows | 0 | 0 |

- All 4 features are already z-scored (mean about 0, std about 1 on the train split).
- Sequence length (steps of 100 us): min 11, median 47, 95th pct 124, 99th pct 167, max 299. Lengths above 300 can't occur, since `MAX_DURATION_US = 30000` and the grid is 100 us. Only 0.27 % of pulses are longer than 200 steps.
- The split is per WAV and already built into `train/` and `validation/`.
- Class imbalance: 8,415 (`MYTAUO`) to 87,999 (`MYTLUC`) training pulses.

The steering files in batnet-lstm say shards have 2 features and that slope and position aren't normalized. The current `config.py` and this shard set have 4 normalized features, so those steering lines are out of date.

### mvts_transformer (stage 4 model, adapted)

What it expects:

- Input format: one pandas DataFrame per dataset, indexed by sample ID, one row per time step, one column per feature. The bundled loaders read `.ts` files (sktime) or CSVs, but the model itself only sees tensors `X (batch, seq_len, n_features)` plus a boolean `padding_masks (batch, seq_len)`.
- Number of features: any. `feat_dim` is taken from the data.
- Normalization: by default it computes its own z-score over the training set (`--normalization standardization`). Our shards are already z-scored, so this step must be skipped.
- Variable length: supported by zero-padding and a padding mask, but every batch is padded to one fixed `max_seq_len`. The classifier head is `Linear(d_model * max_seq_len, n_classes)` over the flattened sequence, so the length must be fixed.
- Splitting: random `StratifiedShuffleSplit` per sample. For us that would put pulses from one WAV in both train and validation, so we won't use it.

Model (`src/models/ts_transformer.py`, `TSTransformerEncoderClassiregressor`):

1. `Linear(n_features, d_model) * sqrt(d_model)`
2. Positional encoding, fixed sinusoidal or learnable, sized to `max_seq_len`
3. `num_layers` transformer encoder layers with the padding mask, using BatchNorm (default) or LayerNorm
4. GELU, dropout, then padded positions are zeroed
5. Flatten to `(batch, max_seq_len * d_model)`, then `Linear` to logits
6. Softmax cross-entropy loss

Other parts:

- Self-supervised pretraining (`task=imputation`): random geometric masks over the input, with the encoder trained to reconstruct the masked values (masked MSE). After that, fine-tune the classifier from those weights. The paper reports gains from this, mostly when labels are scarce.
- Training loop: RAdam/Adam/AdamW, step LR schedule, gradient clipping at norm 4.0, optional L2 on the output layer only, best checkpoint by `key_metric`.
- Dependencies that are outdated or not needed: `sktime`, `xlwt`, `xlrd`, `xlutils` (`.xls` records), `ipdb`, tensorboard.

## 2. The mapping

| mvts concept | batnet source |
|---|---|
| sample | one pulse (`offsets[i]:offsets[i+1]` in a shard) |
| `feature_df` | in-RAM float32 array from `load_split` (batnet-lstm `classifier/dataset.py`) |
| `labels_df` | shard `labels` |
| `class_names` | `class_mapping.json` |
| `normalization` | none, shards are already z-scored |
| train/val split | shard `train/` and `validation/` folders |
| `max_seq_len` | `MAX_SEQUENCE_LENGTH` (300 fits every pulse) |
| `collate_superv` | batnet `collate_fn` plus a padding mask |

So there is no data conversion step. The transformer reads the shards directly. A pandas DataFrame with 26 M rows and `.loc` lookups per sample would be slow and use far more RAM than the existing offset-indexed array.

## 3. Proposed layout

```
batnet-transformer/
  config.py                 stages 1-3 as in batnet-lstm, plus a Stage 4 transformer section
  zc/                       copied from batnet-lstm
  preprocess/               copied from batnet-lstm (incl. normalization_stats.json, pulse_class_filters.json)
  dataset_creation/         copied from batnet-lstm
  transformer/
    dataset.py              load_split, PulseDataset (from batnet-lstm); collate_fn returns padding mask
    model.py                TSTransformerEncoder + classifier (trimmed port of ts_transformer.py)
    train.py                supervised training (batnet-lstm style loop and outputs)
    pretrain.py             optional, only if pretraining is in scope (Q5)
    inference.py            batnet-lstm inference pipeline with the new model
    validation_inference.py same as batnet-lstm
  .kiro/steering/           project-overview, transformer stage, data contracts
  requirements.txt
```

Outputs stay compatible with batnet-lstm: `run_YYYYMMDD_HHMMSS/` with `training_log.txt` and `best_model.pt` (`model_state`, `model_args`, `feature_columns`, `max_sequence_length`, `class_mapping`, `epoch`, `val_loss`), and the same predictions and failed-files CSV formats. Then the analysis you already do on LSTM predictions works on transformer predictions without changes.

## 4. Design points that need care

1. **Fixed length and compute.** With the paper's flatten head, every batch is padded to `max_seq_len`. At 300 steps and a median of 47, about 84 % of the positions are padding, and self-attention cost grows with length squared. Options: cap at 200 (truncates 0.27 % of pulses) or 300 (no truncation), or switch to a pooled head (Q2).
2. **BatchNorm and padding.** The default BatchNorm layers compute statistics over padded positions too. Those positions aren't zero inside the encoder, because the positional encoding is added to them. With 84 % padding the statistics are mostly padding. That works as long as train and inference pad to the same fixed length, but it breaks with per-batch padding. LayerNorm avoids the problem.
3. **Loss.** mvts uses softmax cross-entropy (one class per pulse). batnet-lstm uses independent sigmoid heads with BCE, and the predictions CSV stores sigmoid confidences (Q3).
4. **No double normalization.** The mvts `Normalizer` and its default `standardization` setting are not ported.
5. **WAV-level split.** Uses the shard split as built, the same rule as batnet-lstm.
6. **Inference preprocessing.** Same as batnet-lstm `inference.py`: ZC, PCHIP, slope and position, z-score, no class filtering. Then pad to `max_sequence_length` from the checkpoint.

## 5. Decisions (2026-11-16)

1. Stages 1 to 3 are copied into this repo.
2. Classifier head: paper flatten head with fixed length by default. CLASSIFIER_HEAD switches to masked mean or attention pooling.
3. Loss: softmax cross-entropy. The predictions CSV keeps the batnet-lstm format, with softmax confidences.
4. Norm layer: BatchNorm, as in the paper.
5. Pretraining is in scope: transformer/pretrain.py on PRETRAIN_SHARD_DIR, then transformer/finetune.py on FINETUNE_SHARD_DIR.
6. All four features are normalized and stored in shards. FEATURE_COLUMNS picks what the model sees.
7. mvts training loop, not the batnet-lstm loop. Gradient clipping at 4.0 (mvts clips). Seeding and resume carried over from batnet-lstm. No diagnostics CSVs.
8. mvts default hyperparameters. Each split is loaded into RAM once before training.
