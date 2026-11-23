---
inclusion: fileMatch
fileMatchPattern: "transformer/**"
---

# Stage 4: Transformer

Folder: `transformer/`. A port of mvts_transformer (Zerveas et al., KDD 2021) that reads batnet shards. The user asked to reproduce the paper exactly (BatchNorm, flatten head, softmax CE, gradient clipping at 4.0). All settings are in the "Stage 4" section of `config.py`.

## Modules

| File | Contents |
|---|---|
| `dataset.py` | `read_manifest`, `check_ram`, `load_split` (same as batnet-lstm), `PulseDataset`, `ImputationDataset`, `noise_mask`, `collate_supervised`, `collate_unsupervised` |
| `model.py` | `TSTransformerEncoder` (pretraining), `TSTransformerClassifier` (fine-tuning and inference), BatchNorm encoder layer, positional encodings |
| `training.py` | `fit` (shared training loop), seeding, optimizer, L2 penalty, step LR, gradient clipping |
| `pretrain.py` | Entry point: `python transformer/pretrain.py` |
| `finetune.py` | Entry point: `python transformer/finetune.py` |
| `inference.py` | Entry point: `python transformer/inference.py`. Also defines `run(wav_paths, predictions_path, failed_path)` |
| `validation_inference.py` | Entry point: `python transformer/validation_inference.py` |

Run all scripts from the project root.

## Data

- `load_split` loads a whole split into one float32 array in RAM before training, with the columns in `FEATURE_COLUMNS` (a subset of `NORMALIZATION_COLUMNS`, in any order). Samples longer than `MAX_SEQUENCE_LENGTH` are truncated; samples with NaN are dropped.
- Every batch is zero-padded to exactly `MAX_SEQUENCE_LENGTH` (as in mvts), with a boolean padding mask that is True at real steps. The flatten head needs this. With BatchNorm, the padded positions are part of the normalization statistics, so train and inference must pad to the same length. Keep fixed padding for all heads.
- Shard inputs are already z-score normalized. The mvts `Normalizer` is not ported. Do not normalize again.
- Train and validation come from the shard split folders (WAV-level split). Do not re-split pulses.

## Model

- `embed`: `Linear(feat_dim, d_model) * sqrt(d_model)`, positional encoding (`POS_ENCODING`), `NUM_LAYERS` post-norm encoder layers with the padding mask (`NORMALIZATION_LAYER`, BatchNorm by default), activation, dropout.
- `TSTransformerEncoder`: `output_layer = Linear(d_model, feat_dim)`, reconstruction per step.
- `TSTransformerClassifier`: padded embeddings are zeroed, then `CLASSIFIER_HEAD`:
  - `"flatten"` (paper): `Linear(d_model * max_len, n_classes)` on the flattened sequence.
  - `"mean"`: masked mean over real steps, then `Linear(d_model, n_classes)`.
  - `"attention"`: masked softmax attention (`output_attention`), then `Linear(d_model, n_classes)`.
- All head parameters have names starting with `output_`. Pretrained weights load by dropping `output_*` keys, and `FREEZE_ENCODER` trains only `output_*` parameters. Keep this naming.
- Output is raw logits. Inference applies softmax.

## Pretraining (`pretrain.py`)

- mvts task "imputation". `noise_mask` masks each feature separately with geometric segment lengths (mean `MEAN_MASK_LENGTH`, ratio `MASKING_RATIO`). It's the mvts Markov chain drawn with NumPy. A new mask is drawn on every access, so validation masks change each time too.
- Masked inputs are set to 0. Loss is the MSE over masked real values. Shard labels are ignored. Best model by validation loss.

## Fine-tuning (`finetune.py`)

- Softmax cross-entropy. Metrics: loss, accuracy, and mvts "precision" (precision per class, averaged with class-support weights, over classes predicted at least once). `KEY_METRIC` picks `best_model.pt`.
- `PRETRAINED_ENCODER_PATH` loads every non-`output_*` weight from a pretraining `best_model.pt`. Model settings and `FEATURE_COLUMNS` must match; the script raises an error otherwise. `None` trains from scratch.

## Training loop (`training.fit`)

Follows mvts `src/main.py`:

- Validation before training (epoch 0), after the first epoch, every `VAL_INTERVAL` epochs, and after the last. `best_model.pt` is written when the key metric improves. There is no early stopping.
- Optimizer `OPTIMIZER` (torch built-ins). `L2_REG` is weight decay when `GLOBAL_REG`, otherwise an L2 term on `output_layer.weight` added to the loss.
- Step LR: the rate is `LEARNING_RATE` times every `LR_FACTORS[i]` whose `LR_STEPS[i]` is below the current epoch. It's set at the start of each epoch, so it survives a resume.
- Gradient clipping with `clip_grad_norm_(GRAD_CLIP_NORM)`, 4.0 as in mvts.
- Requires CUDA. DataLoaders use `num_workers=0`, because on Windows each worker would copy the in-RAM dataset.
- Seeding (from batnet-lstm): with `SEED` set, `random`, NumPy, torch and the shuffle generator are seeded with `SEED + epoch` at the start of every epoch (and `SEED + 0` before model init and before the epoch 0 validation). cuDNN is deterministic.
- Resume (from batnet-lstm): `RESUME_FROM` takes a `last_model.pt` or `epochs/epoch_XXX.pt` from a run of the same phase with the same `model_args`. It restores model and optimizer state and the best key-metric value, continues from the next epoch, and always writes a new run folder. That folder's `best_model.pt` is only written if the metric improves after the resume. When resuming fine-tuning, `PRETRAINED_ENCODER_PATH` is not reloaded.
- Run folder: `training_log.txt`, `best_model.pt`, `last_model.pt` (every epoch, with optimizer state), and `epochs/epoch_XXX.pt` when `SAVE_EVERY_EPOCH`.

## Inference

- Same preprocessing as batnet-lstm `classifier/inference.py`: stage 1 (`zc.pipeline._process_wav_unsafe`), PCHIP, slope and position, z-score with `preprocess/normalization_stats.json`. No class filtering, no files written besides the CSVs.
- Uses the checkpoint's `feature_columns`, `max_sequence_length`, `model_args`, `class_mapping`. Pads every batch to `max_sequence_length`. Runs on the CPU.
- `validation_inference.py` reads validation WAVs from `FINETUNE_SHARD_DIR/metadata.jsonl` and joins them with `INPUT_DIR`.

## Rules for this stage

- Keep the mvts behavior unless the user asks for a change: BatchNorm default, fixed padding, flatten head default, softmax CE, clipping at 4.0, validation schedule, no early stopping.
- Keep modules short and simple. No diagnostics CSVs, no mixed precision, no class weighting unless the user asks.
- Keep inference preprocessing in the same order as `preprocess/pipeline.py::process_wav_file`, minus class filtering.
- Do not start a training run from the agent shell. The user runs training and inference.
