---
inclusion: fileMatch
fileMatchPattern: "{transformer/**,dataset_creation/**,preprocess/**}"
---

# Data Contracts Between Stages

These formats are the interface between stages. A change to any of them must be made in every stage that reads or writes the format. Stages 1 to 3 and the CSV formats match batnet-lstm.

## Stage 2 output: preprocessed pulse CSVs

Written by `preprocess/pipeline.py` to `config.OUTPUT_DIR`, mirroring the folder structure of `config.INPUT_DIR`.

- One CSV per WAV file, named `<wav_stem>.csv`, in the matching class subfolder.
- One row per pulse point. Columns include `time_us`, `freq_hz`, `amplitude`, `pulse_id`, `status`, `slope`, `position`.
- The columns in `config.NORMALIZATION_COLUMNS` (currently all four features) are already z-score normalized using `preprocess/normalization_stats.json`. Do not normalize them again.
- `pulse_id` identifies each pulse within a file. Values greater than 0 are real pulses.
- Rejection counts per class are written to `config.PULSE_FILTERED_REJECTION_FILE`. `dataset_creation/build_shards.py` writes the same file, in the same format, to the same path. Whichever script runs last overwrites it. Columns: `class`, `total_pulses`, `accepted`, `rejected`, `rejected_fmin`, `rejected_fmax`, `rejected_fc`, `rejected_duration`.

## Stage 3 output: NumPy shards

Written by `dataset_creation/build_shards.py` to `config.SHARD_PARENT_DIR`. Read by `transformer/` from `PRETRAIN_SHARD_DIR` and `FINETUNE_SHARD_DIR`; both need `train/` and `validation/`.

```
SHARD_PARENT_DIR/
  class_mapping.json          class name -> integer label (persistent across runs)
  metadata.jsonl              one JSON line per sample, in shard order
  manifest.jsonl              one JSON line per completed shard (resume source of truth)
  train/
    train_000000_features.npy
    train_000000_offsets.npy
    train_000000_labels.npy
  validation/                 only created when config.VAL_RATIO > 0
```

- `features`: float32, shape `(total_time_steps, n_features)`. All samples in the shard are concatenated along the first axis. `n_features` is `len(config.NORMALIZATION_COLUMNS)`, currently 4: `freq_hz`, `amplitude`, `slope`, `position`.
- `offsets`: int64, shape `(num_samples + 1,)`. Sample `i` is `features[offsets[i]:offsets[i+1]]`.
- `labels`: int64, shape `(num_samples,)`.

Sample length varies per pulse (100 us grid, at most 300 steps since `MAX_DURATION_US = 30000`). The transformer pads every sample to `MAX_SEQUENCE_LENGTH` with a padding mask.

`metadata.jsonl` fields per sample: `split`, `wav_path` (relative to `INPUT_DIR`, POSIX style), `pulse_id`, `label`, `num_time_steps`, `shard`, `sample_index`.

## Stage 4 output: checkpoints

`best_model.pt`, `last_model.pt` and `epochs/epoch_XXX.pt` in a run folder, saved with `torch.save` as a dict:

- `phase`: `"pretrain"` or `"finetune"`
- `model_state`: state dict (`TSTransformerEncoder` for pretrain, `TSTransformerClassifier` for finetune)
- `model_args`: constructor kwargs; rebuild with `TSTransformerEncoder(**model_args)` or `TSTransformerClassifier(**model_args)`
- `feature_columns`, `max_sequence_length`: the config values used for training
- `epoch`: 1-based epoch of the saved model (0 = before training)
- `val_loss`, `val_metrics` (dict), `key_metric`: from the last validation
- Fine-tuning only: `class_mapping` (from `FINETUNE_SHARD_DIR`), `pretrained_encoder_path`
- `last_model.pt` and `epochs/*` only: `optimizer_state`, `best_value` (used by `RESUME_FROM`)

Checkpoints hold only tensors and plain Python types (metrics are `float`), so they load with the torch 2.6+ default `weights_only=True`. Keep it that way; do not store NumPy values in them.

## Stage 4 output: predictions CSV

Written by `transformer/inference.py` to `config.PREDICTIONS_PATH`. There is one row per pulse found in the WAVs under `config.TEST_DIR`, in sorted file order, then pulse order. Pulses do not go through class-specific filtering. Samples with NaN are dropped. Columns, in order:

1. `file`: full WAV path
2. `pulse_id`
3. `ground_truth`: WAV parent folder name
4. One column per class, named with the class name, holding the softmax confidence (rows sum to 1)
5. One `<class>_logit` column per class, holding the raw logit

Both per-class groups are ordered by the checkpoint's label index. `validation_inference.py` writes the same format to `config.VAL_PREDICTIONS_PATH`.

## Stage 4 output: failed files CSV

Written by `inference.py` to `config.FAILED_FILES_PATH`, and by `validation_inference.py` to `config.VAL_FAILED_FILES_PATH`. Columns are `file` and `error`. It is written even when empty. WAVs with no detected pulses are not listed.

## Normalization statistics

`preprocess/normalization_stats.json` holds the mean and std for each feature in `NORMALIZATION_COLUMNS`. Keep the file and that list in sync. Inference reads the file at run time and checkpoints don't store it, so retrain after regenerating it.

## Changing the feature set

1. Update `config.NORMALIZATION_COLUMNS`.
2. Regenerate `normalization_stats.json` with `get_normalization_stats.py`.
3. Rerun `build_shards.py` into new output directories. Existing shards have the old width.
4. Set `FEATURE_COLUMNS` to a subset of the new list, then pretrain and fine-tune again. The model input size follows `len(FEATURE_COLUMNS)`, and a pretrained encoder only fits a classifier with the same `FEATURE_COLUMNS`.
