# BatNET Transformer: Project Overview

This project turns raw bat echolocation WAV recordings into a labeled dataset and trains a transformer classifier on it. Stages 1 to 3 are copied unchanged from batnet-lstm. Stage 4 is a port of mvts_transformer (Zerveas et al., "A Transformer-based Framework for Multivariate Time Series Representation Learning", KDD 2021): self-supervised encoder pretraining, then classification fine-tuning.

## Pipeline

| Stage | Folder | Entry point | Output |
|---|---|---|---|
| 1. Zero-crossing (ZC) extraction | `zc/` | `python zc/pipeline.py` (`main()` returns in-memory results only) | Pulse point data per WAV |
| 2. Preprocessing | `preprocess/` | `python preprocess/pipeline.py` | Normalized per-pulse CSVs in `config.OUTPUT_DIR`, plus `pulse_rejection_summary.csv` |
| 3. Dataset creation | `dataset_creation/` | `python dataset_creation/build_shards.py` | NumPy shards, `class_mapping.json`, `metadata.jsonl`, `manifest.jsonl` in `config.SHARD_PARENT_DIR` |
| 4a. Encoder pretraining | `transformer/` | `python transformer/pretrain.py` | `pretrain_YYYYMMDD_HHMMSS/` in `config.TRANSFORMER_OUTPUT_DIR` |
| 4b. Classifier fine-tuning | `transformer/` | `python transformer/finetune.py` | `finetune_YYYYMMDD_HHMMSS/` in `config.TRANSFORMER_OUTPUT_DIR` |
| 4c. Inference | `transformer/` | `python transformer/inference.py`, `python transformer/validation_inference.py` | Predictions and failed-files CSVs |

Stage details live in their own steering files:
- `zc-stage.md` for `zc/`
- `preprocess-stage.md` for `preprocess/`
- `dataset-creation.md` for `dataset_creation/`
- `transformer.md` for `transformer/`
- `data-contracts.md` for the file formats passed between stages

## Configuration

All tunable parameters and absolute paths live in `config.py` at the project root. Each stage imports `config` and reads from it.

- Do not hardcode paths, thresholds, or feature lists in stage code. Add them to `config.py`.
- Paths in `config.py` are Windows absolute paths under `C:\Users\Students\...`. Use `pathlib.Path` and raw strings when editing them.
- `NUM_WORKERS` controls the process pool size in stages 1 to 3 and inference.
- Changing any parameter that affects features (ZC settings, PCHIP interval, `NORMALIZATION_COLUMNS`, normalization stats) invalidates existing outputs. Regenerate downstream stages after such a change.

## Running the pipeline

Run every script from the project root so the `sys.path` setup in each module resolves `config` and the stage packages.

1. `preprocess/get_normalization_stats.py` (only when the feature set or the training data changes)
2. `dataset_creation/build_shards.py` with `VAL_RATIO > 0`, once for the pretraining data and once for the fine-tuning data (they can be the same folder). Point `PRETRAIN_SHARD_DIR` and `FINETUNE_SHARD_DIR` at them.
3. `transformer/pretrain.py`
4. Set `PRETRAINED_ENCODER_PATH` to the pretraining run's `best_model.pt`, then `transformer/finetune.py`. With `PRETRAINED_ENCODER_PATH = None` the classifier trains from scratch.
5. Set `CHECKPOINT_PATH` to the fine-tuning run's `best_model.pt`, then `transformer/inference.py` on `TEST_DIR`. It runs stages 1 and 2 internally, without class filtering, and writes no shards.
6. Optional: `transformer/validation_inference.py` on the validation WAVs of `FINETUNE_SHARD_DIR`. `INPUT_DIR` must still point at the training WAVs.

Run long jobs in your own terminal. They use multiprocessing or a GPU and can take hours on the full dataset. Do not start them through the agent shell.

## Conventions

- Python with NumPy, pandas, librosa, PyTorch, and tqdm. Use `ProcessPoolExecutor` with an `initializer` for per-worker setup, as the existing stages do.
- Guard every script entry point with `if __name__ == "__main__":`.
- Keep stage modules importable without side effects. Work happens inside functions.
- Class labels come from the immediate parent folder name of each WAV file, for example `.../INPUT_DIR/MYTCAL/recording.wav` has class `MYTCAL`. This rule is used by preprocessing, normalization stats, dataset creation, and inference (`ground_truth`). Keep it consistent.
- The stage 1 to 3 code skips failed files silently. When you add error handling, log the file path and the reason. Do not add new silent `except: pass` blocks.

## Current scope

- Do not change pipeline behavior or refactor stages unless the user asks.
- Stages 1 to 3 are copies of batnet-lstm. They are not synced automatically. If the user changes them in one repo, ask whether to apply the change in the other.
- Keep modules short and simple. Do not overcomment or overengineer.
- Do not add tests unless the user asks for them.
