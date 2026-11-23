---
inclusion: fileMatch
fileMatchPattern: "config.py"
---

# config.py Rules

`config.py` is the single source for paths and parameters. Stage scripts import it as `import config`.

- Keep the existing section headers (Global, Stage 1 to 4) and the sub-headers of Stage 4. Add new settings under the matching header.
- Model settings (`FEATURE_COLUMNS`, `MAX_SEQUENCE_LENGTH`, `D_MODEL` to `NORMALIZATION_LAYER`) must be the same for pretraining and fine-tuning when a pretrained encoder is used. `finetune.py` checks this and raises an error.
- Keep `NORMALIZATION_COLUMNS` in sync with `preprocess/normalization_stats.json`. Update the comment above the list whenever the list changes.
- `VAL_RATIO` comment: it is set to `0.0` for inference-only conversion. Do not change it without asking.
- Do not add secrets or credentials to this file.
- Changing any setting that affects features or shards makes existing outputs stale. Mention which outputs need regeneration when you change one.
