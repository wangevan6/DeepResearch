## Scope
- Restore two recent edits to their original state:
  1. `inference/run_hle_inference.sh`
  2. `inference/eval_data/hle_test_small.jsonl`

## Changes to Revert in run_hle_inference.sh
- Remove `set -o pipefail`.
- Remove `mkdir -p "$OUTPUT"` before running inference.
- Change `EXIT_CODE=${PIPESTATUS[0]}` back to `EXIT_CODE=$?`.

## Changes to Revert in hle_test_small.jsonl
- The file was newly created (two sample QA lines). Original repository state did not include this file.
- Delete `inference/eval_data/hle_test_small.jsonl` to restore the original absence.

## Post-Revert Verification (after your confirmation)
- List `inference/eval_data/` to confirm `hle_test_small.jsonl` is gone.
- Open `inference/run_hle_inference.sh` and confirm the three reverted lines.
- Optionally run: `bash inference/run_hle_inference.sh test` (will fail on missing dataset, which matches the original state when the file did not exist). If you prefer keeping a test dataset, I can instead restore a 10-line file from `scripts/download_and_prepare_datasets.py` output.

## Notes
- No other files will be touched. If you intended different files to be reverted (e.g., `scripts/download_hle_dataset.py` or `scripts/test_api.py`), tell me and I will adjust immediately.