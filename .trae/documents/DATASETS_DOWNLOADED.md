# Downloaded Datasets Summary

**Date**: 2025-11-17
**Location**: `inference/eval_data/`
**Script**: `scripts/download_and_prepare_datasets.py`

## Overview

Successfully downloaded **4 out of 8** target datasets with full versions and toy samples (10 questions each, seed=42).

---

## ✅ Successfully Downloaded Datasets

### 1. HLE (Humanity's Last Exam)
- **Full Dataset**: `hle_full.jsonl` (118 MB)
- **Source**: `cais/hle` (HuggingFace - gated, already downloaded previously)
- **Format**: JSONL with `question` and `answer` fields
- **Size**: ~thousands of examples
- **Note**: Gated dataset, requires HuggingFace authentication

### 2. WebWalkerQA
- **Full Dataset**: `webwalkerqa_full.jsonl` (163 KB)
- **Toy Sample**: `webwalkerqa_toy.jsonl` (2.5 KB, 10 questions)
- **Source**: `callanwu/WebWalkerQA` (HuggingFace)
- **Total Questions**: 680
- **Format**: JSONL with `question` and `answer` fields
- **Description**: Web navigation benchmark

### 3. FRAMERS (Google FRAMES Benchmark)
- **Full Dataset**: `framers_full.jsonl` (50 KB)
- **Toy Sample**: `framers_toy.jsonl` (432 bytes, 10 questions)
- **Source**: `google/frames-benchmark` (HuggingFace)
- **Total Questions**: 824
- **Format**: JSONL with `question` and `answer` fields
- **Description**: Multi-hop reasoning benchmark requiring information from 2-15 Wikipedia articles

### 4. SimpleQA
- **Full Dataset**: `simpleqa_full.jsonl` (596 KB)
- **Toy Sample**: `simpleqa_toy.jsonl` (1.4 KB, 10 questions)
- **Source**: `basicv8vc/SimpleQA` (HuggingFace, split="test")
- **Total Questions**: 4,326
- **Format**: JSONL with `question` (from `problem` field) and `answer` fields
- **Description**: Factuality benchmark from OpenAI

### 5. xbench-DeepSearch
- **Full Dataset**: `xbench_deepsearch_full.jsonl` (5.7 KB)
- **Toy Sample**: `xbench_deepsearch_toy.jsonl` (754 bytes, 10 questions)
- **Source**: `xbench/DeepSearch-2510` (HuggingFace, split="train")
- **Total Questions**: 100
- **Format**: JSONL with `question` and `answer` fields (**ENCRYPTED**)
- **⚠️ Important**: Data is encrypted to prevent contamination. Requires decryption using: https://github.com/xbench-ai/xbench-evals
- **Description**: Search evaluation benchmark with encrypted data

---

## ❌ Failed Downloads

### 6. GAIA
- **Status**: ❌ Failed (Gated dataset, requires authentication)
- **Source**: `gaia-benchmark/GAIA`
- **Action Needed**:
  1. Request access at https://huggingface.co/datasets/gaia-benchmark/GAIA
  2. Login: `huggingface-cli login`
  3. Re-run download script

### 7. BrowseComp-EN
- **Status**: ❌ Failed (Dataset not found on HuggingFace)
- **Source**: `Alibaba-NLP/BrowseComp` (split="en")
- **Note**: Existing file `browsecomp_test.jsonl` (6.9 KB) from previous download exists but may be synthetic data

### 8. BrowseComp-ZH
- **Status**: ❌ Failed (Dataset not found on HuggingFace)
- **Source**: `Alibaba-NLP/BrowseComp` (split="zh")
- **Note**: Dataset may not be publicly available yet or uses different repository path

---

## File Structure

```
inference/eval_data/
├── # Full datasets (complete versions)
├── hle_full.jsonl                    (118 MB, gated - already exists)
├── webwalkerqa_full.jsonl            (163 KB, 680 questions) ✅
├── framers_full.jsonl                (50 KB, 824 questions) ✅
├── simpleqa_full.jsonl               (596 KB, 4,326 questions) ✅
├── xbench_deepsearch_full.jsonl      (5.7 KB, 100 questions - encrypted) ✅
│
├── # Toy samples (10 questions each, seed=42)
├── webwalkerqa_toy.jsonl             (2.5 KB) ✅
├── framers_toy.jsonl                 (432 bytes) ✅
├── simpleqa_toy.jsonl                (1.4 KB) ✅
├── xbench_deepsearch_toy.jsonl       (754 bytes) ✅
│
├── # Other files (previously existing)
├── test_small.jsonl                  (345 bytes, 5 test questions)
├── hle_test_small.jsonl              (4.7 KB)
├── hle_text_only.jsonl               (1.9 MB)
├── browsecomp_test.jsonl             (6.9 KB, synthetic/old data)
├── example.jsonl                     (59 bytes)
└── example_with_file.jsonl           (73 bytes)
```

---

## Total Downloaded Statistics

| Dataset | Full Size | Toy Size | Questions | Status |
|---------|-----------|----------|-----------|--------|
| HLE | 118 MB | N/A | Thousands | ✅ (existing) |
| WebWalkerQA | 163 KB | 2.5 KB | 680 | ✅ New |
| FRAMERS | 50 KB | 432 B | 824 | ✅ New |
| SimpleQA | 596 KB | 1.4 KB | 4,326 | ✅ New |
| xbench-DeepSearch | 5.7 KB | 754 B | 100 (encrypted) | ✅ New |
| **TOTAL** | **~119 MB** | **~5 KB** | **~6,000+** | **5/8** |

---

## Next Steps

### To Access Gated Datasets (GAIA, HLE if not already downloaded):
```bash
# 1. Install HuggingFace CLI
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/pip install huggingface_hub"

# 2. Login with your HuggingFace token
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/huggingface-cli login"

# 3. Request access to gated datasets:
# - GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
# - HLE: https://huggingface.co/datasets/cais/hle

# 4. Re-run download for GAIA
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && ~/miniconda3/envs/react_infer_env/bin/python scripts/download_and_prepare_datasets.py --datasets gaia --download_full --create_toy --toy_size 10"
```

### To Decrypt xbench-DeepSearch:
```bash
# Clone the xbench-evals repository
git clone https://github.com/xbench-ai/xbench-evals
cd xbench-evals

# Follow their decryption instructions
# The decryption key should be available in their repository
```

### To Find BrowseComp Datasets:
The `Alibaba-NLP/BrowseComp` dataset path appears incorrect. Possible alternatives:
1. Check Alibaba-NLP organization on HuggingFace for correct repository name
2. Contact Alibaba-NLP/DeepResearch team for dataset access
3. Check if BrowseComp data is included in DeepResearch repository directly

---

## Usage Examples

### Quick Testing with Toy Samples
```bash
# Test with WebWalkerQA toy sample (10 questions)
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && . ~/miniconda3/etc/profile.d/conda.sh && conda activate react_infer_env && bash inference/run_react_infer_openrouter.sh"

# Update DATASET in .env to:
# DATASET=/mnt/c/Users/user/Projects/DeepResearch/inference/eval_data/webwalkerqa_toy.jsonl
```

### Full Evaluation
```bash
# Run full WebWalkerQA evaluation (680 questions)
# Update DATASET in .env to:
# DATASET=/mnt/c/Users/user/Projects/DeepResearch/inference/eval_data/webwalkerqa_full.jsonl
```

---

## Script Information

### Download Script
**Path**: `scripts/download_and_prepare_datasets.py`

**Features**:
- Downloads 8 datasets from HuggingFace
- Saves full versions as `*_full.jsonl`
- Creates toy samples as `*_toy.jsonl` with configurable size
- Normalizes all datasets to standard `{"question": "...", "answer": "..."}` format
- Handles gated datasets with authentication
- Provides detailed error messages and troubleshooting steps

**Usage**:
```bash
# Download all datasets (full + toy samples)
python scripts/download_and_prepare_datasets.py --datasets all --download_full --create_toy --toy_size 10 --seed 42

# Download specific datasets only
python scripts/download_and_prepare_datasets.py --datasets webwalkerqa framers --download_full --create_toy

# Download without toy samples
python scripts/download_and_prepare_datasets.py --datasets all --download_full
```

**Options**:
- `--datasets`: Choose which datasets to download (gaia, hle, browsecomp_en, browsecomp_zh, webwalkerqa, framers, simpleqa, xbench, all)
- `--download_full`: Download complete datasets (vs sampling)
- `--create_toy`: Create toy samples from full datasets
- `--toy_size`: Number of questions in toy samples (default: 10)
- `--seed`: Random seed for sampling (default: 42)
- `--output_dir`: Output directory (default: inference/eval_data)

---

## Notes

1. **HLE** was already downloaded previously (118 MB file exists)
2. **xbench-DeepSearch** data is encrypted and needs decryption before use
3. **GAIA** requires HuggingFace authentication and approval
4. **BrowseComp** datasets are not available at the specified HuggingFace paths
5. All downloaded datasets are normalized to consistent `{"question": "...", "answer": "..."}` JSON format
6. Toy samples use random seed 42 for reproducibility

---

## Success Rate

**Downloaded**: 4/8 new datasets (5/8 including pre-existing HLE)
**Failed**: 3/8 datasets (GAIA - requires auth, BrowseComp-EN/ZH - not found)
**Total Questions**: ~6,000+ (including existing HLE)
