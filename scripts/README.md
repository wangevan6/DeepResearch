# DeepResearch Scripts

This directory contains automation scripts for the DeepResearch evaluation pipeline.

## 📋 Script Overview

### 1. `verify_setup.py` - Setup Verification
Validates your environment and configuration before running evaluations.

```bash
python scripts/verify_setup.py
```

**Checks:**
- ✓ Python version (3.10.x)
- ✓ Conda environment
- ✓ Required packages installed
- ✓ `.env` file with API keys
- ✓ Directory structure
- ✓ Dataset files
- ✓ OpenRouter API connection (optional)

**Run this first** to ensure everything is configured correctly!

---

### 2. `download_and_prepare_datasets.py` - Dataset Preparation
Downloads and formats benchmark datasets.

```bash
# Download all datasets (70 questions each)
python scripts/download_and_prepare_datasets.py --sample_size 70

# Download specific datasets
python scripts/download_and_prepare_datasets.py --datasets gaia hle --sample_size 50

# Just create test set
python scripts/download_and_prepare_datasets.py --datasets all --sample_size 10
```

**Options:**
- `--sample_size N`: Number of questions per benchmark (default: 70)
- `--datasets LIST`: Which datasets to download (gaia, hle, browsecomp, all)
- `--output_dir PATH`: Where to save datasets (default: inference/eval_data)
- `--seed N`: Random seed for reproducibility (default: 42)

**Outputs:**
- `inference/eval_data/test_small.jsonl` - 5 test questions
- `inference/eval_data/gaia_test.jsonl` - GAIA questions (requires approval)
- `inference/eval_data/hle_test.jsonl` - HLE questions (requires approval)
- `inference/eval_data/browsecomp_test.jsonl` - BrowseComp questions

**Note:** GAIA and HLE require gated access from HuggingFace.

---

### 3. `run_evaluation.py` - Automated Evaluation
Runs LLM-as-judge evaluation on inference results.

```bash
# Evaluate all available results
python scripts/run_evaluation.py --all

# Evaluate specific benchmark
python scripts/run_evaluation.py --benchmark hle --input_file outputs/.../iter1.jsonl
python scripts/run_evaluation.py --benchmark gaia --input_folder outputs/.../gaia_test
```

**Options:**
- `--all`: Evaluate all results found in output directory
- `--benchmark NAME`: Specific benchmark (hle, gaia, browsecomp)
- `--input_file PATH`: Input file for HLE
- `--input_folder PATH`: Input folder for GAIA/BrowseComp
- `--output_dir PATH`: Where to search for results (default: outputs)

**Requirements:**
- Environment variables: `API_KEY`, `OPENAI_API_KEY`, `Qwen2_5_7B_PATH`
- Completed inference results

**Outputs:**
- `*.eval_details.jsonl` - Per-question evaluation details
- `*.report.json` - Aggregate metrics and statistics
- `summary.jsonl` - Evaluation summaries

---

### 4. `generate_report.py` - Performance Report
Generates comprehensive markdown report from evaluation results.

```bash
python scripts/generate_report.py
python scripts/generate_report.py --output_dir outputs --report_file MY_REPORT.md
```

**Options:**
- `--output_dir PATH`: Directory with evaluation results (default: outputs)
- `--report_file PATH`: Output report filename (default: PERFORMANCE_REPORT.md)

**Outputs:**
- Markdown report with:
  - Overall accuracy summary
  - Per-benchmark detailed metrics
  - Tool usage statistics
  - Comparison with published baselines
  - Recommendations

---

### 5. `run_pipeline.sh` - Full Pipeline Orchestrator
Runs the complete evaluation pipeline in one command.

```bash
# Test with 5 questions
bash scripts/run_pipeline.sh --benchmark test_small

# Run single benchmark
bash scripts/run_pipeline.sh --benchmark browsecomp

# Run all benchmarks
bash scripts/run_pipeline.sh --all
```

**Pipeline steps:**
1. ✓ Verify environment and API keys
2. ✓ Run inference on specified benchmarks
3. ✓ Evaluate results with LLM-as-judge
4. ✓ Generate comprehensive report

**Options:**
- `--benchmark NAME`: Run specific benchmark (test_small, gaia, hle, browsecomp)
- `--all`: Run all available benchmarks

**Duration:**
- `test_small`: ~5-10 minutes
- Single benchmark: ~3-4 hours
- All benchmarks: ~10-15 hours

---

## 🚀 Quick Start Workflow

### Step 1: Verify Setup
```bash
python scripts/verify_setup.py
```
Fix any issues reported before proceeding.

### Step 2: Prepare Datasets
```bash
python scripts/download_and_prepare_datasets.py --sample_size 70
```
Request GAIA/HLE access if needed.

### Step 3: Run Test
```bash
# Update DATASET in .env to: inference/eval_data/test_small.jsonl
bash inference/run_react_infer_openrouter.sh
```

### Step 4: Run Full Evaluation
```bash
# Option A: Run everything automatically
bash scripts/run_pipeline.sh --benchmark browsecomp

# Option B: Run steps manually
bash inference/run_react_infer_openrouter.sh    # 1. Inference
python scripts/run_evaluation.py --all           # 2. Evaluate
python scripts/generate_report.py                # 3. Report
```

---

## 📊 Expected Workflow Timeline

| Phase | Duration | What Happens |
|-------|----------|--------------|
| Setup Verification | 2-5 min | Check environment, API keys, datasets |
| Dataset Preparation | 5-15 min | Download and format benchmark data |
| Small Test (5 Q) | 5-10 min | Verify everything works end-to-end |
| Inference (70 Q) | 2-4 hours | Model answers questions using tools |
| Evaluation | 30-60 min | LLM judges answer quality |
| Report Generation | 1-2 min | Compile results into markdown |

**Total for medium-scale**: ~3-6 hours per benchmark

---

## 🔧 Troubleshooting

### "Module not found" errors
```bash
# Make sure conda environment is activated
conda activate react_infer_env

# Reinstall dependencies if needed
pip install -r requirements.txt
```

### "API key not configured" errors
```bash
# Check .env file has actual keys (not placeholders)
cat .env | grep API_KEY

# Make sure .env is loaded
source .env
```

### "Dataset not found" errors
```bash
# Re-run dataset preparation
python scripts/download_and_prepare_datasets.py

# For GAIA/HLE, check if you have HuggingFace access
huggingface-cli whoami
```

### Evaluation script errors
```bash
# Make sure environment variables are exported
export API_KEY=your_key
export OPENAI_API_KEY=your_key
export OPENAI_API_BASE=https://api.openai.com/v1

# Then run evaluation
python scripts/run_evaluation.py --all
```

---

## 📁 Output File Structure

After running the pipeline, you'll have:

```
DeepResearch/
├── outputs/
│   └── Alibaba-NLP--Tongyi-DeepResearch-30B-A3B/
│       ├── test_small/
│       │   ├── iter1.jsonl                    # Predictions
│       │   ├── iter1.eval_details.jsonl       # Evaluation details
│       │   └── iter1.report.json              # Metrics
│       ├── gaia_test/
│       │   ├── iter1.jsonl
│       │   └── summary.jsonl
│       ├── hle_test/
│       │   ├── iter1.jsonl
│       │   ├── iter1.eval_details.jsonl
│       │   └── iter1.report.json
│       └── browsecomp_test/
│           ├── iter1.jsonl
│           └── summary.jsonl
└── PERFORMANCE_REPORT.md                      # Final report
```

---

## 💡 Pro Tips

1. **Start small**: Always test with `test_small.jsonl` first (5 questions, ~10 min)

2. **Monitor costs**: Each API call costs money. Track usage:
   - OpenRouter dashboard: https://openrouter.ai/activity
   - OpenAI dashboard: https://platform.openai.com/usage

3. **Save intermediate results**: Evaluation scripts can re-run on existing inference outputs without re-running inference

4. **Parallel benchmarks**: Can run multiple benchmarks in parallel on different machines/API keys

5. **Resume failed runs**: If inference crashes, it should resume from last checkpoint (check output folder)

---

## 🆘 Need Help?

- **Setup issues**: See `SETUP_COMPLETE.md`
- **API keys**: See `API_KEYS_SETUP.md`
- **Architecture**: See `CLAUDE.md`
- **Project FAQ**: See `FAQ.md`
- **GitHub Issues**: https://github.com/Alibaba-NLP/DeepResearch/issues
