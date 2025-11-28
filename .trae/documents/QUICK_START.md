# DeepResearch - Quick Start Guide

## 🎯 Goal
Evaluate the Tongyi DeepResearch model on GAIA, HLE, and BrowseComp benchmarks using OpenRouter API (no local GPUs needed).

## ⚡ Quick Start (30 seconds)

```bash
# 1. Activate environment
conda activate react_infer_env

# 2. Verify setup
python scripts/verify_setup.py

# 3. Add your API keys to .env (see below)

# 4. Test with 5 questions (~10 min)
# Edit .env: DATASET=inference/eval_data/test_small.jsonl
bash inference/run_react_infer_openrouter.sh

# 5. Run full benchmark (~3-4 hours per benchmark)
bash scripts/run_pipeline.sh --benchmark browsecomp
```

## 📝 Setup Checklist

### ✅ Already Done
- [x] Conda environment created
- [x] Dependencies installed
- [x] Code configured for OpenRouter API
- [x] Test datasets prepared
- [x] Helper scripts created

### ⏳ You Need To Do
- [ ] **Get API keys** (30-60 minutes)
- [ ] **Add keys to `.env`** (5 minutes)
- [ ] **Run verification** (2 minutes)
- [ ] **Test with 5 questions** (10 minutes)
- [ ] **Request GAIA/HLE access** (1-2 days wait)
- [ ] **Run full evaluations** (8-12 hours)

---

## 🔑 Step 1: Get API Keys (REQUIRED)

You need these API keys. See `API_KEYS_SETUP.md` for detailed instructions.

### Required Keys:

1. **OpenRouter** (https://openrouter.ai/)
   - Sign up → API Keys → Create
   - Cost: ~$20-40 per benchmark

2. **Serper** (https://serper.dev/)
   - Sign up → API Key
   - Free tier: 2,500 searches/month

3. **Jina AI** (https://jina.ai/reader)
   - Sign up → API Key
   - Free tier: 1M tokens/month

4. **OpenAI or Compatible** (https://platform.openai.com/)
   - For summarization
   - Cost: ~$5-15 per benchmark

5. **Dashscope** (https://dashscope.aliyun.com/)
   - For file parsing (GAIA)
   - Cost: ~$5-10

---

## 📝 Step 2: Configure `.env` File

Open `.env` in a text editor and replace the placeholder values:

```bash
# Open in nano (or use your preferred editor)
nano .env

# Replace these values:
OPENROUTER_API_KEY=sk-or-v1-...              # Your actual OpenRouter key
SERPER_KEY_ID=your_actual_serper_key         # Your actual Serper key
JINA_API_KEYS=your_actual_jina_key          # Your actual Jina key
API_KEY=sk-...                               # Your actual OpenAI key
DASHSCOPE_API_KEY=your_actual_dashscope_key # Your actual Dashscope key

# Keep these as-is:
API_BASE=https://api.openai.com/v1
SUMMARY_MODEL_NAME=gpt-4o-mini
```

Save and close (Ctrl+X, then Y, then Enter in nano).

---

## ✓ Step 3: Verify Setup

```bash
conda activate react_infer_env
python scripts/verify_setup.py
```

**Expected output:**
```
Passed: 7/7 checks
✓ Python Version
✓ Conda Env
✓ Packages
✓ Env File
✓ Directories
✓ Scripts
✓ Datasets

🎉 Setup verification passed!
```

If any checks fail, follow the instructions to fix them.

---

## 🧪 Step 4: Run Test (5 Questions)

Test your setup with 5 simple questions (~10 minutes):

```bash
# 1. Update DATASET in .env
nano .env
# Change: DATASET=inference/eval_data/test_small.jsonl

# 2. Run test
cd inference
bash run_react_infer_openrouter.sh

# 3. Check results
cat ../outputs/Alibaba-NLP--Tongyi-DeepResearch-30B-A3B/test_small/iter1.jsonl | head -50
```

**What to expect:**
- Script will process 5 questions
- Each question takes 1-3 minutes
- Results saved to `outputs/` directory
- You should see reasoning traces and answers

**If it works:** Proceed to Step 5!
**If it fails:** Check error messages for API key issues.

---

## 📊 Step 5: Run Full Benchmarks

### Option A: Automated Pipeline (Recommended)

Run everything with one command:

```bash
# BrowseComp (70 questions, ~2-3 hours)
bash scripts/run_pipeline.sh --benchmark browsecomp

# After GAIA/HLE approval:
bash scripts/run_pipeline.sh --benchmark gaia
bash scripts/run_pipeline.sh --benchmark hle

# Or run all at once:
bash scripts/run_pipeline.sh --all
```

The pipeline will:
1. ✓ Run inference
2. ✓ Evaluate with LLM-as-judge
3. ✓ Generate performance report
4. ✓ Save to `PERFORMANCE_REPORT.md`

### Option B: Manual Steps

```bash
# 1. Run inference
cd inference
bash run_react_infer_openrouter.sh

# 2. Evaluate results
cd ../
python scripts/run_evaluation.py --all

# 3. Generate report
python scripts/generate_report.py
```

---

## 📈 Step 6: Request GAIA/HLE Access (Optional)

Both GAIA and HLE require approval from HuggingFace:

### GAIA Dataset
1. Visit: https://huggingface.co/datasets/gaia-benchmark/GAIA
2. Click "Request Access"
3. Fill in form (mention research/evaluation)
4. Wait 1-2 days for approval
5. Login: `huggingface-cli login`
6. Re-run: `python scripts/download_and_prepare_datasets.py --datasets gaia`

### HLE Dataset
1. Visit: https://huggingface.co/datasets/cais/hle
2. Click "Request Access"
3. Fill in form
4. Wait 1-2 days
5. Re-run: `python scripts/download_and_prepare_datasets.py --datasets hle`

**You can proceed with BrowseComp while waiting for approval.**

---

## 📊 Step 7: Review Results

After evaluation completes, check your performance report:

```bash
cat PERFORMANCE_REPORT.md
```

**Report includes:**
- Overall accuracy across benchmarks
- Detailed metrics per benchmark
- Tool usage statistics
- Comparison with published results
- Recommendations for improvement

**Example results:**

| Benchmark | Questions | Your Score | Published |
|-----------|-----------|------------|-----------|
| BrowseComp | 70 | 65% | 68% |
| GAIA | 70 | 38% | 41% |
| HLE | 70 | 49% | 52% |

---

## 🕐 Time Estimates

| Task | Duration |
|------|----------|
| Get API keys | 30-60 min |
| Configure .env | 5 min |
| Verify setup | 2 min |
| Test run (5 Q) | 10 min |
| **Ready to evaluate!** | **~1 hour** |
| | |
| BrowseComp (70 Q) | 2-3 hours |
| GAIA (70 Q) | 3-4 hours |
| HLE (70 Q) | 3-4 hours |
| Evaluation | 30-60 min |
| **Full benchmarks** | **~10-15 hours** |

---

## 💰 Cost Estimates

For 70 questions per benchmark (210 total):

| Service | Cost |
|---------|------|
| OpenRouter (inference) | $60-120 |
| OpenAI (summarization) | $15-30 |
| Serper (search) | Free or $10 |
| Jina AI (web reading) | Free |
| Dashscope (file parsing) | $10-20 |
| OpenAI (evaluation judge) | $10-20 |
| **Total** | **$100-200** |

**For test run (5 Q):** ~$1-2

---

## 🐛 Common Issues

### "API key not configured"
- Check .env file has actual keys (not placeholders)
- Keys must not have quotes or extra spaces

### "Rate limit exceeded"
- Reduce MAX_WORKERS in .env from 5 to 2
- Wait a few minutes before retrying

### "Tool timeout"
- Check Serper and Jina keys are valid
- Test manually: `curl -H "X-API-KEY: your_key" https://serper.dev/health`

### "Module not found"
- Activate conda: `conda activate react_infer_env`
- Reinstall: `pip install -r requirements_cpu.txt`

---

## 📚 Additional Documentation

- **`SETUP_COMPLETE.md`** - Detailed setup guide
- **`API_KEYS_SETUP.md`** - How to get each API key
- **`scripts/README.md`** - All script documentation
- **`CLAUDE.md`** - Technical architecture details
- **`FAQ.md`** - Frequently asked questions

---

## 🆘 Need Help?

1. Check error messages carefully
2. Run verification: `python scripts/verify_setup.py`
3. Review relevant documentation above
4. Check GitHub issues: https://github.com/Alibaba-NLP/DeepResearch/issues

---

## ✅ Success Checklist

Before running full benchmarks, make sure:

- [ ] Verification script passes all checks
- [ ] Test run (5 questions) completes successfully
- [ ] API costs are acceptable for your budget
- [ ] You understand the time commitment (10-15 hours total)

**Once verified, you're ready to go! 🚀**

---

**Next Step:** Get your API keys and add them to `.env`, then run `python scripts/verify_setup.py`!
