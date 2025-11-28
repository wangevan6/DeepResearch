# HLE Evaluation Progress Log

**Model**: Tongyi-DeepResearch-30B-A3B\
**Dataset**: HLE Text-Only (2,158 questions)\
**Started**: 2025-11-13\
**Status**: 🟡 Ready to Execute

***

## API Provider Dashboard Links (Check Quota/Usage)

| Service | Purpose | Dashboard Link |
|---------|---------|----------------|
| **OpenRouter** | Model inference (DeepResearch-30B) | [https://openrouter.ai/settings/credits](https://openrouter.ai/settings/credits) |
| **Serper** | Web search (Google Search) | [https://serper.dev/dashboard](https://serper.dev/dashboard) |
| **Jina** | Web page reading/extraction | [https://jina.ai/dashboard](https://jina.ai/) |
| **Dashscope** | File parsing (PDF, video, etc.) | [https://dashscope.console.aliyun.com/](https://dashscope.console.aliyun.com/) |
| **SandboxFusion** | Python code execution | Local Docker (http://localhost:8080) |

**Estimated API usage per question**: ~20 API calls (search + visit + model inference)

***

## TL;DR: Run Everything in 3 Steps

### Step 1: Test API Keys
```bash
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && ~/miniconda3/envs/react_infer_env/bin/python test_api_keys.py"
```

### Step 2: Run Test Inference (10 questions, ~10 min)
```bash
wsl.exe -d Ubuntu-22.04 bash -c ". ~/miniconda3/etc/profile.d/conda.sh && conda activate react_infer_env && cd /mnt/c/Users/user/Projects/DeepResearch && bash inference/run_hle_inference.sh test"
```

### Step 3: Run Full Inference (2158 questions, ~4-5 hours)
```bash
wsl.exe -d Ubuntu-22.04 bash -c ". ~/miniconda3/etc/profile.d/conda.sh && conda activate react_infer_env && cd /mnt/c/Users/user/Projects/DeepResearch && bash inference/run_hle_inference.sh full"
```

### Step 4: Run Evaluation (after inference completes)
```bash
# For test run
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch/evaluation && ~/miniconda3/envs/react_infer_env/bin/python evaluate_hle_official.py --input_fp ../outputs/hle_test/Tongyi-DeepResearch-30B-A3B_sglang/eval_data/hle_test_small.jsonl/iter1.jsonl --tokenizer_path Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"

# For full run
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch/evaluation && ~/miniconda3/envs/react_infer_env/bin/python evaluate_hle_official.py --input_fp ../outputs/hle_evaluation/Tongyi-DeepResearch-30B-A3B_sglang/eval_data/hle_text_only.jsonl/iter1.jsonl --tokenizer_path Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
```

### Monitor Progress
```bash
# Check how many questions completed (test)
wsl.exe -d Ubuntu-22.04 bash -c "wc -l /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_test/Tongyi-DeepResearch-30B-A3B_sglang/eval_data/hle_test_small.jsonl/iter1.jsonl 2>/dev/null || echo '0 (not started)'"

# Check how many questions completed (full)
wsl.exe -d Ubuntu-22.04 bash -c "wc -l /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_evaluation/Tongyi-DeepResearch-30B-A3B_sglang/eval_data/hle_text_only.jsonl/iter1.jsonl 2>/dev/null || echo '0 (not started)'"
```

### Detailed Status Check Commands

```bash
# 1. Find all output files
wsl.exe -d Ubuntu-22.04 bash -c "find /mnt/c/Users/user/Projects/DeepResearch/outputs -name '*.jsonl' -o -name '*.log' 2>/dev/null"

# 2. View last few results (check predictions)
wsl.exe -d Ubuntu-22.04 bash -c "tail -3 /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_test/Tongyi-DeepResearch-30B-A3B_sglang/eval_data/hle_test_small.jsonl/iter1.jsonl"

# 3. Check inference log for errors (test run)
wsl.exe -d Ubuntu-22.04 bash -c "tail -50 /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_test/inference.log"

# 4. Check inference log for errors (full run)
wsl.exe -d Ubuntu-22.04 bash -c "tail -50 /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_evaluation/inference.log"

# 5. Count errors in results
wsl.exe -d Ubuntu-22.04 bash -c "grep -c '\"error\":' /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_test/Tongyi-DeepResearch-30B-A3B_sglang/eval_data/hle_test_small.jsonl/iter1.jsonl"
```

***

## Quick Reference: Essential Commands

### Activate Python Environment (REQUIRED FIRST!)

See C:/Users/user/Projects/DeepResearch/.trae/documents/WSL\_ENVIRONMENT.md for full details.

**Key paths:**

* Conda: `/home/evan_hero_linux/miniconda3`

* Python: `~/miniconda3/envs/react_infer_env/bin/python` (Python 3.10.0)

* Project: `/mnt/c/Users/user/Projects/DeepResearch`

**Activate environment:**

```bash
# Method 1: Direct python (fastest)
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/python --version"

# Method 2: With conda activation
wsl.exe -d Ubuntu-22.04 bash -c ". ~/miniconda3/etc/profile.d/conda.sh && conda activate react_infer_env && python --version"

# Method 3: Login shell
wsl.exe -d Ubuntu-22.04 bash -l -c "conda activate react_infer_env && python --version"
```

Expected: `Python 3.10.0`

***

## Phase 1: Environment Setup

### 1.1 Start Docker Desktop

```powershell
docker ps
```

### 1.2 Pull SandboxFusion Image

```bash
docker pull vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

### 1.3 Run SandboxFusion Container

```bash
docker run -d --rm --privileged -p 8080:8080 --name sandbox-fusion vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

Verify:

```bash
docker ps | grep sandbox-fusion
```

### 1.4 Test Container

```bash
curl -X POST http://localhost:8080/run_code -H "Content-Type: application/json" -d "{\"code\":\"print('Hello!')\", \"language\":\"python\"}"
```

### 1.5 Test API Keys

```bash
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && ~/miniconda3/envs/react_infer_env/bin/python test_api_keys_hle.py"
```

***

## Phase 2: Test Run (10 Questions)

### 2.1 Verify Test Dataset

```bash
wsl.exe -d Ubuntu-22.04 bash -c "wc -l /mnt/c/Users/user/Projects/DeepResearch/inference/eval_data/hle_test_small.jsonl"
```

Expected: 10 lines

### 2.2 Run Test Inference

```bash
wsl.exe -d Ubuntu-22.04 bash -l -c "cd /mnt/c/Users/user/Projects/DeepResearch && conda activate react_infer_env && bash inference/run_hle_inference.sh test"
```

Duration: \~20-30 minutes\
Output: `outputs/hle_test/iter1.jsonl`

### 2.3 Monitor Progress

```bash
wsl.exe -d Ubuntu-22.04 bash -c "wc -l /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_test/iter1.jsonl"
```

### 2.4 Run Test Evaluation

```bash
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch/evaluation && ~/miniconda3/envs/react_infer_env/bin/python evaluate_hle_official.py --input_fp ../outputs/hle_test/iter1.jsonl --tokenizer_path Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
```

### 2.5 View Test Results

```bash
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/python -m json.tool /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_test/iter1.report.json"
```

***

## Phase 3: Full Evaluation (2,158 Questions)

### 3.1 Verify Full Dataset

```bash
wsl.exe -d Ubuntu-22.04 bash -c "wc -l /mnt/c/Users/user/Projects/DeepResearch/inference/eval_data/hle_text_only.jsonl"
```

Expected: 2158 lines

### 3.2 Run Full Inference

```bash
wsl.exe -d Ubuntu-22.04 bash -l -c "cd /mnt/c/Users/user/Projects/DeepResearch && conda activate react_infer_env && bash inference/run_hle_inference.sh full"
```

Config: 1 rollout, 30 workers, \~4-5 hours\
Output: `outputs/hle_evaluation/iter1.jsonl`

### 3.3 Monitor Progress

```bash
wsl.exe -d Ubuntu-22.04 bash -c "wc -l /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_evaluation/iter1.jsonl"
```

### 3.4 Run Full Evaluation

```bash
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch/evaluation && ~/miniconda3/envs/react_infer_env/bin/python evaluate_hle_official.py --input_fp ../outputs/hle_evaluation/iter1.jsonl --tokenizer_path Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
```

### 3.5 View Final Results

```bash
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/python -m json.tool /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_evaluation/iter1.report.json"
```

***

## Troubleshooting

### Docker Issues

```bash
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && ~/miniconda3/envs/react_infer_env/bin/python test_api_keys_hle.py"
```

### Python Issues

```bash
# Verify Python version
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/python --version"
```

### Check Progress

```bash
# Test run
wsl.exe -d Ubuntu-22.04 bash -c "wc -l /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_test/iter1.jsonl"

# Full run
wsl.exe -d Ubuntu-22.04 bash -c "wc -l /mnt/c/Users/user/Projects/DeepResearch/outputs/hle_evaluation/iter1.jsonl"
```

***

## Execution Log

| Phase         | Status | Started | Completed | Notes  |
| ------------- | ------ | ------- | --------- | ------ |
| Setup         | ⏳      | \_\_\_  | \_\_\_    | \_\_\_ |
| Test (10 Q)   | ⏳      | \_\_\_  | \_\_\_    | \_\_\_ |
| Full (2158 Q) | ⏳      | \_\_\_  | \_\_\_    | \_\_\_ |

***

## References

* HLE Dataset: <https://huggingface.co/datasets/cais/hle>

* Model: <https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B>

* WSL Setup: .trae/documents/WSL\_ENVIRONMENT.md

***

## API Validation Results (2025-11-13)

### Test Command

```bash
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && ~/miniconda3/envs/react_infer_env/bin/python test_api_keys_hle.py"
```

### Results Summary

| API Service | Status      | Error Code | Issue                         | Fix Required             |
| ----------- | ----------- | ---------- | ----------------------------- | ------------------------ |
| OpenRouter  | ✅ WORKING   | -          | Fixed with max\_tokens=99482  | None                     |
| Serper      | ❌ FAILED    | 403        | Invalid API key (placeholder) | Replace with real key    |
| Jina        | ❌ FAILED    | 402        | Insufficient balance          | Add credits              |
| Dashscope   | ⚠️ OPTIONAL | 400        | Bad request                   | Not needed for text-only |

### Detailed Error Analysis

#### 1. OpenRouter API - ✅ RESOLVED

**Status**: WORKING
**Model**: alibaba/tongyi-deepresearch-30b-a3b
**Solution**: Set `max_tokens=99482` in API calls
**Validation**: Successfully tested with reasoning-enabled queries

**Test Code**:

```python
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="sk-or-v1-..."
)

response = client.chat.completions.create(
  model="alibaba/tongyi-deepresearch-30b-a3b",
  messages=[{"role": "user", "content": "test query"}],
  max_tokens=99482,
  extra_body={"reasoning": {"enabled": True}}
)
```

#### 2. Serper API - ❌ INVALID KEY

**Endpoint**: <https://google.serper.dev/search>
**Error**: `{"message":"Unauthorized.","statusCode":403}`
**Cause**: API key in .env is set to placeholder "your\_key"

**Test Command**:

```python
import requests
url = 'https://google.serper.dev/search'
headers = {'X-API-KEY': serper_key, 'Content-Type': 'application/json'}
data = {'q': 'test query', 'num': 5}
response = requests.post(url, headers=headers, json=data)
# Returns: 403 Forbidden
```

**How to Fix**:

1. Go to <https://serper.dev/>
2. Sign up or log in
3. Get API key from dashboard
4. Update .env: `SERPER_KEY_ID=your_actual_key_here`
5. Re-test with command above

**Used By**: Search tool in ReAct agent (critical for web search)

#### 3. Jina API - ❌ INSUFFICIENT BALANCE

**Endpoint**: <https://r.jina.ai/>
**Error**: `{"code":402,"message":"Account balance not enough to run this query, please recharge."}`
**Cause**: Jina account has no credits

**Test Command**:

```python
import requests
url = 'https://r.jina.ai/https://example.com'
headers = {'Authorization': f'Bearer {jina_key}', 'X-With-Generated-Alt': 'true'}
response = requests.get(url, headers=headers)
# Returns: 402 Payment Required
```

**How to Fix**:

1. Go to <https://jina.ai/>
2. Log in to your account
3. Add credits/payment method
4. Or get a new API key with free tier
5. Update .env: `JINA_API_KEYS=your_key_with_credits`
6. Re-test with command above

**Used By**: Visit tool in ReAct agent (critical for reading web pages)

#### 4. Dashscope API - ⚠️ OPTIONAL FOR HLE TEXT-ONLY

**Endpoint**: Dashscope file parsing
**Error**: 400 Bad Request
**Cause**: Configuration issue (not investigated further)
**Impact**: None for HLE text-only evaluation (only needed for file-based questions)

### Impact on HLE Evaluation

**Without working Serper and Jina APIs**:

* ❌ Agent cannot search the web

* ❌ Agent cannot visit and read web pages

* ✅ Agent can still reason with internal knowledge

* ✅ Python interpreter will work (SandboxFusion)

**Expected Performance Impact**:

* HLE questions often require external knowledge

* Without web tools, accuracy will be significantly reduced

* Estimated: 20-40% accuracy (vs 49-52% with tools)

**Recommendation**: Fix Serper and Jina before running full evaluation

### Re-validation Commands

**Test individual APIs**:

```bash
# Test Serper
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && ~/miniconda3/envs/react_infer_env/bin/python -c \"
import os, requests
from dotenv import load_dotenv
load_dotenv()
r = requests.post('https://google.serper.dev/search',
    headers={'X-API-KEY': os.getenv('SERPER_KEY_ID'), 'Content-Type': 'application/json'},
    json={'q': 'test', 'num': 1})
print(f'Serper: {r.status_code} - {r.text}')
\""

# Test Jina
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && ~/miniconda3/envs/react_infer_env/bin/python -c \"
import os, requests
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('JINA_API_KEYS').split(',')[0] if os.getenv('JINA_API_KEYS') else ''
r = requests.get('https://r.jina.ai/https://example.com',
    headers={'Authorization': f'Bearer {key}'})
print(f'Jina: {r.status_code} - {r.text[:200]}')
\""
```

### Next Steps

1. ✅ OpenRouter: Working - no action needed
2. ❌ Serper: Update API key in .env file
3. ❌ Jina: Add credits or get new key
4. 🔄 Re-run validation: `python test_api_keys_hle.py`
5. ✅ Proceed to test run when all APIs pass

***

## Execution Status Update

**Last Updated**: 2025-11-13 21:57 UTC

| Phase          | Status     | Started | Completed | Notes                                  |
| -------------- | ---------- | ------- | --------- | -------------------------------------- |
| Docker Setup   | ✅ DONE     | 21:50   | 21:50     | Container running on port 8080         |
| API Validation | ⚠️ PARTIAL | 21:52   | 21:57     | OpenRouter OK, Serper/Jina need fixing |
| Test (10 Q)    | ⏳ WAITING  | -       | -         | Blocked on API keys                    |
| Full (2158 Q)  | ⏳ WAITING  | -       | -         | Blocked on API keys                    |

**Current Blocker**: Serper and Jina API keys need to be fixed before proceeding
