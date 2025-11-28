# Complete GPU Server Setup for DeepResearch HLE Evaluation

## Overview

This guide covers the complete setup for running DeepResearch evaluation on a GPU server with 8 GPUs, including:
- Downloading the model and dataset
- Setting up 5 SandboxFusion instances
- Running 8 vLLM servers
- Running full HLE inference (hle_text_only - 1,423 questions)
- Running evaluation

---

## Phase 1: Environment Setup

### 1.1 Create Conda Environment

```bash
# Create environment with Python 3.10.0 (strict requirement)
conda create -n react_infer_env python=3.10.0 -y
conda activate react_infer_env
```

### 1.2 Clone Repository

```bash
git clone https://github.com/Alibaba-NLP/DeepResearch.git
cd DeepResearch
```

### 1.3 Install Dependencies

```bash
# Install PyTorch with CUDA 12.8 support
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Install other dependencies
pip install -r requirements.txt
```

---

## Phase 2: Download Model

### 2.1 Model Details

| Property | Value |
|----------|-------|
| Model | Tongyi-DeepResearch-30B-A3B |
| Parameters | 30.5B total (3.3B active per token) |
| Architecture | Mixture-of-Experts (MoE) |
| Context | 128K tokens |
| Format | SafeTensors, BF16 |

### 2.2 Download from HuggingFace

```bash
# Install huggingface-cli if not installed
pip install huggingface_hub

# Create model directory
mkdir -p /data/models

# Download model (requires ~60GB disk space)
huggingface-cli download Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --local-dir /data/models/Tongyi-DeepResearch-30B-A3B \
  --local-dir-use-symlinks False
```

### 2.3 Alternative: Download from ModelScope (China)

```bash
pip install modelscope
modelscope download --model iic/Tongyi-DeepResearch-30B-A3B \
  --local_dir /data/models/Tongyi-DeepResearch-30B-A3B
```

---

## Phase 3: Download Dataset

### 3.1 HLE Dataset (Text-Only)

The `hle_text_only.jsonl` file should already be in `inference/eval_data/`. If not:

```bash
# The dataset contains 1,423 text-only questions from HLE benchmark
# Format: {"question": "...", "answer": "..."}
ls inference/eval_data/hle_text_only.jsonl
```

### 3.2 Verify Dataset

```bash
# Check line count
wc -l inference/eval_data/hle_text_only.jsonl
# Expected: 1423

# Preview first entry
head -1 inference/eval_data/hle_text_only.jsonl | python -m json.tool
```

---

## Phase 4: Setup SandboxFusion (5 Instances)

### 4.1 Pull Docker Image

```bash
docker pull vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

### 4.2 Start 5 Sandbox Containers

```bash
# Start 5 sandbox instances on ports 8080-8084
for port in 8080 8081 8082 8083 8084; do
  docker run -d \
    --name sandbox_${port} \
    --rm \
    --privileged \
    -p ${port}:8080 \
    vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
  echo "Started sandbox on port ${port}"
done
```

### 4.3 Verify Sandboxes are Running

```bash
# Check all containers are running
docker ps | grep sandbox

# Test each endpoint
for port in 8080 8081 8082 8083 8084; do
  echo "Testing port ${port}..."
  curl -s -X POST http://localhost:${port}/run_code \
    -H "Content-Type: application/json" \
    -d '{"code":"print(1+1)", "language":"python"}' | head -c 100
  echo ""
done
```

### 4.4 Configure Environment Variable

```bash
# In .env file, set comma-separated endpoints
SANDBOX_FUSION_ENDPOINT=http://localhost:8080,http://localhost:8081,http://localhost:8082,http://localhost:8083,http://localhost:8084
```

---

## Phase 5: Configure Environment Variables

### 5.1 Copy and Edit .env File

```bash
cp .env.example .env
```

### 5.2 Required API Keys

Edit `.env` with your API keys:

```bash
# ============ REQUIRED FOR INFERENCE ============

# Web Search (https://serper.dev/) - $50 for 50K searches
SERPER_KEY_ID=your_serper_key

# Web Page Reading (https://jina.ai/) - Free tier available
JINA_API_KEYS=your_jina_key

# Summarization API (OpenAI-compatible)
API_KEY=your_openai_or_openrouter_key
API_BASE=https://openrouter.ai/api/v1
SUMMARY_MODEL_NAME=openai/gpt-4o-mini

# File Parsing (https://dashscope.aliyun.com/)
DASHSCOPE_API_KEY=your_dashscope_key

# Python Sandbox (configured in Phase 4)
SANDBOX_FUSION_ENDPOINT=http://localhost:8080,http://localhost:8081,http://localhost:8082,http://localhost:8083,http://localhost:8084

# ============ MODEL PATH ============
MODEL_PATH=/data/models/Tongyi-DeepResearch-30B-A3B

# ============ INFERENCE SETTINGS ============
TEMPERATURE=0.85
PRESENCE_PENALTY=1.1
MAX_WORKERS=30
ROLLOUT_COUNT=1
MAX_LLM_CALL_PER_RUN=100
```

---

## Phase 6: Start vLLM Servers (8 GPUs - A100 40GB)

### 6.1 Memory Considerations for A100 40GB

With 40GB VRAM, we need to optimize settings:
- The 30B MoE model has ~3.3B active parameters per token
- BF16 weights + KV cache must fit in 40GB
- Reduce max context length to 65K (instead of 128K)

### 6.2 Create vLLM Startup Script

Create `start_vllm_servers.sh`:

```bash
#!/bin/bash

MODEL_PATH=/data/models/Tongyi-DeepResearch-30B-A3B

echo "Starting 8 vLLM servers on A100 40GB GPUs..."

# Start 8 vLLM servers, one per GPU
# Settings optimized for A100 40GB:
#   --max-model-len 65536 (reduced from 128K to fit in 40GB)
#   --gpu-memory-utilization 0.92 (leave some headroom)
#   --enforce-eager (disable CUDA graphs to save memory)
for i in {0..7}; do
  port=$((6001 + i))
  CUDA_VISIBLE_DEVICES=$i vllm serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port $port \
    --disable-log-requests \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.92 \
    --enforce-eager \
    --dtype bfloat16 \
    &
  echo "Started vLLM server on GPU $i, port $port"
  sleep 5  # Stagger startup to avoid memory spikes
done

echo "Waiting for all servers to be ready..."

# Wait for all servers to be healthy (timeout 30 min)
for port in {6001..6008}; do
  timeout=1800
  elapsed=0
  while ! curl -s http://localhost:$port/v1/models > /dev/null 2>&1; do
    echo "Waiting for port $port... (${elapsed}s)"
    sleep 10
    elapsed=$((elapsed + 10))
    if [ $elapsed -ge $timeout ]; then
      echo "ERROR: Server on port $port failed to start"
      exit 1
    fi
  done
  echo "Server on port $port is ready"
done

echo "All 8 vLLM servers are running!"
```

**Note**: With 65K context, adjust `max_tokens` in react_agent.py from 110K to 60K.

### 6.3 Start Servers

```bash
chmod +x start_vllm_servers.sh
./start_vllm_servers.sh
```

### 6.4 Verify All Servers

```bash
# Check all servers are responding
for port in {6001..6008}; do
  echo "Port $port: $(curl -s http://localhost:$port/v1/models | head -c 50)"
done
```

---

## Phase 7: Run Full HLE Inference

### 7.1 Code Modifications for Self-Hosted vLLM

The current codebase uses HF Spaces or OpenRouter. For self-hosted vLLM, make these changes:

#### 7.1.1 Modify `inference/react_agent.py`

**Change 1: Replace API client initialization (around line 60-64)**

```python
# OLD CODE:
# use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"
# self.api_client = create_client(use_openrouter=use_openrouter)

# NEW CODE - Use local vLLM via OpenAI client:
from openai import OpenAI

class VLLMClient:
    """Client for local vLLM servers"""
    def __init__(self, ports=[6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008]):
        self.ports = ports
        self.current_idx = 0

    def get_client(self):
        port = self.ports[self.current_idx % len(self.ports)]
        self.current_idx += 1
        return OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

    def chat_completion(self, messages, temperature=0.85, top_p=0.95,
                        max_tokens=10000, presence_penalty=1.1, stop=None, **kwargs):
        client = self.get_client()
        response = client.chat.completions.create(
            model="Tongyi-DeepResearch-30B-A3B",  # vLLM uses this as identifier
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            stop=stop
        )
        return response

    def extract_content(self, response):
        return response.choices[0].message.content

# In __init__:
self.api_client = VLLMClient()
print("✅ API client initialized: Local vLLM")
```

**Change 2: Adjust token limit for 40GB GPUs (line 210)**

```python
# OLD:
# max_tokens = 110 * 1024

# NEW (for A100 40GB with 65K context):
max_tokens = 60 * 1024
```

#### 7.1.2 Alternative: Use Environment Variable Switch

Add to `.env`:
```bash
USE_LOCAL_VLLM=true
VLLM_PORTS=6001,6002,6003,6004,6005,6006,6007,6008
```

Then in `react_agent.py`:
```python
use_local_vllm = os.getenv("USE_LOCAL_VLLM", "false").lower() == "true"
if use_local_vllm:
    ports = [int(p) for p in os.getenv("VLLM_PORTS", "6001").split(",")]
    self.api_client = VLLMClient(ports=ports)
else:
    # Existing HF Spaces / OpenRouter code
    use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    self.api_client = create_client(use_openrouter=use_openrouter)
```

### 7.2 Create Full Inference Script

Create `run_hle_full.sh`:

```bash
#!/bin/bash

# Configuration
MODEL_PATH=/data/models/Tongyi-DeepResearch-30B-A3B
DATASET=eval_data/hle_text_only.jsonl
OUTPUT=../outputs/hle_full_evaluation
MAX_WORKERS=30
ROLLOUT_COUNT=1
TEMPERATURE=0.85
PRESENCE_PENALTY=1.1

cd inference

echo "=============================================="
echo "DeepResearch HLE Full Evaluation"
echo "=============================================="
echo "Dataset: $DATASET (1,423 questions)"
echo "Output: $OUTPUT"
echo "Workers: $MAX_WORKERS"
echo "Rollouts: $ROLLOUT_COUNT"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT"

# Run inference
python -u run_multi_react.py \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --max_workers $MAX_WORKERS \
  --model $MODEL_PATH \
  --temperature $TEMPERATURE \
  --presence_penalty $PRESENCE_PENALTY \
  --roll_out_count $ROLLOUT_COUNT \
  2>&1 | tee "$OUTPUT/inference.log"

echo "Inference completed!"
echo "Results saved to: $OUTPUT"
```

### 7.3 Run Inference

```bash
chmod +x run_hle_full.sh
./run_hle_full.sh
```

### 7.4 Monitor Progress

```bash
# Watch progress in real-time
tail -f outputs/hle_full_evaluation/inference.log

# Check number of completed questions
wc -l outputs/hle_full_evaluation/*/iter1.jsonl
```

### 7.5 Estimated Time and Cost

| Metric | Estimate |
|--------|----------|
| Questions | 1,423 |
| Avg rounds/question | 15-25 |
| Time per question | 1-3 minutes |
| **Total time** | **~30-60 hours** |
| LLM cost (self-hosted) | $0 |
| Tool API cost | ~$15-30 |

---

## Phase 8: Run Evaluation

### 8.1 Configure Evaluation API Keys

The evaluation uses LLM-as-judge. Add to `.env`:

```bash
# Judge model API (OpenRouter recommended)
API_KEY=your_openrouter_key
BASE_URL=https://openrouter.ai/api/v1

# Judge model (default: openai/o3-mini)
JUDGE_MODEL=openai/o3-mini
```

### 8.2 Run HLE Evaluation

```bash
cd evaluation

python evaluate_hle_official.py \
  --input_fp ../outputs/hle_full_evaluation/Tongyi-DeepResearch-30B-A3B_sglang/eval_data/hle_text_only/iter1.jsonl \
  --tokenizer_path /data/models/Tongyi-DeepResearch-30B-A3B
```

### 8.3 Evaluation Output

The script generates:

1. **Detailed results**: `iter1.eval_details.jsonl`
   - Per-question judgments with reasoning

2. **Summary report**: `iter1.report.json`
   ```json
   {
     "evaluated_nums": 1423,
     "valid_nums": 1400,
     "metrics": 52.5,
     "judge_model": "openai/o3-mini",
     "avg_prompt_tokens": 2500,
     "avg_completion_tokens": 1100,
     "is_answer_rate": 0.98
   }
   ```

### 8.4 View Results

```bash
# View summary metrics
cat outputs/hle_full_evaluation/*/iter1.report.json | python -m json.tool

# Count correct/incorrect
grep -c '"correct": true' outputs/hle_full_evaluation/*/iter1.eval_details.jsonl
grep -c '"correct": false' outputs/hle_full_evaluation/*/iter1.eval_details.jsonl
```

---

## Phase 9: Monitoring and Troubleshooting

### 9.1 Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check specific GPU
nvidia-smi -i 0
```

### 9.2 Monitor Sandbox Health

```bash
# Check sandbox container logs
docker logs sandbox_8080

# Restart failed sandbox
docker restart sandbox_8080
```

### 9.3 Common Issues

| Issue | Solution |
|-------|----------|
| vLLM OOM | Reduce `--gpu-memory-utilization` to 0.85 |
| Sandbox timeout | Increase timeout in tool_python.py |
| API rate limit | Reduce MAX_WORKERS or add delays |
| Stuck question | Check MAX_LLM_CALL_PER_RUN limit |

### 9.4 Resume from Checkpoint

The inference automatically resumes from where it left off:

```bash
# Just re-run the same command - it skips completed questions
./run_hle_full.sh
```

---

## Quick Reference: Complete Command Sequence

```bash
# 1. Setup environment
conda create -n react_infer_env python=3.10.0 -y
conda activate react_infer_env
pip install -r requirements.txt

# 2. Download model
huggingface-cli download Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --local-dir /data/models/Tongyi-DeepResearch-30B-A3B

# 3. Start 5 sandboxes
for port in 8080 8081 8082 8083 8084; do
  docker run -d --name sandbox_${port} --rm --privileged -p ${port}:8080 \
    vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
done

# 4. Configure .env (edit with your API keys)
cp .env.example .env
nano .env

# 5. Start 8 vLLM servers
./start_vllm_servers.sh

# 6. Run inference
cd inference
./run_hle_full.sh

# 7. Run evaluation
cd ../evaluation
python evaluate_hle_official.py \
  --input_fp ../outputs/hle_full_evaluation/*/iter1.jsonl \
  --tokenizer_path /data/models/Tongyi-DeepResearch-30B-A3B

# 8. View results
cat ../outputs/hle_full_evaluation/*/iter1.report.json
```

---

## Cost Summary

| Component | Self-Hosted | Notes |
|-----------|-------------|-------|
| LLM Inference | $0 | Using local vLLM |
| Serper (Search) | ~$15 | ~15K searches |
| Jina (Web Read) | ~$10 | ~10K page reads |
| Evaluation Judge | ~$5 | 1,423 questions × o3-mini |
| **Total** | **~$30** | For full HLE evaluation |

---

## Hardware Requirements (Your Setup: 8× A100 40GB)

| Component | Your Setup | Notes |
|-----------|------------|-------|
| GPUs | 8× A100 40GB | ✅ Sufficient with reduced context (65K) |
| Context Length | 65K tokens | Reduced from 128K to fit 40GB |
| RAM | 128GB+ | System memory |
| Disk | 100GB+ | Model (~60GB) + outputs |
| Network | Stable internet | For tool API calls |

### A100 40GB Specific Settings

- `--max-model-len 65536` (not 131072)
- `--gpu-memory-utilization 0.92`
- `--enforce-eager` (saves ~2GB per GPU)
- Max token limit in agent: 60K (not 110K)

### Files to Modify

| File | Change |
|------|--------|
| `inference/react_agent.py` | Add VLLMClient class, change token limit to 60K |
| `start_vllm_servers.sh` | Use A100-40GB optimized settings |
| `.env` | Add `USE_LOCAL_VLLM=true`, `VLLM_PORTS=...`|
