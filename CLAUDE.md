# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tongyi DeepResearch is a 30.5B parameter Mixture-of-Experts (MoE) model designed for long-horizon, deep information-seeking tasks. The project includes:
- ReAct-based agentic inference system with tool use
- Family of WebAgent research implementations
- Synthetic data generation and training pipelines
- Benchmark evaluation suite

## Environment Setup

**Python Version**: 3.10.0 (strict requirement - other versions may cause dependency issues)

Always use conda for environment management:

```bash
# Create environment
conda create -n react_infer_env python=3.10.0
conda activate react_infer_env

# Install dependencies
pip install -r requirements.txt
```

**Environment Configuration**:
1. Copy `.env.example` to `.env`: `cp .env.example .env`
2. Configure required API keys:
   - `SERPER_KEY_ID`: Web search (https://serper.dev/)
   - `JINA_API_KEYS`: Web page reading (https://jina.ai/)
   - `API_KEY/API_BASE`: OpenAI-compatible API for summarization
   - `DASHSCOPE_API_KEY`: File parsing (https://dashscope.aliyun.com/)
   - `SANDBOX_FUSION_ENDPOINT`: Python sandbox (https://github.com/bytedance/SandboxFusion)
   - `MODEL_PATH`: Path to model weights
   - `DATASET`: Path to evaluation data file
   - `OUTPUT_PATH`: Results output directory

## Running Inference

**Standard ReAct Inference**:
```bash
# 1. Configure run_react_infer.sh with MODEL_PATH, DATASET, OUTPUT_PATH
# 2. Run the script
bash inference/run_react_infer.sh
```

The script will:
1. Start 8 vLLM servers on ports 6001-6008 (one per GPU)
2. Wait for all servers to be ready
3. Run parallel inference with `run_multi_react.py`

**Single Question Testing**:
```bash
conda activate react_infer_env
cd inference
python -u run_multi_react.py \
  --dataset eval_data/test.jsonl \
  --output ./outputs \
  --max_workers 30 \
  --model $MODEL_PATH \
  --temperature 0.85 \
  --presence_penalty 1.1
```

## Evaluation

**HLE Benchmark**:
```bash
export API_KEY=your_api_key
export BASE_URL=your_base_url
cd evaluation
python evaluate_hle_official.py --input_fp your_input_folder --model_path your_qwen_model_path
```

**Other Benchmarks** (DeepSearch, BrowseComp, GAIA, etc.):
```bash
export OPENAI_API_KEY=your_openai_key
export OPENAI_API_BASE=your_openai_base
export API_KEY=your_api_key
export BASE_URL=your_base_url
export Qwen2_5_7B_PATH=your_qwen_path
cd evaluation
python evaluate_deepsearch_official.py --input_fp your_input_folder --dataset your_dataset
```

## Architecture Overview

### Core Components

**1. Inference Layer** (`/inference/`)
- `react_agent.py`: MultiTurnReactAgent implementing the ReAct loop
- `run_multi_react.py`: Parallel execution orchestrator
- `run_react_infer.sh`: Server startup and inference launcher
- `tool_*.py`: Tool implementations (search, visit, file, python, scholar)
- `prompt.py`: System and user prompt templates

**2. WebAgent Family** (`/WebAgent/`)
Research agents with progressive capabilities:
- WebWalker: Foundation web traversal benchmark
- WebDancer: Native agentic model with SFT+RL training
- WebSailor/WebSailor-V2: Extended reasoning + scalable RL
- WebWatcher: Vision-language multimodal agent
- WebResearcher/WebResummer/WebWeaver/WebLeaper: Advanced reasoning patterns

Each WebAgent has independent codebase but shares ReAct architecture.

**3. Agent Training** (`/Agent/`)
- AgentFounder: Continual pre-training methodology
- AgentScaler: Scaling strategies via environment interaction

**4. Evaluation** (`/evaluation/`)
- LLM-as-judge implementations for various benchmarks
- Structured output parsing with confidence scoring

### ReAct Agent Architecture

**Execution Flow**:
```
User Question
  ↓
System Prompt + Tool Schemas (JSON) + User Message
  ↓
Loop (max 100 rounds or token limit):
  ├─ LLM generates: <think>reasoning</think><tool_call>{json}</tool_call>
  ├─ Parse tool call → Execute tool → Format <tool_response>
  ├─ Append to conversation history
  ├─ Check: <answer> tag? → Extract and return
  └─ Check: token/round/time limits? → Force answer or error
  ↓
Result Extraction
```

**Key Patterns**:
- XML-tagged structured output: `<think>`, `<tool_call>`, `<tool_response>`, `<answer>`
- Tools specified as JSON within `<tool_call>` tags
- Special format for Python tool: empty params `{}` + code in `<code>` tag
- Token counting with model-specific tokenizers (max 110K context)
- Graceful degradation: force answer on limits, error handling with retries

**Multi-Server Architecture**:
- 8 vLLM servers on different ports for parallel execution
- Round-robin + sticky assignment per question
- Enables concurrent processing with shared model instances

### Tool System

All tools inherit from `BaseTool` (qwen-agent) with standardized `call(params, **kwargs)` interface:

1. **Search**: Google search via Serper API, batched queries, language detection
2. **Visit**: Webpage extraction via Jina + LLM summarization (goal-directed)
3. **PythonInterpreter**: Sandboxed execution via SandboxFusion endpoints
4. **FileParser**: Multi-format support (PDF/DOCX/PPTX/CSV/XLSX/video/audio)
5. **Scholar**: Google Scholar academic search

**Tool Registration**:
```python
@register_tool('tool_name', allow_overwrite=True)
class ToolName(BaseTool):
    def call(self, params: str, **kwargs) -> str:
        # Implementation
        return result
```

### Dataset Format

Supports JSON and JSONL formats with `question` and `answer` fields:

**JSONL** (recommended):
```jsonl
{"question": "What is quantum computing?", "answer": "reference answer"}
{"question": "Analyze report.pdf findings", "answer": "reference answer"}
```

**File References**:
- Prepend filename to question: `"report.pdf What are the key findings?"`
- Place files in `eval_data/file_corpus/`

### Context Window Management

- Model supports 128K context, agent uses max 110K (with overhead)
- Token counting with AutoTokenizer from model path
- Dynamic truncation for long tool responses
- File content compression for context efficiency
- Force answer generation when approaching token limit

### Parallel Execution

- ThreadPoolExecutor with configurable `max_workers` (default: 30)
- Thread-safe file writing with locks per rollout
- Multiple rollouts (Pass@K) for robustness
- Question-level parallelism across vLLM servers

## Testing

No formal test suite provided. Testing approach:
1. Create small eval dataset: `eval_data/test.jsonl`
2. Run inference with single worker for debugging
3. Check output JSON files for reasoning traces
4. Validate tool calls and responses in output

## Common Issues

**Dependency conflicts**: Use Python 3.10.0 exactly as specified
**vLLM server fails**: Check GPU availability, CUDA setup, port conflicts
**Tool timeouts**: Increase timeout in tool implementation or check API credentials
**Token limit errors**: Reduce conversation history or file sizes
**API rate limits**: Configure proper API keys and check quotas

## Code Style

- Uses qwen-agent framework for agent abstractions
- XML tags for structured LLM output parsing
- Async/await for I/O-bound operations (file parsing, web requests)
- Environment variables via `.env` file (never commit secrets)
- Thread-safe parallel execution patterns

## OpenRouter API Usage

To use OpenRouter instead of local vLLM:

1. Modify `inference/react_agent.py` in `call_server()` function:
   - Set API key and base URL to OpenRouter credentials
   - Change model name to `alibaba/tongyi-deepresearch-30b-a3b`
   - Uncomment lines 88-90 for reasoning content concatenation

## Related Papers

The WebAgent family has 12 associated papers (see README). Each WebAgent subdirectory may have specific requirements - consult individual READMEs.
