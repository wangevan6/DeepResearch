# API Keys Setup Guide

This file lists all the API keys you need to obtain and add to your `.env` file for running DeepResearch benchmarks.

## ✅ Required API Keys (Must Have)

### 1. OpenRouter API (for model inference)
- **Service**: OpenRouter
- **Website**: https://openrouter.ai/
- **Purpose**: Access to Tongyi-DeepResearch-30B-A3B model
- **Cost**: ~$0.80 per million input tokens, ~$3.20 per million output tokens
- **Setup**:
  1. Sign up at OpenRouter
  2. Get your API key from the API Keys page
  3. You'll add this to `inference/react_agent.py` later (not in .env)

### 2. Serper API (for web search)
- **Service**: Serper.dev
- **Website**: https://serper.dev/
- **Purpose**: Google search and Google Scholar integration
- **Cost**: Free tier: 2,500 searches/month, then $50/month for 5,000 searches
- **Setup**:
  1. Sign up at Serper.dev
  2. Get your API key
  3. Add to `.env` file:
     ```bash
     SERPER_KEY_ID=your_actual_serper_key
     ```

### 3. Jina AI API (for web page reading)
- **Service**: Jina AI Reader
- **Website**: https://jina.ai/reader
- **Purpose**: Extract and parse web page content
- **Cost**: Free tier: 1,000,000 tokens/month, then paid tiers available
- **Setup**:
  1. Sign up at Jina.ai
  2. Get your API key
  3. Add to `.env` file:
     ```bash
     JINA_API_KEYS=your_actual_jina_key
     ```

### 4. OpenAI API or Compatible (for summarization)
- **Service**: OpenAI or alternatives (Anthropic Claude, OpenRouter, etc.)
- **Website**: https://platform.openai.com/
- **Purpose**: Summarize long web pages
- **Cost**: Varies by provider
  - OpenAI GPT-4o-mini: ~$0.15/$0.60 per million tokens
  - OpenAI GPT-4o: ~$2.50/$10 per million tokens
- **Setup**:
  1. Get API key from your provider
  2. Add to `.env` file:
     ```bash
     API_KEY=your_actual_openai_key
     API_BASE=https://api.openai.com/v1
     SUMMARY_MODEL_NAME=gpt-4o-mini
     ```

### 5. Dashscope API (for file parsing - needed for GAIA)
- **Service**: Alibaba Cloud Dashscope
- **Website**: https://dashscope.aliyun.com/
- **Purpose**: Parse PDF, DOCX, PPTX, Excel files
- **Cost**: Varies, but reasonable for testing
- **Setup**:
  1. Sign up for Alibaba Cloud account
  2. Enable Dashscope service
  3. Get your API key
  4. Add to `.env` file:
     ```bash
     DASHSCOPE_API_KEY=your_actual_dashscope_key
     DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/api/v1
     VIDEO_MODEL_NAME=qwen-omni-turbo
     VIDEO_ANALYSIS_MODEL_NAME=qwen-plus-latest
     ```

## ⚠️ Optional API Keys (Can Skip for Initial Testing)

### 6. SandboxFusion (for Python code execution)
- **Service**: Self-hosted sandbox
- **GitHub**: https://github.com/bytedance/SandboxFusion
- **Purpose**: Execute Python code safely
- **Setup**: Complex - requires hosting your own sandbox service
- **Alternative**: Can disable Python interpreter tool for initial testing
- **To skip**: Leave as `SANDBOX_FUSION_ENDPOINT=your_sandbox_endpoint` in `.env`

### 7. IDP Service (advanced file parsing)
- **Service**: Alibaba Cloud IDP
- **Purpose**: Enhanced document parsing
- **Setup**: Already disabled in `.env` with `USE_IDP=False`
- **Can skip for now**: Basic Dashscope parsing is sufficient

## 📝 Configuration Checklist

After obtaining the keys, edit your `.env` file:

```bash
# 1. Update Serper API
SERPER_KEY_ID=your_actual_serper_key_here

# 2. Update Jina API
JINA_API_KEYS=your_actual_jina_key_here

# 3. Update summarization model API
API_KEY=your_actual_openai_or_compatible_key_here
API_BASE=https://api.openai.com/v1
SUMMARY_MODEL_NAME=gpt-4o-mini

# 4. Update Dashscope API
DASHSCOPE_API_KEY=your_actual_dashscope_key_here
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/api/v1
VIDEO_MODEL_NAME=qwen-omni-turbo
VIDEO_ANALYSIS_MODEL_NAME=qwen-plus-latest

# 5. (Optional) If you setup Python sandbox
# SANDBOX_FUSION_ENDPOINT=http://your-sandbox-ip:8080
```

## 💰 Estimated Costs for Medium-Scale Testing

For running 50-100 questions per benchmark (total ~200 questions):

1. **OpenRouter** (Tongyi-DeepResearch): ~$20-40
2. **Serper** (search): Free tier sufficient or ~$10
3. **Jina AI**: Free tier sufficient
4. **OpenAI/Summary**: ~$5-15 depending on model
5. **Dashscope**: ~$5-10

**Total estimated cost**: $50-100 for full medium-scale evaluation

## 🚀 Next Steps

1. Obtain all required API keys (1-5 above)
2. Edit `.env` file with your actual keys
3. Continue with Phase 2: Configure OpenRouter API in `inference/react_agent.py`

---

**Note**: Store your API keys securely. Never commit the `.env` file to version control!
