#!/bin/bash

################################################################################
# HLE Inference Script for Tongyi DeepResearch via HuggingFace Spaces API
################################################################################
#
# This script runs inference on the Humanity's Last Exam (HLE) benchmark
# using the Tongyi-DeepResearch model via HuggingFace Spaces API.
#
# Prerequisites:
# 1. Dataset prepared: inference/eval_data/hle_text_only.jsonl
# 2. API keys configured in .env file
# 3. Python environment activated with required packages
#
# Usage:
#   bash inference/run_hle_inference.sh [dryrun1|dryrun10|dryrun100|full]
#
# Arguments:
#   dryrun1   - Run on 1 question (quick pipeline test)
#   dryrun10  - Run on 10 questions (small test)
#   dryrun100 - Run on 100 questions (medium test)
#   full      - Run on full HLE text-only dataset (2158 questions, default)
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo -e "${BLUE}🚀 Tongyi DeepResearch - HLE Inference${NC}"
echo "================================================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if .env file exists
if [ ! -f "../.env" ]; then
    echo -e "${RED}❌ Error: .env file not found${NC}"
    echo ""
    echo "Please create a .env file with the following variables:"
    echo "  - HF_SPACES_API_KEY or HF_TOKEN"
    echo "  - SERPER_KEY_ID"
    echo "  - JINA_API_KEYS"
    echo "  - API_KEY (for judge model)"
    echo "  - BASE_URL (for judge model)"
    echo ""
    exit 1
fi

# Load environment variables
set -a
source ../.env
set +a

# Determine run mode
MODE="${1:-full}"

if [ "$MODE" = "dryrun1" ]; then
    DATASET="eval_data/hle_dryrun_1q.jsonl"
    OUTPUT="../outputs/hle_dryrun_1q"
    ROLLOUT_COUNT=1
    MAX_WORKERS=1
    echo -e "${YELLOW}🧪 Running in DRYRUN1 mode (1 question)${NC}"
elif [ "$MODE" = "dryrun10" ]; then
    DATASET="eval_data/hle_test_small.jsonl"
    OUTPUT="../outputs/hle_dryrun_10q"
    ROLLOUT_COUNT=1
    MAX_WORKERS=5
    echo -e "${YELLOW}🧪 Running in DRYRUN10 mode (10 questions)${NC}"
elif [ "$MODE" = "dryrun100" ]; then
    DATASET="eval_data/hle_dryrun_100q.jsonl"
    OUTPUT="../outputs/hle_dryrun_100q"
    ROLLOUT_COUNT=1
    MAX_WORKERS=10
    echo -e "${YELLOW}🧪 Running in DRYRUN100 mode (100 questions)${NC}"
elif [ "$MODE" = "full" ]; then
    DATASET="eval_data/hle_text_only.jsonl"
    OUTPUT="../outputs/hle_full"
    ROLLOUT_COUNT=3
    MAX_WORKERS=30
    echo -e "${GREEN}📊 Running in FULL mode (2158 questions)${NC}"
else
    echo -e "${RED}❌ Invalid mode: $MODE${NC}"
    echo "Usage: bash run_hle_inference.sh [dryrun1|dryrun10|dryrun100|full]"
    exit 1
fi

echo ""

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo -e "${RED}❌ Error: Dataset not found: $DATASET${NC}"
    echo ""
    echo "Please run the dataset preparation scripts first:"
    echo "  1. python ../scripts/download_hle_dataset.py"
    echo "  2. python ../scripts/prepare_hle_textonly.py"
    echo ""
    exit 1
fi

# Count questions in dataset
NUM_QUESTIONS=$(wc -l < "$DATASET")
echo -e "${BLUE}📁 Dataset:${NC} $DATASET"
echo -e "${BLUE}📊 Questions:${NC} $NUM_QUESTIONS"
echo -e "${BLUE}🔄 Rollouts:${NC} $ROLLOUT_COUNT"
echo -e "${BLUE}👥 Workers:${NC} $MAX_WORKERS"
echo -e "${BLUE}📂 Output:${NC} $OUTPUT"
echo ""

# Calculate estimated time
TOTAL_INFERENCES=$((NUM_QUESTIONS * ROLLOUT_COUNT))
EST_TIME_PER_Q=120  # 2 minutes per question (conservative estimate)
EST_TOTAL_SECONDS=$((TOTAL_INFERENCES * EST_TIME_PER_Q / MAX_WORKERS))
EST_HOURS=$((EST_TOTAL_SECONDS / 3600))
EST_MINUTES=$(((EST_TOTAL_SECONDS % 3600) / 60))

echo -e "${YELLOW}⏱️  Estimated time:${NC} ~${EST_HOURS}h ${EST_MINUTES}m"
echo -e "${YELLOW}💰 Estimated API calls:${NC} ~$((TOTAL_INFERENCES * 20)) (assuming ~20 calls/question)"
echo ""

# Confirmation prompt for full mode
if [ "$MODE" = "full" ]; then
    echo -e "${YELLOW}⚠️  This will run a full evaluation with significant API costs.${NC}"
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    echo ""
fi

# Get model path for tokenizer
MODEL_PATH="${MODEL_PATH:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"

# Configuration
TEMPERATURE="${TEMPERATURE:-0.85}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.1}"
TOP_P="${TOP_P:-0.95}"

echo "================================================================================"
echo -e "${GREEN}▶️  Starting inference...${NC}"
echo "================================================================================"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT"

# Run inference
python run_multi_react.py \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --max_workers "$MAX_WORKERS" \
    --roll_out_count "$ROLLOUT_COUNT" \
    --temperature "$TEMPERATURE" \
    --presence_penalty "$PRESENCE_PENALTY" \
    --top_p "$TOP_P" \
    2>&1 | tee "$OUTPUT/inference.log"

EXIT_CODE=$?

echo ""
echo "================================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Inference completed successfully!${NC}"
    echo ""
    echo "Output files:"
    for i in $(seq 1 $ROLLOUT_COUNT); do
        OUTPUT_FILE="$OUTPUT/iter${i}.jsonl"
        if [ -f "$OUTPUT_FILE" ]; then
            NUM_RESULTS=$(wc -l < "$OUTPUT_FILE")
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
            echo -e "  📄 ${OUTPUT_FILE} (${NUM_RESULTS} results, ${FILE_SIZE})"
        fi
    done
    echo ""
    echo "Next steps:"
    echo "  1. Run evaluation:"
    echo "     cd ../evaluation"
    echo "     python evaluate_hle_official.py --input_fp $OUTPUT/iter1.jsonl"
    echo ""
    echo "  2. Calculate metrics:"
    echo "     python ../scripts/calculate_hle_metrics.py --input_dir $OUTPUT"
    echo ""
else
    echo -e "${RED}❌ Inference failed with exit code: $EXIT_CODE${NC}"
    echo ""
    echo "Check the log file for details: $OUTPUT/inference.log"
    echo ""
fi

echo "================================================================================"
