#!/bin/bash

###############################################################################
# DeepResearch Full Evaluation Pipeline
#
# This script orchestrates the complete evaluation pipeline:
# 1. Verify environment and API keys
# 2. Run inference on specified benchmarks
# 3. Evaluate results
# 4. Generate comprehensive report
#
# Usage:
#   bash scripts/run_pipeline.sh --benchmark test_small    # Quick test
#   bash scripts/run_pipeline.sh --benchmark gaia          # Run GAIA
#   bash scripts/run_pipeline.sh --all                     # Run all benchmarks
###############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠  $1${NC}"
}

# Parse arguments
BENCHMARK=""
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --benchmark <name> | --all"
            echo "Benchmarks: test_small, gaia, hle, browsecomp"
            exit 1
            ;;
    esac
done

print_header "DeepResearch Full Evaluation Pipeline"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

###############################################################################
# Step 1: Environment Verification
###############################################################################

print_header "Step 1: Environment Verification"

# Check conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    print_error "Conda environment not activated"
    echo "Please run: conda activate react_infer_env"
    exit 1
fi

if [[ "$CONDA_DEFAULT_ENV" != "react_infer_env" ]]; then
    print_warning "Active environment: $CONDA_DEFAULT_ENV (expected: react_infer_env)"
fi

print_success "Conda environment: $CONDA_DEFAULT_ENV"

# Check .env file
if [[ ! -f ".env" ]]; then
    print_error ".env file not found"
    echo "Please create .env from .env.example"
    exit 1
fi

print_success ".env file found"

# Load environment variables
set -a
source .env
set +a

# Verify critical API keys
missing_keys=()

if [[ "$OPENROUTER_API_KEY" == "your_openrouter_key" ]] || [[ -z "$OPENROUTER_API_KEY" ]]; then
    missing_keys+=("OPENROUTER_API_KEY")
fi

if [[ "$SERPER_KEY_ID" == "your_key" ]] || [[ -z "$SERPER_KEY_ID" ]]; then
    missing_keys+=("SERPER_KEY_ID")
fi

if [[ "$JINA_API_KEYS" == "your_key" ]] || [[ -z "$JINA_API_KEYS" ]]; then
    missing_keys+=("JINA_API_KEYS")
fi

if [[ "${#missing_keys[@]}" -gt 0 ]]; then
    print_error "Missing API keys:"
    for key in "${missing_keys[@]}"; do
        echo "  - $key"
    done
    echo ""
    echo "Please configure these keys in .env file"
    echo "See API_KEYS_SETUP.md for instructions"
    exit 1
fi

print_success "All required API keys configured"

# Check dataset files
if [[ "$RUN_ALL" == true ]]; then
    BENCHMARKS=("test_small" "browsecomp" "gaia" "hle")
else
    BENCHMARKS=("$BENCHMARK")
fi

echo ""
echo "Benchmarks to run: ${BENCHMARKS[@]}"

###############################################################################
# Step 2: Run Inference
###############################################################################

print_header "Step 2: Running Inference"

for bench in "${BENCHMARKS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Benchmark: $bench"
    echo "----------------------------------------"

    # Determine dataset file
    case $bench in
        test_small)
            DATASET_FILE="inference/eval_data/test_small.jsonl"
            ;;
        gaia)
            DATASET_FILE="inference/eval_data/gaia_test.jsonl"
            ;;
        hle)
            DATASET_FILE="inference/eval_data/hle_test.jsonl"
            ;;
        browsecomp)
            DATASET_FILE="inference/eval_data/browsecomp_test.jsonl"
            ;;
        *)
            print_error "Unknown benchmark: $bench"
            continue
            ;;
    esac

    # Check if dataset exists
    if [[ ! -f "$DATASET_FILE" ]]; then
        print_warning "Dataset not found: $DATASET_FILE"
        if [[ "$bench" == "gaia" ]] || [[ "$bench" == "hle" ]]; then
            echo "This dataset requires HuggingFace approval"
            echo "See SETUP_COMPLETE.md for instructions"
        fi
        continue
    fi

    # Update DATASET in environment
    export DATASET="$DATASET_FILE"

    print_success "Dataset: $DATASET_FILE"

    # Run inference
    echo "Starting inference (this may take several hours)..."
    echo "Progress will be shown below:"
    echo ""

    cd inference
    if bash run_react_infer_openrouter.sh; then
        print_success "Inference complete for $bench"
    else
        print_error "Inference failed for $bench"
        cd "$PROJECT_ROOT"
        continue
    fi
    cd "$PROJECT_ROOT"

done

###############################################################################
# Step 3: Run Evaluation
###############################################################################

print_header "Step 3: Running Evaluation"

# Export API keys for evaluation
export API_KEY="${API_KEY}"
export API_BASE="${API_BASE}"
export OPENAI_API_KEY="${API_KEY}"
export OPENAI_API_BASE="${API_BASE}"
export Qwen2_5_7B_PATH="Qwen/Qwen2.5-7B-Instruct"

# Run evaluation script
if python scripts/run_evaluation.py --all; then
    print_success "Evaluation complete"
else
    print_error "Evaluation failed"
    exit 1
fi

###############################################################################
# Step 4: Generate Report
###############################################################################

print_header "Step 4: Generating Performance Report"

if python scripts/generate_report.py --output_dir outputs --report_file PERFORMANCE_REPORT.md; then
    print_success "Report generated: PERFORMANCE_REPORT.md"
else
    print_error "Report generation failed"
    exit 1
fi

###############################################################################
# Summary
###############################################################################

print_header "Pipeline Complete! 🎉"

echo ""
echo "Results:"
echo "  - Inference outputs: ./outputs/"
echo "  - Evaluation results: ./outputs/*/*.eval_details.jsonl"
echo "  - Performance report: ./PERFORMANCE_REPORT.md"
echo ""
echo "Next steps:"
echo "  1. Review PERFORMANCE_REPORT.md"
echo "  2. Compare results with published benchmarks"
echo "  3. Iterate on configuration if needed"
echo ""

print_success "All done!"
