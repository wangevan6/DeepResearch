#!/usr/bin/env python3
"""
Automated evaluation script for DeepResearch benchmarks.

This script automates the evaluation of inference results using LLM-as-judge.

Usage:
    python scripts/run_evaluation.py --benchmark hle --input_file outputs/.../iter1.jsonl
    python scripts/run_evaluation.py --benchmark gaia --input_folder outputs/.../gaia_test
    python scripts/run_evaluation.py --all  # Evaluate all available results
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def find_inference_outputs(output_dir: str = "outputs") -> Dict[str, List[str]]:
    """Find all inference output files."""
    output_path = Path(output_dir)
    results = {
        "hle": [],
        "gaia": [],
        "browsecomp": []
    }

    if not output_path.exists():
        print(f"Warning: Output directory {output_dir} not found")
        return results

    # Search for iter*.jsonl files
    for jsonl_file in output_path.rglob("iter*.jsonl"):
        file_str = str(jsonl_file)

        if "hle" in file_str.lower():
            results["hle"].append(str(jsonl_file))
        elif "gaia" in file_str.lower():
            # For GAIA, we need the folder containing iter files
            results["gaia"].append(str(jsonl_file.parent))
        elif "browsecomp" in file_str.lower():
            results["browsecomp"].append(str(jsonl_file.parent))

    # Deduplicate folders
    results["gaia"] = list(set(results["gaia"]))
    results["browsecomp"] = list(set(results["browsecomp"]))

    return results


def evaluate_hle(input_file: str, tokenizer_path: str = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"):
    """Evaluate HLE results."""
    print("\n" + "="*60)
    print("Evaluating HLE Dataset")
    print("="*60)

    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        return False

    # Check environment variables
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    if not api_key:
        print("Error: API_KEY or OPENAI_API_KEY not set")
        print("Export your API key: export API_KEY=your_key")
        return False

    print(f"Input file: {input_file}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Judge API: {api_base}")

    cmd = [
        "python", "evaluation/evaluate_hle_official.py",
        "--input_fp", input_file,
        "--tokenizer_path", tokenizer_path,
        "--repeat_times", "1"
    ]

    try:
        print("\nRunning evaluation...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Check for output files
        eval_details = input_file.replace(".jsonl", ".eval_details.jsonl")
        eval_report = input_file.replace(".jsonl", ".report.json")

        if Path(eval_details).exists() and Path(eval_report).exists():
            print(f"\n✓ Evaluation complete!")
            print(f"  Details: {eval_details}")
            print(f"  Report: {eval_report}")

            # Show summary
            with open(eval_report, 'r') as f:
                report = json.load(f)
                print(f"\n📊 Results Summary:")
                print(f"  Accuracy: {report.get('accuracy', 'N/A')}")
                print(f"  Total questions: {report.get('total', 'N/A')}")

            return True
        else:
            print("Warning: Evaluation completed but output files not found")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def evaluate_gaia_or_browsecomp(input_folder: str, dataset: str):
    """Evaluate GAIA or BrowseComp results."""
    print("\n" + "="*60)
    print(f"Evaluating {dataset.upper()} Dataset")
    print("="*60)

    if not Path(input_folder).exists():
        print(f"Error: Input folder not found: {input_folder}")
        return False

    # Check for required iter files
    iter_files = list(Path(input_folder).glob("iter*.jsonl"))
    if not iter_files:
        print(f"Error: No iter*.jsonl files found in {input_folder}")
        return False

    print(f"Input folder: {input_folder}")
    print(f"Found {len(iter_files)} iteration files")

    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    qwen_path = os.getenv("Qwen2_5_7B_PATH", "Qwen/Qwen2.5-7B-Instruct")

    if not openai_key:
        print("Error: OPENAI_API_KEY not set")
        print("Export your API key: export OPENAI_API_KEY=your_key")
        return False

    print(f"Judge API: {openai_base}")
    print(f"Qwen model: {qwen_path}")

    cmd = [
        "python", "evaluation/evaluate_deepsearch_official.py",
        "--input_folder", input_folder,
        "--dataset", dataset
    ]

    try:
        print("\nRunning evaluation...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        print(f"\n✓ Evaluation complete for {dataset}!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepResearch benchmark results")
    parser.add_argument("--benchmark", choices=["hle", "gaia", "browsecomp"],
                       help="Which benchmark to evaluate")
    parser.add_argument("--input_file", type=str,
                       help="Input JSONL file (for HLE)")
    parser.add_argument("--input_folder", type=str,
                       help="Input folder containing iter files (for GAIA/BrowseComp)")
    parser.add_argument("--all", action="store_true",
                       help="Evaluate all available results")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory to search (default: outputs)")

    args = parser.parse_args()

    if not args.all and not args.benchmark:
        parser.print_help()
        sys.exit(1)

    print("="*60)
    print("DeepResearch Evaluation Script")
    print("="*60)

    # Check environment setup
    print("\nChecking environment...")
    api_keys_ok = True

    if not os.getenv("API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: API_KEY/OPENAI_API_KEY not set")
        api_keys_ok = False

    if not os.getenv("OPENAI_API_BASE"):
        print("⚠️  Warning: OPENAI_API_BASE not set (using default)")

    if not api_keys_ok:
        print("\nPlease set required API keys:")
        print("  export API_KEY=your_openai_key")
        print("  export OPENAI_API_KEY=your_openai_key")
        print("  export OPENAI_API_BASE=https://api.openai.com/v1")
        sys.exit(1)

    results = {}

    if args.all:
        print("\nSearching for inference results...")
        available = find_inference_outputs(args.output_dir)

        print(f"\nFound results:")
        for benchmark, files in available.items():
            print(f"  {benchmark.upper()}: {len(files)} result(s)")

        # Evaluate HLE
        if available["hle"]:
            for hle_file in available["hle"]:
                success = evaluate_hle(hle_file)
                results[f"hle_{Path(hle_file).stem}"] = success

        # Evaluate GAIA
        if available["gaia"]:
            for gaia_folder in available["gaia"]:
                success = evaluate_gaia_or_browsecomp(gaia_folder, "gaia")
                results[f"gaia_{Path(gaia_folder).name}"] = success

        # Evaluate BrowseComp
        if available["browsecomp"]:
            for bc_folder in available["browsecomp"]:
                success = evaluate_gaia_or_browsecomp(bc_folder, "browsecomp_en")
                results[f"browsecomp_{Path(bc_folder).name}"] = success

    elif args.benchmark == "hle":
        if not args.input_file:
            print("Error: --input_file required for HLE evaluation")
            sys.exit(1)
        results["hle"] = evaluate_hle(args.input_file)

    elif args.benchmark in ["gaia", "browsecomp"]:
        if not args.input_folder:
            print(f"Error: --input_folder required for {args.benchmark.upper()} evaluation")
            sys.exit(1)
        dataset = "gaia" if args.benchmark == "gaia" else "browsecomp_en"
        results[args.benchmark] = evaluate_gaia_or_browsecomp(args.input_folder, dataset)

    # Summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    total = len(results)
    success_count = sum(1 for s in results.values() if s)
    print(f"\nTotal: {success_count}/{total} successful")


if __name__ == "__main__":
    main()
