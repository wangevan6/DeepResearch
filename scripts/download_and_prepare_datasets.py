#!/usr/bin/env python3
"""
Download and prepare benchmark datasets for DeepResearch evaluation.

Datasets:
- GAIA: Question answering with file attachments (requires HuggingFace approval)
- HLE: Humanity's Last Exam
- BrowseComp-EN: Web browsing comprehension (English)
- BrowseComp-ZH: Web browsing comprehension (Chinese)
- WebWalkerQA: Web navigation benchmark (680 questions)
- FRAMERS: Multi-hop reasoning benchmark (824 questions)
- SimpleQA: Factuality benchmark (4,326 questions)
- xbench-DeepSearch: Search evaluation benchmark (encrypted)

Usage:
    # Download all full datasets
    python download_and_prepare_datasets.py --datasets all --download_full

    # Download full + create toy samples (10 questions each)
    python download_and_prepare_datasets.py --datasets all --download_full --create_toy --toy_size 10
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict
import shutil


def setup_directories(output_dir: str):
    """Create necessary directories for datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_corpus = output_path / "file_corpus"
    file_corpus.mkdir(exist_ok=True)

    print(f"✓ Created output directory: {output_path}")
    print(f"✓ Created file corpus directory: {file_corpus}")
    return output_path, file_corpus


def create_toy_sample(input_file: Path, output_file: Path, toy_size: int, seed: int = 42):
    """Create a toy sample from a full dataset."""
    if not input_file.exists():
        print(f"  ⚠️  Input file not found: {input_file}, skipping toy creation")
        return False

    try:
        # Read full dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        if len(data) == 0:
            print(f"  ⚠️  Input file is empty: {input_file}")
            return False

        # Sample with seed
        random.seed(seed)
        sample_size = min(toy_size, len(data))
        sampled_data = random.sample(data, sample_size)

        # Write toy sample
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in sampled_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"  ✓ Created toy sample: {output_file} ({sample_size} questions)")
        return True

    except Exception as e:
        print(f"  ✗ Failed to create toy sample: {e}")
        return False


def download_gaia(output_dir: Path, file_corpus: Path, download_full: bool = False):
    """Download and prepare GAIA dataset."""
    print("\n" + "="*60)
    print("Downloading GAIA Dataset")
    print("="*60)

    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download

        print("Attempting to download GAIA dataset...")
        print("Note: This dataset requires approval from HuggingFace.")
        print("      If download fails, please request access at:")
        print("      https://huggingface.co/datasets/gaia-benchmark/GAIA")
        print()

        # Try to download the dataset
        try:
            # Download validation split for testing
            dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
            print(f"✓ Successfully loaded GAIA dataset: {len(dataset)} examples")

            # Determine output filename
            output_file = output_dir / ("gaia_full.jsonl" if download_full else "gaia_test.jsonl")

            # Convert to JSONL format
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    # Handle file attachments
                    question = item['Question']
                    if 'file_name' in item and item['file_name']:
                        # Prepend filename to question
                        question = f"{item['file_name']} {question}"

                    entry = {
                        "question": question,
                        "answer": item['Final answer']
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"✓ Saved GAIA dataset to: {output_file}")
            print(f"  Total questions: {len(dataset)}")

            # Note about file downloads
            print("\n⚠️  File Attachments:")
            print("    GAIA includes file attachments (PDFs, images, etc.)")
            print("    You need to manually download them from:")
            print("    https://huggingface.co/datasets/gaia-benchmark/GAIA/tree/main/2023/validation")
            print(f"    Place them in: {file_corpus}/")

            return True, output_file

        except Exception as e:
            print(f"✗ Failed to download GAIA: {e}")
            print("\n  Possible reasons:")
            print("  1. You need to request access at https://huggingface.co/datasets/gaia-benchmark/GAIA")
            print("  2. You need to login: huggingface-cli login")
            print("  3. Dataset structure may have changed")
            print("\n  Skipping GAIA for now...")
            return False, None

    except ImportError:
        print("✗ Error: 'datasets' package not installed")
        print("  Run: pip install datasets")
        return False, None


def download_hle(output_dir: Path, download_full: bool = False):
    """Download and prepare HLE dataset."""
    print("\n" + "="*60)
    print("Downloading HLE Dataset")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Downloading HLE (Humanity's Last Exam) dataset...")

        # Download test split
        dataset = load_dataset("cais/hle", split="test")
        print(f"✓ Successfully loaded HLE dataset: {len(dataset)} examples")

        # Determine output filename
        output_file = output_dir / ("hle_full.jsonl" if download_full else "hle_test.jsonl")

        # Convert to JSONL format
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                entry = {
                    "question": item['question'],
                    "answer": item.get('answer', '')  # HLE may have empty answers
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"✓ Saved HLE dataset to: {output_file}")
        print(f"  Total questions: {len(dataset)}")

        return True, output_file

    except Exception as e:
        print(f"✗ Failed to download HLE: {e}")
        print("  Skipping HLE for now...")
        return False, None


def download_browsecomp_en(output_dir: Path, download_full: bool = False):
    """Download and prepare BrowseComp-EN dataset."""
    print("\n" + "="*60)
    print("Downloading BrowseComp-EN Dataset")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Downloading BrowseComp-EN dataset...")

        try:
            dataset = load_dataset("Alibaba-NLP/BrowseComp", "en", split="test")
            print(f"✓ Successfully loaded BrowseComp-EN dataset: {len(dataset)} examples")

            # Determine output filename
            output_file = output_dir / ("browsecomp_en_full.jsonl" if download_full else "browsecomp_en_test.jsonl")

            # Convert to JSONL format
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    entry = {
                        "question": item['question'],
                        "answer": item.get('answer', '')
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"✓ Saved BrowseComp-EN dataset to: {output_file}")
            print(f"  Total questions: {len(dataset)}")

            return True, output_file

        except Exception as e:
            print(f"✗ Failed to download BrowseComp-EN: {e}")
            print("  Dataset may not be available on HuggingFace")
            return False, None

    except ImportError:
        print("✗ Error: 'datasets' package not installed")
        print("  Run: pip install datasets")
        return False, None


def download_browsecomp_zh(output_dir: Path, download_full: bool = False):
    """Download and prepare BrowseComp-ZH dataset."""
    print("\n" + "="*60)
    print("Downloading BrowseComp-ZH Dataset")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Downloading BrowseComp-ZH (Chinese) dataset...")

        try:
            dataset = load_dataset("Alibaba-NLP/BrowseComp", "zh", split="test")
            print(f"✓ Successfully loaded BrowseComp-ZH dataset: {len(dataset)} examples")

            # Determine output filename
            output_file = output_dir / ("browsecomp_zh_full.jsonl" if download_full else "browsecomp_zh_test.jsonl")

            # Convert to JSONL format
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    entry = {
                        "question": item['question'],
                        "answer": item.get('answer', '')
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"✓ Saved BrowseComp-ZH dataset to: {output_file}")
            print(f"  Total questions: {len(dataset)}")

            return True, output_file

        except Exception as e:
            print(f"✗ Failed to download BrowseComp-ZH: {e}")
            print("  Dataset may not be available on HuggingFace")
            return False, None

    except ImportError:
        print("✗ Error: 'datasets' package not installed")
        print("  Run: pip install datasets")
        return False, None


def download_webwalkerqa(output_dir: Path, download_full: bool = False):
    """Download and prepare WebWalkerQA dataset."""
    print("\n" + "="*60)
    print("Downloading WebWalkerQA Dataset")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Downloading WebWalkerQA dataset...")
        print("Source: https://huggingface.co/datasets/callanwu/WebWalkerQA")

        try:
            dataset = load_dataset("callanwu/WebWalkerQA", split="main")
            print(f"✓ Successfully loaded WebWalkerQA dataset: {len(dataset)} examples")

            # Determine output filename
            output_file = output_dir / ("webwalkerqa_full.jsonl" if download_full else "webwalkerqa_test.jsonl")

            # Convert to JSONL format (normalize field names)
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    entry = {
                        "question": item.get('Question', item.get('question', '')),
                        "answer": item.get('Answer', item.get('answer', ''))
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"✓ Saved WebWalkerQA dataset to: {output_file}")
            print(f"  Total questions: {len(dataset)}")

            return True, output_file

        except Exception as e:
            print(f"✗ Failed to download WebWalkerQA: {e}")
            print("  Dataset may not be available or structure changed")
            return False, None

    except ImportError:
        print("✗ Error: 'datasets' package not installed")
        print("  Run: pip install datasets")
        return False, None


def download_framers(output_dir: Path, download_full: bool = False):
    """Download and prepare FRAMERS benchmark dataset."""
    print("\n" + "="*60)
    print("Downloading FRAMERS Dataset")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Downloading FRAMERS (Google FRAMES) dataset...")
        print("Source: https://huggingface.co/datasets/google/frames-benchmark")

        try:
            dataset = load_dataset("google/frames-benchmark", split="test")
            print(f"✓ Successfully loaded FRAMERS dataset: {len(dataset)} examples")

            # Determine output filename
            output_file = output_dir / ("framers_full.jsonl" if download_full else "framers_test.jsonl")

            # Convert to JSONL format
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    # Extract question and answer from the dataset structure
                    entry = {
                        "question": item.get('Question', item.get('question', '')),
                        "answer": item.get('Answer', item.get('answer', ''))
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"✓ Saved FRAMERS dataset to: {output_file}")
            print(f"  Total questions: {len(dataset)}")
            print("  Note: FRAMERS contains multi-hop reasoning questions")

            return True, output_file

        except Exception as e:
            print(f"✗ Failed to download FRAMERS: {e}")
            print("  Dataset may not be available or structure changed")
            return False, None

    except ImportError:
        print("✗ Error: 'datasets' package not installed")
        print("  Run: pip install datasets")
        return False, None


def download_simpleqa(output_dir: Path, download_full: bool = False):
    """Download and prepare SimpleQA dataset."""
    print("\n" + "="*60)
    print("Downloading SimpleQA Dataset")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Downloading SimpleQA dataset...")
        print("Source: https://huggingface.co/datasets/basicv8vc/SimpleQA")

        try:
            # Try the community version on HuggingFace (uses "test" split)
            dataset = load_dataset("basicv8vc/SimpleQA", split="test")
            print(f"✓ Successfully loaded SimpleQA dataset: {len(dataset)} examples")

            # Determine output filename
            output_file = output_dir / ("simpleqa_full.jsonl" if download_full else "simpleqa_test.jsonl")

            # Convert to JSONL format
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    # Normalize field names (SimpleQA might use different field names)
                    question = item.get('problem', item.get('question', ''))
                    answer = item.get('answer', '')

                    entry = {
                        "question": question,
                        "answer": answer
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"✓ Saved SimpleQA dataset to: {output_file}")
            print(f"  Total questions: {len(dataset)}")
            print("  Note: SimpleQA is a factuality benchmark")

            return True, output_file

        except Exception as e:
            print(f"✗ Failed to download SimpleQA: {e}")
            print("  Dataset may not be available or structure changed")
            return False, None

    except ImportError:
        print("✗ Error: 'datasets' package not installed")
        print("  Run: pip install datasets")
        return False, None


def download_xbench_deepsearch(output_dir: Path, download_full: bool = False):
    """Download and prepare xbench-DeepSearch dataset."""
    print("\n" + "="*60)
    print("Downloading xbench-DeepSearch Dataset")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Downloading xbench-DeepSearch dataset...")
        print("Source: https://huggingface.co/datasets/xbench/DeepSearch-2510")
        print()
        print("⚠️  IMPORTANT NOTE:")
        print("    This dataset is encrypted to prevent contamination.")
        print("    You will need to decrypt it using the xbench-evals repository:")
        print("    https://github.com/xbench-ai/xbench-evals")
        print()

        try:
            # Try to load the latest version (uses "train" split)
            dataset = load_dataset("xbench/DeepSearch-2510", split="train")
            print(f"✓ Successfully loaded xbench-DeepSearch dataset: {len(dataset)} examples")

            # Determine output filename
            output_file = output_dir / ("xbench_deepsearch_full.jsonl" if download_full else "xbench_deepsearch_test.jsonl")

            # Convert to JSONL format (data will be encrypted)
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    # Try to extract question/answer, but they may be encrypted
                    entry = {
                        "question": item.get('question', item.get('encrypted_question', '')),
                        "answer": item.get('answer', item.get('encrypted_answer', ''))
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"✓ Saved xbench-DeepSearch dataset to: {output_file}")
            print(f"  Total questions: {len(dataset)}")
            print()
            print("⚠️  Remember to decrypt the data before use!")

            return True, output_file

        except Exception as e:
            print(f"✗ Failed to download xbench-DeepSearch: {e}")
            print("  Dataset may not be available or structure changed")
            return False, None

    except ImportError:
        print("✗ Error: 'datasets' package not installed")
        print("  Run: pip install datasets")
        return False, None


def create_small_test_set(output_dir: Path):
    """Create a small 5-question test set for initial testing."""
    print("\n" + "="*60)
    print("Creating Small Test Set (5 questions)")
    print("="*60)

    test_questions = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "answer": "William Shakespeare"
        },
        {
            "question": "What is 15 multiplied by 8?",
            "answer": "120"
        },
        {
            "question": "In what year did World War II end?",
            "answer": "1945"
        },
        {
            "question": "What is the chemical symbol for gold?",
            "answer": "Au"
        }
    ]

    output_file = output_dir / "test_small.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in test_questions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✓ Created small test set: {output_file}")
    print("  Use this for initial setup verification")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare benchmark datasets")
    parser.add_argument("--output_dir", type=str, default="inference/eval_data",
                       help="Output directory for datasets (default: inference/eval_data)")
    parser.add_argument("--datasets", nargs="+",
                       default=["all"],
                       choices=["gaia", "hle", "browsecomp_en", "browsecomp_zh",
                               "webwalkerqa", "framers", "simpleqa", "xbench", "all"],
                       help="Which datasets to download (default: all)")
    parser.add_argument("--download_full", action="store_true",
                       help="Download full datasets (saves as *_full.jsonl)")
    parser.add_argument("--create_toy", action="store_true",
                       help="Create toy samples from full datasets")
    parser.add_argument("--toy_size", type=int, default=10,
                       help="Number of questions in toy samples (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling (default: 42)")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("="*60)
    print("DeepResearch Dataset Download and Preparation")
    print("="*60)
    print(f"Download full datasets: {args.download_full}")
    print(f"Create toy samples: {args.create_toy}")
    if args.create_toy:
        print(f"Toy sample size: {args.toy_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")

    # Setup directories
    output_dir, file_corpus = setup_directories(args.output_dir)

    # Create small test set
    create_small_test_set(output_dir)

    # Determine which datasets to download
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["gaia", "hle", "browsecomp_en", "browsecomp_zh",
                               "webwalkerqa", "framers", "simpleqa", "xbench"]

    results = {}
    downloaded_files = {}

    # Download datasets
    if "gaia" in datasets_to_download:
        success, file_path = download_gaia(output_dir, file_corpus, args.download_full)
        results["gaia"] = success
        downloaded_files["gaia"] = file_path

    if "hle" in datasets_to_download:
        success, file_path = download_hle(output_dir, args.download_full)
        results["hle"] = success
        downloaded_files["hle"] = file_path

    if "browsecomp_en" in datasets_to_download:
        success, file_path = download_browsecomp_en(output_dir, args.download_full)
        results["browsecomp_en"] = success
        downloaded_files["browsecomp_en"] = file_path

    if "browsecomp_zh" in datasets_to_download:
        success, file_path = download_browsecomp_zh(output_dir, args.download_full)
        results["browsecomp_zh"] = success
        downloaded_files["browsecomp_zh"] = file_path

    if "webwalkerqa" in datasets_to_download:
        success, file_path = download_webwalkerqa(output_dir, args.download_full)
        results["webwalkerqa"] = success
        downloaded_files["webwalkerqa"] = file_path

    if "framers" in datasets_to_download:
        success, file_path = download_framers(output_dir, args.download_full)
        results["framers"] = success
        downloaded_files["framers"] = file_path

    if "simpleqa" in datasets_to_download:
        success, file_path = download_simpleqa(output_dir, args.download_full)
        results["simpleqa"] = success
        downloaded_files["simpleqa"] = file_path

    if "xbench" in datasets_to_download:
        success, file_path = download_xbench_deepsearch(output_dir, args.download_full)
        results["xbench"] = success
        downloaded_files["xbench"] = file_path

    # Create toy samples if requested
    if args.create_toy and args.download_full:
        print("\n" + "="*60)
        print(f"Creating Toy Samples ({args.toy_size} questions each)")
        print("="*60)

        for dataset_name, file_path in downloaded_files.items():
            if file_path and file_path.exists():
                toy_file = output_dir / file_path.name.replace("_full.jsonl", "_toy.jsonl")
                create_toy_sample(file_path, toy_file, args.toy_size, args.seed)

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("Downloaded datasets:")
    for dataset, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {dataset.upper()}")

    print("\nNext steps:")
    print("1. Review the API_KEYS_SETUP.md file for required API keys")
    print("2. Fill in your API keys in the .env file")
    print("3. If using GAIA, download file attachments manually")
    if args.create_toy:
        print("4. Use toy samples (*_toy.jsonl) for quick testing")
        print("5. Use full datasets (*_full.jsonl) for complete evaluation")
    print(f"\nAll datasets saved to: {output_dir}/")


if __name__ == "__main__":
    main()
