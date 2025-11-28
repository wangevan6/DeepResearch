#!/usr/bin/env python3
"""
Filter text-only questions from HLE dataset and convert to DeepResearch format.

This script:
1. Loads the full HLE dataset
2. Filters for text-only questions (no images)
3. Converts to DeepResearch format: {"question": "...", "answer": "..."}
4. Creates a small test subset for pipeline validation

Usage:
    python scripts/prepare_hle_textonly.py [--input PATH] [--output PATH]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


def has_image(item: Dict[str, Any]) -> bool:
    """Check if a question contains images."""
    # Check various possible image field names
    image_fields = ['images', 'image', 'image_path', 'image_url', 'img']

    for field in image_fields:
        if field in item:
            value = item[field]
            # Check if field has content
            if value:
                if isinstance(value, str) and value.strip():
                    return True
                elif isinstance(value, list) and len(value) > 0:
                    return True
                elif isinstance(value, dict) and value:
                    return True

    return False


def convert_to_deepresearch_format(item: Dict[str, Any]) -> Dict[str, str]:
    """Convert HLE format to DeepResearch format."""

    # Extract question text
    # Try different possible field names
    question_text = None
    for field in ['question', 'query', 'prompt', 'text']:
        if field in item and item[field]:
            question_text = item[field]
            break

    if not question_text:
        raise ValueError(f"No question field found in item: {item.keys()}")

    # Extract answer
    # Try different possible field names
    answer_text = None
    for field in ['answer', 'correct_answer', 'solution', 'ground_truth']:
        if field in item and item[field]:
            answer_text = item[field]
            break

    if not answer_text:
        # Some datasets might have answers in choices/options
        if 'choices' in item and 'correct_choice' in item:
            correct_idx = item['correct_choice']
            answer_text = item['choices'][correct_idx]
        else:
            raise ValueError(f"No answer field found in item: {item.keys()}")

    # Convert to string if necessary
    if not isinstance(question_text, str):
        question_text = str(question_text)
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)

    return {
        "question": question_text.strip(),
        "answer": answer_text.strip()
    }


def prepare_hle_textonly(
    input_path: str = "inference/eval_data/hle_full.json",
    output_path: str = "inference/eval_data/hle_text_only.jsonl",
    test_output_path: str = "inference/eval_data/hle_test_small.jsonl",
    test_size: int = 10
):
    """Filter text-only questions and convert format."""

    print("=" * 70)
    print("Preparing HLE Text-Only Dataset for DeepResearch")
    print("=" * 70)
    print()

    # Load input dataset
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        print()
        print("Please run first: python scripts/download_hle_dataset.py")
        return False

    print(f"📂 Loading dataset from: {input_path}")

    try:
        items = None
        if input_path.lower().endswith('.jsonl'):
            with open(input_path, 'r', encoding='utf-8') as f:
                items = [json.loads(line) for line in f if line.strip()]
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'data' in data:
                items = data['data']
            elif isinstance(data, list):
                items = data
            else:
                items = [data]

        print(f"✅ Loaded {len(items)} total questions")
        print()

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

    # Filter text-only questions
    print("🔍 Filtering text-only questions...")
    text_only_items = []

    for idx, item in enumerate(items):
        if not has_image(item):
            try:
                converted = convert_to_deepresearch_format(item)
                text_only_items.append(converted)
            except Exception as e:
                print(f"   ⚠️  Warning: Could not convert item {idx}: {e}")
                continue

    print(f"✅ Found {len(text_only_items)} text-only questions")
    print()

    if len(text_only_items) == 0:
        print("❌ Error: No text-only questions found!")
        return False

    # Display statistics
    print("📊 Statistics:")
    print(f"   - Total questions: {len(items)}")
    print(f"   - Text-only: {len(text_only_items)}")
    print(f"   - Multimodal: {len(items) - len(text_only_items)}")
    print(f"   - Percentage text-only: {len(text_only_items)/len(items)*100:.1f}%")
    print()

    # Display sample
    print("📝 Sample questions:")
    for i, item in enumerate(text_only_items[:3]):
        print(f"\n   Question {i+1}:")
        question = item['question']
        answer = item['answer']
        print(f"   Q: {question[:150]}..." if len(question) > 150 else f"   Q: {question}")
        print(f"   A: {answer[:100]}..." if len(answer) > 100 else f"   A: {answer}")
    print()

    # Save full text-only dataset
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving text-only dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in text_only_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Saved {len(text_only_items)} questions")
    print()

    # Create small test subset
    print(f"🧪 Creating test subset ({test_size} questions)...")
    test_items = text_only_items[:test_size]

    test_output_file = Path(test_output_path)
    with open(test_output_path, 'w', encoding='utf-8') as f:
        for item in test_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Test subset saved to: {test_output_path}")
    print()

    # Verification
    print("🔍 Verifying output files...")
    main_size_mb = output_file.stat().st_size / (1024 * 1024)
    test_size_mb = test_output_file.stat().st_size / (1024 * 1024)

    print(f"   - Main dataset: {main_size_mb:.2f} MB ({len(text_only_items)} questions)")
    print(f"   - Test subset: {test_size_mb:.2f} MB ({len(test_items)} questions)")
    print()

    print("=" * 70)
    print("✅ Preparation complete!")
    print("=" * 70)
    print()
    print("Files created:")
    print(f"  📄 {output_path}")
    print(f"  📄 {test_output_path}")
    print()
    print("Next steps:")
    print("  1. Test with small subset:")
    print(f"     python inference/run_multi_react.py --dataset {test_output_path}")
    print()
    print("  2. Run full evaluation:")
    print(f"     python inference/run_multi_react.py --dataset {output_path}")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Filter text-only HLE questions and convert to DeepResearch format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="inference/eval_data/hle_full.jsonl",
        help="Input HLE dataset path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference/eval_data/hle_text_only.jsonl",
        help="Output path for text-only questions"
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default="inference/eval_data/hle_test_small.jsonl",
        help="Output path for small test subset"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10,
        help="Number of questions in test subset"
    )

    args = parser.parse_args()

    success = prepare_hle_textonly(
        input_path=args.input,
        output_path=args.output,
        test_output_path=args.test_output,
        test_size=args.test_size
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
