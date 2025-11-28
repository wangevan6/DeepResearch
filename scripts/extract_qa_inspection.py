#!/usr/bin/env python
"""
Extract questions and answers from inference result JSONL files for manual inspection.

Usage:
    python scripts/extract_qa_inspection.py --input <input.jsonl> [--output <output.txt>]

Example:
    python scripts/extract_qa_inspection.py \
        --input outputs/hle_test/Tongyi-DeepResearch-30B-A3B_sglang/eval_data/hle_test_small/iter1.jsonl \
        --output outputs/hle_test/Tongyi-DeepResearch-30B-A3B_sglang/eval_data/hle_test_small/iter1_inspection.txt

Data Format Notes:
    The JSONL file contains items with these fields:
    - question: The original question
    - answer: Ground truth answer
    - prediction: Content extracted from <answer>...</answer> tags (tags already stripped)
    - messages: Full conversation history (system, user, assistant, tool_response)
    - termination: How inference ended - "answer" (success) or error message

    The <answer> tags exist in messages[-1]['content'] (last assistant message),
    but 'prediction' already contains the extracted content.
"""

import argparse
import json
import re
import os
from pathlib import Path


def load_jsonl(filepath: str) -> list:
    """Load a JSONL file and return a list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def get_answer_from_item(item: dict, max_length: int = 1500) -> tuple[str, str]:
    """
    Extract the model's answer from an inference result item.

    The 'prediction' field contains content extracted from <answer> tags (tags stripped).
    The 'termination' field indicates if extraction was successful.

    Args:
        item: A single item from the JSONL file
        max_length: Maximum length for truncated answers

    Returns:
        tuple: (answer_text, status) where status is 'success', 'error', or 'truncated'
    """
    prediction = item.get('prediction', '')
    termination = item.get('termination', '')

    # Check termination status
    if termination == 'answer':
        # Successful extraction - prediction contains the answer
        status = 'success'
    elif 'error' in termination.lower() or 'limit' in termination.lower():
        # Some error occurred
        status = f'error: {termination}'
    else:
        status = f'unknown: {termination}'

    # Truncate if needed
    answer = prediction.strip()
    if len(answer) > max_length:
        answer = answer[:max_length] + "..."
        if status == 'success':
            status = 'truncated'

    return answer, status


def format_inspection_item(index: int, item: dict, max_answer_length: int = 1500) -> str:
    """Format a single item for inspection output."""
    question = item.get('question', 'N/A')
    ground_truth = item.get('answer', 'N/A')
    termination = item.get('termination', 'N/A')

    answer, status = get_answer_from_item(item, max_answer_length)

    # Add status prefix if not successful
    if status != 'success' and status != 'truncated':
        answer = f"[{status.upper()}]\n{answer}"
    elif status == 'truncated':
        answer = f"[TRUNCATED]\n{answer}"

    lines = [
        "=" * 80,
        f"ITEM {index}",
        "=" * 80,
        "",
        "QUESTION:",
        question,
        "",
        f"MODEL ANSWER: (termination: {termination})",
        answer,
        "",
        "GROUND TRUTH:",
        ground_truth,
        ""
    ]

    return "\n".join(lines)


def generate_summary(items: list) -> str:
    """Generate a summary of the extraction results."""
    total = len(items)

    # Count by termination status
    success_count = 0
    error_count = 0
    other_count = 0

    for item in items:
        termination = item.get('termination', '')
        if termination == 'answer':
            success_count += 1
        elif 'error' in termination.lower() or 'limit' in termination.lower():
            error_count += 1
        else:
            other_count += 1

    # Count unique questions
    unique_questions = len(set(item.get('question', '') for item in items))

    lines = [
        "=" * 80,
        "SUMMARY",
        "=" * 80,
        "",
        f"Total items:              {total}",
        f"Unique questions:         {unique_questions}",
        f"Successful (answer):      {success_count}",
        f"Errors/Limits:            {error_count}",
        f"Other termination:        {other_count}",
        "",
        "=" * 80,
        ""
    ]

    return "\n".join(lines)


def extract_qa_inspection(input_path: str, output_path: str = None, max_answer_length: int = 1500) -> str:
    """
    Extract questions and answers from a JSONL file and save to inspection file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output text file (default: input_path with .txt extension)
        max_answer_length: Maximum length for truncated answers

    Returns:
        Path to the output file
    """
    # Determine output path
    if output_path is None:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_inspection.txt")

    # Load data
    items = load_jsonl(input_path)

    # Generate output
    output_lines = []

    # Add summary at the top
    output_lines.append(generate_summary(items))

    # Add each item
    for i, item in enumerate(items, 1):
        output_lines.append(format_inspection_item(i, item, max_answer_length))

    # Write output
    output_content = "\n".join(output_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_content)

    print(f"Inspection file saved to: {output_path}")
    print(f"Total items: {len(items)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract questions and answers from inference JSONL for manual inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input JSONL file containing inference results'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path to output text file (default: <input>_inspection.txt)'
    )
    parser.add_argument(
        '--max-length', '-m',
        type=int,
        default=1500,
        help='Maximum length for truncated answers without <answer> tag (default: 1500)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    extract_qa_inspection(args.input, args.output, args.max_length)
    return 0


if __name__ == '__main__':
    exit(main())
