#!/usr/bin/env python3
"""
Quick script to filter text-only questions from HLE dataset for DeepResearch inference.
"""
import json

input_file = "inference/eval_data/hle_full.jsonl"
output_file = "inference/eval_data/hle_text_only.jsonl"
test_file = "inference/eval_data/hle_test_small.jsonl"

text_only = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        # Check if has image (base64 encoded or URL)
        has_image = False
        if 'image' in item and item['image']:
            if isinstance(item['image'], str) and (item['image'].startswith('data:image') or item['image'].startswith('http')):
                has_image = True

        if not has_image:
            # Convert to DeepResearch format
            question = item.get('question', item.get('prompt', ''))
            answer = item.get('answer', item.get('correct_answer', ''))

            if isinstance(answer, list):
                answer = answer[0] if answer else ''

            converted = {
                "question": question.strip(),
                "answer": str(answer).strip()
            }
            text_only.append(converted)

print(f"Found {len(text_only)} text-only questions out of 2500 total")

# Save full text-only
with open(output_file, 'w', encoding='utf-8') as f:
    for item in text_only:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Saved to {output_file}")

# Save small test set
with open(test_file, 'w', encoding='utf-8') as f:
    for item in text_only[:20]:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Saved test set (20 questions) to {test_file}")
