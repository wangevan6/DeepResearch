#!/usr/bin/env python3
"""
Generate comprehensive performance report from evaluation results.

This script consolidates all evaluation results into a single report.

Usage:
    python scripts/generate_report.py --output_dir outputs --report_file PERFORMANCE_REPORT.md
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics


def find_evaluation_results(output_dir: str) -> Dict[str, List[Path]]:
    """Find all evaluation result files."""
    output_path = Path(output_dir)
    results = {
        "hle_details": [],
        "hle_reports": [],
        "gaia_summaries": [],
        "browsecomp_summaries": []
    }

    if not output_path.exists():
        return results

    # Find HLE evaluation files
    for detail_file in output_path.rglob("*.eval_details.jsonl"):
        if "hle" in str(detail_file).lower():
            results["hle_details"].append(detail_file)

    for report_file in output_path.rglob("*.report.json"):
        if "hle" in str(report_file).lower():
            results["hle_reports"].append(report_file)

    # Find GAIA/BrowseComp summary files
    for summary_file in output_path.rglob("summary.jsonl"):
        parent = str(summary_file.parent)
        if "gaia" in parent.lower():
            results["gaia_summaries"].append(summary_file)
        elif "browsecomp" in parent.lower():
            results["browsecomp_summaries"].append(summary_file)

    return results


def analyze_hle_results(detail_files: List[Path], report_files: List[Path]) -> Dict[str, Any]:
    """Analyze HLE evaluation results."""
    if not report_files:
        return None

    # Use the first report file
    report_path = report_files[0]
    with open(report_path, 'r') as f:
        report = json.load(f)

    analysis = {
        "benchmark": "HLE (Humanity's Last Exam)",
        "total_questions": report.get("total", 0),
        "accuracy": report.get("accuracy", 0),
        "correct": report.get("correct", 0),
        "incorrect": report.get("incorrect", 0),
        "avg_input_tokens": report.get("avg_input_tokens", 0),
        "avg_output_tokens": report.get("avg_output_tokens", 0),
        "total_cost_estimate": report.get("total_cost", 0),
        "report_file": str(report_path)
    }

    # Analyze details if available
    if detail_files:
        detail_path = detail_files[0]
        correct_by_category = {}
        total_by_category = {}

        with open(detail_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                category = item.get("category", "Unknown")

                if category not in total_by_category:
                    total_by_category[category] = 0
                    correct_by_category[category] = 0

                total_by_category[category] += 1
                if item.get("is_correct", False):
                    correct_by_category[category] += 1

        analysis["by_category"] = {
            cat: {
                "total": total_by_category[cat],
                "correct": correct_by_category.get(cat, 0),
                "accuracy": correct_by_category.get(cat, 0) / total_by_category[cat]
            }
            for cat in total_by_category
        }

    return analysis


def analyze_gaia_browsecomp_results(summary_files: List[Path], benchmark_name: str) -> Dict[str, Any]:
    """Analyze GAIA or BrowseComp results."""
    if not summary_files:
        return None

    summary_path = summary_files[0]
    results = []

    with open(summary_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        return None

    # Extract metrics
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    accuracy = correct / total if total > 0 else 0

    analysis = {
        "benchmark": benchmark_name,
        "total_questions": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "summary_file": str(summary_path)
    }

    # Calculate tool usage statistics
    tool_counts = {}
    for result in results:
        tools_used = result.get("tools_used", [])
        for tool in tools_used:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    analysis["tool_usage"] = tool_counts

    # Calculate average rounds
    rounds = [r.get("num_rounds", 0) for r in results if "num_rounds" in r]
    if rounds:
        analysis["avg_rounds"] = statistics.mean(rounds)
        analysis["max_rounds"] = max(rounds)
        analysis["min_rounds"] = min(rounds)

    return analysis


def generate_markdown_report(analyses: Dict[str, Any], output_file: str):
    """Generate markdown report."""
    report_lines = []

    # Header
    report_lines.append("# DeepResearch Performance Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Overall Summary
    report_lines.append("## 📊 Overall Summary")
    report_lines.append("")
    report_lines.append("| Benchmark | Questions | Correct | Accuracy |")
    report_lines.append("|-----------|-----------|---------|----------|")

    total_questions = 0
    total_correct = 0

    for name, analysis in analyses.items():
        if analysis:
            total = analysis.get("total_questions", 0)
            correct = analysis.get("correct", 0)
            accuracy = analysis.get("accuracy", 0)

            total_questions += total
            total_correct += correct

            benchmark = analysis.get("benchmark", name)
            report_lines.append(f"| {benchmark} | {total} | {correct} | {accuracy:.2%} |")

    if total_questions > 0:
        overall_accuracy = total_correct / total_questions
        report_lines.append(f"| **Overall** | **{total_questions}** | **{total_correct}** | **{overall_accuracy:.2%}** |")

    report_lines.append("")

    # Detailed Results per Benchmark
    for name, analysis in analyses.items():
        if not analysis:
            continue

        report_lines.append("---")
        report_lines.append("")
        report_lines.append(f"## {analysis['benchmark']}")
        report_lines.append("")

        # Basic metrics
        report_lines.append("### Metrics")
        report_lines.append("")
        report_lines.append(f"- **Total Questions**: {analysis.get('total_questions', 'N/A')}")
        report_lines.append(f"- **Correct**: {analysis.get('correct', 'N/A')}")
        report_lines.append(f"- **Incorrect**: {analysis.get('incorrect', 'N/A')}")
        report_lines.append(f"- **Accuracy**: {analysis.get('accuracy', 0):.2%}")
        report_lines.append("")

        # Token usage (HLE specific)
        if "avg_input_tokens" in analysis:
            report_lines.append("### Token Usage")
            report_lines.append("")
            report_lines.append(f"- **Average Input Tokens**: {analysis.get('avg_input_tokens', 0):,.0f}")
            report_lines.append(f"- **Average Output Tokens**: {analysis.get('avg_output_tokens', 0):,.0f}")
            report_lines.append(f"- **Estimated Cost**: ${analysis.get('total_cost_estimate', 0):.2f}")
            report_lines.append("")

        # Tool usage (GAIA/BrowseComp specific)
        if "tool_usage" in analysis:
            report_lines.append("### Tool Usage")
            report_lines.append("")
            tool_usage = analysis["tool_usage"]
            for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"- **{tool}**: {count} times")
            report_lines.append("")

        # Rounds statistics
        if "avg_rounds" in analysis:
            report_lines.append("### Reasoning Rounds")
            report_lines.append("")
            report_lines.append(f"- **Average**: {analysis.get('avg_rounds', 0):.1f}")
            report_lines.append(f"- **Minimum**: {analysis.get('min_rounds', 0)}")
            report_lines.append(f"- **Maximum**: {analysis.get('max_rounds', 0)}")
            report_lines.append("")

        # Category breakdown (HLE specific)
        if "by_category" in analysis:
            report_lines.append("### Performance by Category")
            report_lines.append("")
            report_lines.append("| Category | Questions | Correct | Accuracy |")
            report_lines.append("|----------|-----------|---------|----------|")

            for category, stats in sorted(analysis["by_category"].items()):
                total = stats["total"]
                correct = stats["correct"]
                accuracy = stats["accuracy"]
                report_lines.append(f"| {category} | {total} | {correct} | {accuracy:.2%} |")

            report_lines.append("")

        # File references
        report_lines.append("### Files")
        report_lines.append("")
        if "report_file" in analysis:
            report_lines.append(f"- Report: `{analysis['report_file']}`")
        if "summary_file" in analysis:
            report_lines.append(f"- Summary: `{analysis['summary_file']}`")
        report_lines.append("")

    # Comparison with Published Results
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 📈 Comparison with Published Results")
    report_lines.append("")
    report_lines.append("| Benchmark | Your Score | Published Score | Difference |")
    report_lines.append("|-----------|------------|-----------------|------------|")

    # Published benchmarks from the paper
    published = {
        "HLE": 0.52,  # From the paper
        "GAIA": 0.41,  # Approximate from paper
        "BrowseComp": 0.68  # Approximate from paper
    }

    for name, analysis in analyses.items():
        if not analysis:
            continue

        benchmark = analysis.get("benchmark", "")
        your_score = analysis.get("accuracy", 0)

        for pub_name, pub_score in published.items():
            if pub_name.lower() in benchmark.lower():
                diff = your_score - pub_score
                diff_str = f"+{diff:.2%}" if diff >= 0 else f"{diff:.2%}"
                report_lines.append(f"| {pub_name} | {your_score:.2%} | {pub_score:.2%} | {diff_str} |")

    report_lines.append("")
    report_lines.append("*Note: Published scores are approximate from the technical report.*")
    report_lines.append("")

    # Recommendations
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 💡 Recommendations")
    report_lines.append("")

    for name, analysis in analyses.items():
        if not analysis:
            continue

        accuracy = analysis.get("accuracy", 0)
        benchmark = analysis.get("benchmark", name)

        if accuracy < 0.3:
            report_lines.append(f"- **{benchmark}**: Performance is significantly below expected. Check:")
            report_lines.append("  - API keys are configured correctly")
            report_lines.append("  - Tools (Search, Visit, etc.) are working properly")
            report_lines.append("  - Model is receiving proper prompts")
        elif accuracy < 0.5:
            report_lines.append(f"- **{benchmark}**: Performance below published baseline. Consider:")
            report_lines.append("  - Increasing temperature for more diverse reasoning")
            report_lines.append("  - Running multiple rollouts (Pass@3)")
            report_lines.append("  - Checking tool timeout settings")
        else:
            report_lines.append(f"- **{benchmark}**: Good performance! ✓")

    report_lines.append("")

    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate performance report")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Directory containing evaluation results (default: outputs)")
    parser.add_argument("--report_file", type=str, default="PERFORMANCE_REPORT.md",
                       help="Output report file (default: PERFORMANCE_REPORT.md)")

    args = parser.parse_args()

    print("="*60)
    print("DeepResearch Report Generator")
    print("="*60)

    print(f"\nSearching for evaluation results in: {args.output_dir}")
    results = find_evaluation_results(args.output_dir)

    print(f"\nFound results:")
    print(f"  HLE reports: {len(results['hle_reports'])}")
    print(f"  GAIA summaries: {len(results['gaia_summaries'])}")
    print(f"  BrowseComp summaries: {len(results['browsecomp_summaries'])}")

    if not any(results.values()):
        print("\n❌ No evaluation results found!")
        print("\nPlease run evaluations first:")
        print("  python scripts/run_evaluation.py --all")
        return

    print("\nAnalyzing results...")

    analyses = {}

    # Analyze HLE
    if results['hle_reports']:
        analyses['hle'] = analyze_hle_results(
            results['hle_details'],
            results['hle_reports']
        )
        print("  ✓ HLE analysis complete")

    # Analyze GAIA
    if results['gaia_summaries']:
        analyses['gaia'] = analyze_gaia_browsecomp_results(
            results['gaia_summaries'],
            "GAIA"
        )
        print("  ✓ GAIA analysis complete")

    # Analyze BrowseComp
    if results['browsecomp_summaries']:
        analyses['browsecomp'] = analyze_gaia_browsecomp_results(
            results['browsecomp_summaries'],
            "BrowseComp"
        )
        print("  ✓ BrowseComp analysis complete")

    print("\nGenerating report...")
    generate_markdown_report(analyses, args.report_file)

    print("\n✅ Report generation complete!")
    print(f"\nView your report: cat {args.report_file}")


if __name__ == "__main__":
    main()
