"""
Tool Execution Logger for DeepResearch Inference

This module provides comprehensive logging for each tool call during inference,
capturing full input/output, timing, and success status.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List


class ToolLogger:
    """
    Logger for capturing detailed tool execution information.

    Creates per-question directories with individual JSON files for each tool call,
    plus a summary file with aggregated statistics.
    """

    def __init__(self, base_dir: str, question_index: int, question_text: str = ""):
        """
        Initialize the tool logger.

        Args:
            base_dir: Base directory for tool logs (e.g., outputs/.../tool_logs/)
            question_index: Index of the question being processed
            question_text: The question text (for reference in logs)
        """
        self.base_dir = Path(base_dir)
        self.question_index = question_index
        self.question_text = question_text
        self.question_dir = self.base_dir / f"q_{question_index:04d}"
        self.question_dir.mkdir(parents=True, exist_ok=True)

        self.round_num = 0
        self.tool_call_num = 0
        self.tool_calls: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def log_tool_call(
        self,
        round_num: int,
        tool_name: str,
        tool_input: Any,
        tool_output: str,
        duration: float,
        success: bool,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Log a single tool call execution.

        Args:
            round_num: The current round number in the ReAct loop
            tool_name: Name of the tool being called
            tool_input: Input arguments passed to the tool
            tool_output: Output returned by the tool
            duration: Execution time in seconds
            success: Whether the tool execution succeeded
            error: Error message if the execution failed

        Returns:
            The log entry dictionary
        """
        self.tool_call_num += 1

        # Serialize input, handling circular references and complex objects
        try:
            if isinstance(tool_input, dict):
                # Remove 'params' key if it exists to avoid circular reference
                serialized_input = {k: v for k, v in tool_input.items() if k != 'params'}
            else:
                serialized_input = str(tool_input)
        except Exception:
            serialized_input = str(tool_input)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question_index": self.question_index,
            "round": round_num,
            "tool_call_index": self.tool_call_num,
            "tool_name": tool_name,
            "input": serialized_input,
            "output": tool_output,
            "output_length": len(tool_output) if tool_output else 0,
            "duration_seconds": round(duration, 4),
            "success": success,
            "error": error
        }

        # Write individual log file
        filename = f"r{round_num:02d}_t{self.tool_call_num:02d}_{tool_name}.json"
        log_path = self.question_dir / filename

        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ToolLogger] Warning: Failed to write log file {log_path}: {e}")

        self.tool_calls.append(log_entry)
        return log_entry

    def write_summary(
        self,
        question: str,
        answer: str,
        prediction: str,
        termination: str,
        total_rounds: int
    ) -> None:
        """
        Write a summary file for this question's tool usage.

        Args:
            question: The original question text
            answer: The reference answer (if available)
            prediction: The model's predicted answer
            termination: Reason for termination (e.g., 'answer', 'timeout')
            total_rounds: Total number of ReAct rounds
        """
        total_duration = time.time() - self.start_time

        summary = {
            "question_index": self.question_index,
            "question": question,
            "reference_answer": answer,
            "prediction": prediction,
            "termination_reason": termination,
            "total_rounds": total_rounds,
            "total_tool_calls": len(self.tool_calls),
            "total_duration_seconds": round(total_duration, 2),
            "tool_usage": self._compute_tool_stats(),
            "tool_calls_timeline": [
                {
                    "round": tc["round"],
                    "tool": tc["tool_name"],
                    "duration": tc["duration_seconds"],
                    "success": tc["success"],
                    "output_length": tc["output_length"]
                } for tc in self.tool_calls
            ]
        }

        summary_path = self.question_dir / "summary.json"
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ToolLogger] Warning: Failed to write summary file: {e}")

    def _compute_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Compute aggregated statistics per tool type."""
        stats: Dict[str, Dict[str, Any]] = {}

        for tc in self.tool_calls:
            name = tc["tool_name"]
            if name not in stats:
                stats[name] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "avg_duration": 0.0,
                    "failures": 0,
                    "total_output_length": 0
                }
            stats[name]["count"] += 1
            stats[name]["total_duration"] += tc["duration_seconds"]
            stats[name]["total_output_length"] += tc["output_length"]
            if not tc["success"]:
                stats[name]["failures"] += 1

        # Calculate averages
        for name in stats:
            if stats[name]["count"] > 0:
                stats[name]["avg_duration"] = round(
                    stats[name]["total_duration"] / stats[name]["count"], 4
                )
                stats[name]["total_duration"] = round(stats[name]["total_duration"], 4)

        return stats


def write_aggregate_stats(tool_logs_dir: str) -> None:
    """
    Scan all question directories and write aggregate statistics.

    Args:
        tool_logs_dir: Path to the tool_logs directory
    """
    tool_logs_path = Path(tool_logs_dir)
    if not tool_logs_path.exists():
        return

    all_summaries = []
    total_tool_calls = 0
    total_duration = 0.0
    tool_totals: Dict[str, Dict[str, Any]] = {}

    # Scan all q_* directories
    for q_dir in sorted(tool_logs_path.glob("q_*")):
        summary_file = q_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    all_summaries.append({
                        "question_index": summary.get("question_index"),
                        "total_rounds": summary.get("total_rounds"),
                        "total_tool_calls": summary.get("total_tool_calls"),
                        "duration": summary.get("total_duration_seconds"),
                        "termination": summary.get("termination_reason")
                    })
                    total_tool_calls += summary.get("total_tool_calls", 0)
                    total_duration += summary.get("total_duration_seconds", 0)

                    # Aggregate tool usage
                    for tool_name, tool_stats in summary.get("tool_usage", {}).items():
                        if tool_name not in tool_totals:
                            tool_totals[tool_name] = {
                                "count": 0,
                                "total_duration": 0.0,
                                "failures": 0
                            }
                        tool_totals[tool_name]["count"] += tool_stats.get("count", 0)
                        tool_totals[tool_name]["total_duration"] += tool_stats.get("total_duration", 0)
                        tool_totals[tool_name]["failures"] += tool_stats.get("failures", 0)
            except Exception as e:
                print(f"[ToolLogger] Warning: Failed to read {summary_file}: {e}")

    # Calculate averages for tool totals
    for tool_name in tool_totals:
        if tool_totals[tool_name]["count"] > 0:
            tool_totals[tool_name]["avg_duration"] = round(
                tool_totals[tool_name]["total_duration"] / tool_totals[tool_name]["count"], 4
            )

    aggregate = {
        "generated_at": datetime.now().isoformat(),
        "total_questions": len(all_summaries),
        "total_tool_calls": total_tool_calls,
        "total_duration_seconds": round(total_duration, 2),
        "avg_tool_calls_per_question": round(total_tool_calls / len(all_summaries), 2) if all_summaries else 0,
        "avg_duration_per_question": round(total_duration / len(all_summaries), 2) if all_summaries else 0,
        "tool_usage_totals": tool_totals,
        "questions_summary": all_summaries
    }

    aggregate_path = tool_logs_path / "aggregate_stats.json"
    try:
        with open(aggregate_path, 'w', encoding='utf-8') as f:
            json.dump(aggregate, f, indent=2, ensure_ascii=False)
        print(f"[ToolLogger] Aggregate stats written to {aggregate_path}")
    except Exception as e:
        print(f"[ToolLogger] Warning: Failed to write aggregate stats: {e}")
