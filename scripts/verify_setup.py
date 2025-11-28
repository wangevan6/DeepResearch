#!/usr/bin/env python3
"""
Verify DeepResearch setup and configuration.

This script checks that all prerequisites are met before running evaluations.

Usage:
    python scripts/verify_setup.py
"""

import os
import sys
from pathlib import Path
import subprocess


class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_header(text):
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠  {text}{Colors.NC}")


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor == 10:
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_warning(f"Python version: {version.major}.{version.minor}.{version.micro} (expected 3.10.x)")
        print("  Recommended: Use Python 3.10.0 to avoid dependency issues")
        return True  # Warning but not fatal


def check_conda_environment():
    """Check if running in conda environment."""
    conda_env = os.getenv("CONDA_DEFAULT_ENV")
    if conda_env:
        print_success(f"Conda environment: {conda_env}")
        if conda_env == "react_infer_env":
            return True
        else:
            print_warning("  Expected environment: react_infer_env")
            return True  # Warning but not fatal
    else:
        print_error("Not running in a conda environment")
        print("  Please run: conda activate react_infer_env")
        return False


def check_required_packages():
    """Check if required packages are installed."""
    required_packages = [
        ("openai", "OpenAI"),
        ("transformers", "Transformers"),
        ("qwen_agent", "Qwen-Agent"),
        ("datasets", "HuggingFace Datasets"),
        ("torch", "PyTorch"),
    ]

    all_installed = True
    for package_name, display_name in required_packages:
        try:
            __import__(package_name)
            print_success(f"{display_name} installed")
        except ImportError:
            print_error(f"{display_name} not installed")
            print(f"  Install with: pip install {package_name}")
            all_installed = False

    return all_installed


def check_env_file():
    """Check .env file exists and has required keys."""
    env_path = Path(".env")

    if not env_path.exists():
        print_error(".env file not found")
        print("  Create from template: cp .env.example .env")
        return False

    print_success(".env file exists")

    # Load and check required keys
    required_keys = [
        "OPENROUTER_API_KEY",
        "SERPER_KEY_ID",
        "JINA_API_KEYS",
        "API_KEY",
        "API_BASE",
    ]

    placeholder_values = [
        "your_openrouter_key",
        "your_key",
        "your_api_key",
        "/your/model/path",
        "your_dataset_name",
    ]

    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()

    all_configured = True
    for key in required_keys:
        if key not in env_vars:
            print_error(f"  Missing key: {key}")
            all_configured = False
        elif any(placeholder in env_vars[key] for placeholder in placeholder_values):
            print_warning(f"  {key} not configured (still has placeholder value)")
            all_configured = False
        else:
            print_success(f"  {key} configured")

    if not all_configured:
        print("\n  Please edit .env and add your actual API keys")
        print("  See API_KEYS_SETUP.md for instructions")

    return all_configured


def check_dataset_files():
    """Check if dataset files exist."""
    datasets = {
        "test_small": "inference/eval_data/test_small.jsonl",
        "browsecomp": "inference/eval_data/browsecomp_test.jsonl",
        "gaia": "inference/eval_data/gaia_test.jsonl",
        "hle": "inference/eval_data/hle_test.jsonl",
    }

    available = []
    missing = []

    for name, path in datasets.items():
        if Path(path).exists():
            print_success(f"{name}: {path}")
            available.append(name)
        else:
            print_warning(f"{name}: Not found - {path}")
            missing.append(name)

    if missing:
        print("\n  Missing datasets:")
        for name in missing:
            if name in ["gaia", "hle"]:
                print(f"    - {name}: Requires HuggingFace approval")
            else:
                print(f"    - {name}: Run: python scripts/download_and_prepare_datasets.py")

    return len(available) > 0


def check_directory_structure():
    """Check if necessary directories exist."""
    required_dirs = [
        "inference",
        "inference/eval_data",
        "evaluation",
        "scripts",
    ]

    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_success(f"Directory exists: {dir_path}")
        else:
            print_error(f"Directory missing: {dir_path}")
            all_exist = False

    return all_exist


def check_scripts_executable():
    """Check if scripts are executable."""
    scripts = [
        "inference/run_react_infer_openrouter.sh",
        "scripts/download_and_prepare_datasets.py",
        "scripts/run_evaluation.py",
        "scripts/generate_report.py",
    ]

    all_executable = True
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            if os.access(script_path, os.X_OK):
                print_success(f"Executable: {script}")
            else:
                print_warning(f"Not executable: {script}")
                print(f"  Fix with: chmod +x {script}")
        else:
            print_error(f"Script not found: {script}")
            all_executable = False

    return all_executable


def test_api_connection():
    """Test OpenRouter API connection."""
    print("\nTesting API connection...")

    # Check if OPENROUTER_API_KEY is set
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key or "your_openrouter_key" in api_key:
        print_warning("OPENROUTER_API_KEY not configured, skipping connection test")
        return True

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=10.0
        )

        # Simple test request
        response = client.chat.completions.create(
            model="alibaba/tongyi-deepresearch-30b-a3b",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )

        if response and response.choices:
            print_success("OpenRouter API connection successful")
            return True
        else:
            print_error("OpenRouter API connection failed: Empty response")
            return False

    except Exception as e:
        print_error(f"OpenRouter API connection failed: {str(e)}")
        print("  Check your OPENROUTER_API_KEY in .env")
        return False


def main():
    print_header("DeepResearch Setup Verification")

    checks = {}

    # Run all checks
    print("\n" + "-"*60)
    print("Checking Python and Environment")
    print("-"*60)
    checks["python_version"] = check_python_version()
    checks["conda_env"] = check_conda_environment()

    print("\n" + "-"*60)
    print("Checking Required Packages")
    print("-"*60)
    checks["packages"] = check_required_packages()

    print("\n" + "-"*60)
    print("Checking Configuration")
    print("-"*60)
    checks["env_file"] = check_env_file()

    print("\n" + "-"*60)
    print("Checking Directory Structure")
    print("-"*60)
    checks["directories"] = check_directory_structure()

    print("\n" + "-"*60)
    print("Checking Scripts")
    print("-"*60)
    checks["scripts"] = check_scripts_executable()

    print("\n" + "-"*60)
    print("Checking Datasets")
    print("-"*60)
    checks["datasets"] = check_dataset_files()

    # Optional API test
    if checks["env_file"] and checks["packages"]:
        print("\n" + "-"*60)
        print("Testing API Connection (Optional)")
        print("-"*60)
        checks["api_connection"] = test_api_connection()

    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)

    critical_checks = ["conda_env", "packages", "env_file", "directories"]
    critical_passed = all(checks.get(check, False) for check in critical_checks)

    total_checks = len(checks)
    passed_checks = sum(1 for v in checks.values() if v)

    print(f"\nPassed: {passed_checks}/{total_checks} checks")

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        check_name = check.replace("_", " ").title()
        print(f"  {status} {check_name}")

    print()

    if critical_passed:
        print_success("🎉 Setup verification passed!")
        print("\nYou're ready to run evaluations!")
        print("\nQuick start:")
        print("  1. Test with 5 questions:")
        print("     bash inference/run_react_infer_openrouter.sh")
        print("     (Update DATASET in .env to: inference/eval_data/test_small.jsonl)")
        print()
        print("  2. Run full pipeline:")
        print("     bash scripts/run_pipeline.sh --benchmark browsecomp")
        return 0
    else:
        print_error("Setup verification failed")
        print("\nPlease fix the issues above before proceeding")
        print("See SETUP_COMPLETE.md for detailed instructions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
