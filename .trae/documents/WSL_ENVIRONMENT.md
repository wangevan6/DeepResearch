# WSL Ubuntu 22.04 Environment Configuration

## Environment Details

**WSL Distribution**: Ubuntu-22.04
**Username**: evan_hero_linux
**Networking Mode**: Mirrored (configured in `.wslconfig`)
**Proxy Support**: ✅ Enabled (can access Windows localhost proxy)
**Last Updated**: 2025-11-13

---

## Proxy Configuration

### Windows Proxy Settings
Your Windows environment has a proxy configured at:
```
HTTP_PROXY=http://127.0.0.1:1235
HTTPS_PROXY=http://127.0.0.1:1235
```

### WSL Configuration File
**Location**: `C:\Users\user\.wslconfig`

**Contents**:
```ini
[wsl2]
# Enable mirrored networking mode to allow WSL to access Windows localhost
# This allows WSL applications to reach your Windows proxy at 127.0.0.1:1235
networkingMode=mirrored

# Improve DNS resolution (especially helpful with proxies)
dnsTunneling=true

# Automatically mirror Windows proxy settings to WSL
autoProxy=true
```

**What this does**:
- ✅ Allows WSL to access Windows `127.0.0.1` (including your proxy)
- ✅ Automatically copies Windows proxy settings into WSL environment
- ✅ Eliminates the "检测到 localhost 代理配置" warning message
- ✅ Enables DeepResearch in WSL to make API calls through the proxy

**To apply changes**: After modifying `.wslconfig`, run:
```bash
wsl.exe --shutdown
```
Then start WSL again (it auto-starts on first command).

---

## Miniconda Installation

### Miniconda Path
```bash
/home/evan_hero_linux/miniconda3
```

### Conda Binary Locations
```bash
# Main conda executable
/home/evan_hero_linux/miniconda3/bin/conda

# Conda activation script (RECOMMENDED for initialization)
/home/evan_hero_linux/miniconda3/etc/profile.d/conda.sh

# Condabin (used after conda.sh is sourced)
/home/evan_hero_linux/miniconda3/condabin/conda
```

### Base Python Version
```bash
Python 3.13.9
```

### ⚠️ Important: Conda is NOT in PATH by Default
When running WSL commands from Windows, conda is not automatically available. You must initialize it first!

---

## How to Initialize Conda (CRITICAL!)

### Method 1: Source conda.sh (RECOMMENDED)
This is the correct way to initialize conda in WSL:

```bash
# Source the conda initialization script
. /home/evan_hero_linux/miniconda3/etc/profile.d/conda.sh

# Or using 'source' (same thing)
source /home/evan_hero_linux/miniconda3/etc/profile.d/conda.sh

# After sourcing, conda commands work:
conda env list
conda activate react_infer_env
```

### Method 2: Use Full Path (Works but limited)
```bash
# Works for basic commands, but 'conda activate' won't work
~/miniconda3/bin/conda env list
~/miniconda3/bin/conda info
```

### Method 3: Login Shell (Auto-loads bashrc)
```bash
# The -l flag makes it a login shell, which loads ~/.bashrc
wsl.exe -d Ubuntu-22.04 bash -l -c "conda env list"
```

**Why conda isn't in PATH**:
- Conda is configured in `~/.bashrc` via `conda init`
- Non-login shells (like `wsl.exe ... bash -c`) don't load `.bashrc`
- Must manually source `conda.sh` or use login shell `-l` flag

---

## Conda Environments

### Available Environments

1. **base** (default)
   - Path: `/home/evan_hero_linux/miniconda3`
   - Python: 3.13.9

2. **arpo**
   - Path: `/home/evan_hero_linux/miniconda3/envs/arpo`

3. **react_infer_env** ⭐ (Primary environment for DeepResearch)
   - Path: `/home/evan_hero_linux/miniconda3/envs/react_infer_env`
   - Python: **3.10.0** (Required version for DeepResearch)
   - Python Binary: `/home/evan_hero_linux/miniconda3/envs/react_infer_env/bin/python`

---

## Activating the Environment

### From Windows (Git Bash/PowerShell) - CORRECT METHOD ✅

**Method 1: Using conda.sh (RECOMMENDED)**
```bash
# Single command - initialize conda and activate environment
wsl.exe -d Ubuntu-22.04 bash -c ". ~/miniconda3/etc/profile.d/conda.sh && conda activate react_infer_env && python --version"
```

**Method 2: Using login shell**
```bash
# The -l flag loads ~/.bashrc which initializes conda
wsl.exe -d Ubuntu-22.04 bash -l -c "conda activate react_infer_env && python --version"
```

**Method 3: Direct python path (no conda activation needed)**
```bash
# Use the environment's python directly
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/python --version"
```

### From WSL Ubuntu Terminal (Interactive Session)
```bash
# Option 1: Conda is already initialized in ~/.bashrc
conda activate react_infer_env
python --version
# Should output: Python 3.10.0

# Option 2: If conda not initialized, source it first
. ~/miniconda3/etc/profile.d/conda.sh
conda activate react_infer_env
python --version
```

---

## Which Python to Use?

### ❌ WRONG - These won't work or will use wrong version:
```bash
# This doesn't exist or uses wrong version
which python          # May not be found or use system python
python --version      # Won't work if not in PATH

# This is base environment (Python 3.13.9) - WRONG VERSION!
~/miniconda3/bin/python --version
```

### ✅ CORRECT - Use one of these methods:

**Method 1: Activate environment first (RECOMMENDED)**
```bash
# Initialize conda
. ~/miniconda3/etc/profile.d/conda.sh

# Activate react_infer_env
conda activate react_infer_env

# Now 'python' points to correct version (3.10.0)
python --version
which python  # Shows: /home/evan_hero_linux/miniconda3/envs/react_infer_env/bin/python
```

**Method 2: Use direct path (no activation needed)**
```bash
# Always points to Python 3.10.0
~/miniconda3/envs/react_infer_env/bin/python --version
~/miniconda3/envs/react_infer_env/bin/python your_script.py
```

**Method 3: Full absolute path**
```bash
/home/evan_hero_linux/miniconda3/envs/react_infer_env/bin/python --version
```

---

## Quick Commands (From Windows)

### Check Python Version in react_infer_env ✅
```bash
# Method 1: Direct path (fastest, no conda needed)
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/python --version"

# Method 2: With conda activation
wsl.exe -d Ubuntu-22.04 bash -c ". ~/miniconda3/etc/profile.d/conda.sh && conda activate react_infer_env && python --version"
```

### List All Conda Environments ✅
```bash
# Method 1: Direct path
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/bin/conda env list"

# Method 2: With conda initialization
wsl.exe -d Ubuntu-22.04 bash -c ". ~/miniconda3/etc/profile.d/conda.sh && conda env list"

# Method 3: Login shell
wsl.exe -d Ubuntu-22.04 bash -l -c "conda env list"
```

### Run Python Script in react_infer_env ✅
```bash
# Method 1: Direct path (RECOMMENDED - most reliable)
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/python your_script.py"

# Method 2: With conda activation
wsl.exe -d Ubuntu-22.04 bash -c ". ~/miniconda3/etc/profile.d/conda.sh && conda activate react_infer_env && python your_script.py"

# Method 3: Login shell with activation
wsl.exe -d Ubuntu-22.04 bash -l -c "conda activate react_infer_env && python your_script.py"
```

### Install Package in react_infer_env ✅
```bash
# Method 1: Using pip directly
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/pip install package_name"

# Method 2: Using conda
wsl.exe -d Ubuntu-22.04 bash -c ". ~/miniconda3/etc/profile.d/conda.sh && conda activate react_infer_env && pip install package_name"
```

---

## Running DeepResearch Inference on WSL

### Method 1: Direct Execution (One Command) ✅
```bash
# CORRECT: Initialize conda properly
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && . ~/miniconda3/etc/profile.d/conda.sh && conda activate react_infer_env && bash inference/run_react_infer_openrouter.sh"
```

### Method 2: Using Direct Python Path ✅
```bash
# No conda activation needed - use python directly
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && ~/miniconda3/envs/react_infer_env/bin/python inference/run_multi_react.py"
```

### Method 3: Login Shell ✅
```bash
# Login shell automatically loads conda from ~/.bashrc
wsl.exe -d Ubuntu-22.04 bash -l -c "cd /mnt/c/Users/user/Projects/DeepResearch && conda activate react_infer_env && bash inference/run_react_infer_openrouter.sh"
```

### Method 4: Interactive WSL Session ✅
```bash
# Open WSL terminal
wsl.exe -d Ubuntu-22.04

# In WSL terminal (conda already initialized from ~/.bashrc):
cd /mnt/c/Users/user/Projects/DeepResearch
conda activate react_infer_env
bash inference/run_react_infer_openrouter.sh
```

### Installing Requirements ✅
```bash
# Method 1: Direct pip path (RECOMMENDED)
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && ~/miniconda3/envs/react_infer_env/bin/pip install -r requirements.txt"

# Method 2: With conda activation
wsl.exe -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/user/Projects/DeepResearch && . ~/miniconda3/etc/profile.d/conda.sh && conda activate react_infer_env && pip install -r requirements.txt"

# Method 3: Interactive session
wsl.exe -d Ubuntu-22.04
cd /mnt/c/Users/user/Projects/DeepResearch
conda activate react_infer_env
pip install -r requirements.txt
```

---

## Important Notes

1. **Python Version Requirement**: DeepResearch strictly requires Python 3.10.0
   - Base environment has Python 3.13.9 (incompatible)
   - **Always use `react_infer_env`** for DeepResearch work

2. **Path Conversion**: When working from Windows:
   - Windows path: `C:\Users\user\Projects\DeepResearch`
   - WSL path: `/mnt/c/Users/user/Projects/DeepResearch`

3. **Conda Activation**: Must activate conda base first, then the specific environment:
   ```bash
   source ~/miniconda3/bin/activate  # First
   conda activate react_infer_env    # Then
   ```

4. **Command Execution from Windows**:
   - Use `wsl.exe -d Ubuntu-22.04 bash -c "commands"`
   - Always use absolute paths or cd to correct directory
   - Chain commands with `&&` to ensure proper execution order

---

## Environment Variables

When running in WSL, ensure these are set (usually in `.env` file):

```bash
# API Keys
OPENROUTER_API_KEY=your_key_here
SERPER_KEY_ID=your_key_here
JINA_API_KEYS=your_key_here
API_KEY=your_openai_key_here
DASHSCOPE_API_KEY=your_key_here

# Paths (use WSL paths, not Windows paths)
MODEL_PATH=/path/to/model
DATASET=/mnt/c/Users/user/Projects/DeepResearch/inference/eval_data/test_small.jsonl
OUTPUT_PATH=/mnt/c/Users/user/Projects/DeepResearch/outputs
```

---

## Troubleshooting

### Issue: "conda: command not found"
**Solution**: Conda is not in PATH by default. Use full path:
```bash
~/miniconda3/bin/conda env list
```

### Issue: "python: command not found"
**Solution**: Base Python might not be in PATH. Activate environment or use full path:
```bash
~/miniconda3/envs/react_infer_env/bin/python --version
```

### Issue: Wrong Python Version
**Solution**: Verify you're in the correct environment:
```bash
conda activate react_infer_env
python --version  # Should show 3.10.0
```

### Issue: WSL Commands from Windows Show Encoding Errors
**Solution**: Use `bash -c` with proper quoting:
```bash
wsl.exe -d Ubuntu-22.04 bash -c "your commands here"
```

### Issue: "wsl: 检测到 localhost 代理配置，但未镜像到 WSL" Warning
**Translation**: "WSL: detected localhost proxy configuration, but not mirrored to WSL"

**Why it happens**:
- Windows has proxy at `127.0.0.1:1235`
- WSL in NAT mode cannot access Windows localhost
- Warning appears but WSL commands still work (just without proxy access)

**Solution**: ✅ **Already Fixed!**
The `.wslconfig` file has been created with `networkingMode=mirrored` which resolves this issue. If you see this warning again:
1. Verify `C:\Users\user\.wslconfig` exists
2. Run `wsl.exe --shutdown` to restart WSL
3. Test again - warning should be gone

---

## Verification Checklist

Run these commands to verify environment is correct:

```bash
# 1. Check WSL distribution
wsl.exe -d Ubuntu-22.04 bash -c "lsb_release -a"
# Should show: Ubuntu 22.04

# 2. Check miniconda location
wsl.exe -d Ubuntu-22.04 bash -c "test -d ~/miniconda3 && echo 'Found' || echo 'Not found'"
# Should show: Found

# 3. Check react_infer_env exists
wsl.exe -d Ubuntu-22.04 bash -c "test -d ~/miniconda3/envs/react_infer_env && echo 'Found' || echo 'Not found'"
# Should show: Found

# 4. Check Python version in react_infer_env
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/python --version"
# Should show: Python 3.10.0

# 5. List all environments
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/bin/conda env list"
# Should list: base, arpo, react_infer_env
```

---

## Summary

✅ **Miniconda Location**: `/home/evan_hero_linux/miniconda3`
✅ **Target Environment**: `react_infer_env`
✅ **Python Version**: 3.10.0 (Correct!)
✅ **WSL Distribution**: Ubuntu-22.04
✅ **Username**: evan_hero_linux
✅ **Networking**: Mirrored mode enabled
✅ **Proxy**: Windows proxy accessible from WSL
✅ **Warnings**: Proxy warning eliminated

**Status**: Environment is properly configured and ready for DeepResearch inference! 🚀

## What We Fixed

**Problem**: WSL proxy warning message appearing every time you run WSL commands
**Root Cause**: WSL in NAT mode couldn't access Windows localhost proxy at `127.0.0.1:1235`
**Solution**: Created `C:\Users\user\.wslconfig` with mirrored networking mode
**Result**:
- ✅ Warning eliminated
- ✅ WSL can now access Windows proxy
- ✅ DeepResearch API calls will work through the proxy
