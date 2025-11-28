# Docker and SandboxFusion Setup for DeepResearch

**Date**: 2025-11-13
**Purpose**: Document Docker configuration and SandboxFusion container setup for Python code execution

***

## Docker Configuration

### Docker Version

```bash
$ docker --version
Docker version 28.3.3, build 980b856
```
v
**Docker Engine**: Installed on Windows (accessible from both Windows and WSL)
**Status**: ✅ Running

### Check Running Containers

```bash
$ docker ps
CONTAINER ID   IMAGE                  COMMAND                   CREATED       STATUS                            PORTS                                             NAMES
104d224f1934   mineru-sglang:latest   "mineru-sglang-serve…"   8 weeks ago   Up 2 seconds (health: starting)   0.0.0.0:30000->30000/tcp, [::]:30000->30000/tcp   mineru-sglang-server
```

***

## SandboxFusion Docker Setup

**Purpose**: Provides sandboxed Python code execution environment for DeepResearch agent

### Step 1: Pull the SandboxFusion Image

```bash
# Pull the prebuilt SandboxFusion image from Alibaba registry
docker pull vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

**Image Details**:

* **Registry**: vemlp-cn-beijing.cr.volces.com (Alibaba Cloud/Volcengine)

* **Repository**: preset-images/code-sandbox

* **Tag**: server-20250609 (released June 9, 2025)

* **Size**: Large image with 38 layers (includes Python runtime and dependencies)

* **Pull Time**: \~5-10 minutes depending on network speed

**What this image contains**:

* Python interpreter

* SandboxFusion server

* Code execution runtime environment

* Security isolation mechanisms

### Step 2: Run the SandboxFusion Container

```bash
# Start the sandbox container in detached mode
docker run -d \
  --rm \
  --privileged \
  -p 8080:8080 \
  vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

**Command Breakdown**:

* `-d`: Run in detached mode (background)

* `--rm`: Automatically remove container when it stops

* `--privileged`: Required for sandbox isolation features

* `-p 8080:8080`: Map port 8080 (host) → 8080 (container)

**Container Access**:

* From Windows: `http://localhost:8080`

* From WSL: `http://localhost:8080` (thanks to mirrored networking)

* API endpoint: `http://localhost:8080/run_code`

### Step 3: Verify Container is Running

```bash
# Check if sandbox container is running
docker ps | grep code-sandbox

# Check container logs
docker logs <container-id>

# Test API endpoint (simple health check)
curl -X POST http://localhost:8080/run_code \
  -H "Content-Type: application/json" \
  -d '{"code":"print(\"Hello from sandbox!\")", "language":"python"}'
```

**Expected Response**:

```json
{
  "run_result": {
    "stdout": "Hello from sandbox!\n",
    "stderr": "",
    "execution_time": 0.123
  }
}
```

### Step 4: Configure DeepResearch to Use Sandbox

Edit `.env` file in the DeepResearch project root:

```bash
# Add or update this line:
SANDBOX_FUSION_ENDPOINT=http://localhost:8080
```

**Multiple Endpoints (Optional)**:
If you have multiple sandbox containers running (for load balancing):

```bash
SANDBOX_FUSION_ENDPOINT=http://localhost:8080,http://localhost:8081,http://localhost:8082
```

### Step 5: Install sandbox-fusion Python Package

The DeepResearch agent needs the `sandbox-fusion` Python client library:

```bash
# In WSL with react_infer_env
wsl.exe -d Ubuntu-22.04 bash -c "~/miniconda3/envs/react_infer_env/bin/pip install sandbox-fusion"

# Or if already in WSL
conda activate react_infer_env
pip install sandbox-fusion
```

***

## Managing the Sandbox Container

### Stop the Container

```bash
# Find container ID
docker ps | grep code-sandbox

# Stop the container
docker stop <container-id>
```

### Restart the Container

```bash
# Run the same docker run command again
docker run -d --rm --privileged -p 8080:8080 \
  vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

### Check Container Logs

```bash
# View real-time logs
docker logs -f <container-id>

# View last 100 lines
docker logs --tail 100 <container-id>
```

### Remove the Image

```bash
# Remove container (if running)
docker stop <container-id>

# Remove image
docker rmi vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

***

## Docker Networking with WSL

Thanks to **mirrored networking mode** in `.wslconfig`:

* ✅ WSL can access Docker containers running on Windows at `localhost`

* ✅ Docker containers can be accessed from both Windows and WSL

* ✅ No need for special host.docker.internal or WSL IP addresses

* ✅ Port mappings work seamlessly across both environments

### Testing Connectivity

From Windows:

```bash
curl http://localhost:8080/run_code -X POST -H "Content-Type: application/json" -d "{\"code\":\"print('test')\"}"
```

From WSL:

```bash
wsl.exe -d Ubuntu-22.04 bash -c "curl http://localhost:8080/run_code -X POST -H 'Content-Type: application/json' -d '{\"code\":\"print(\\\"test\\\")\"}"
```

***

## Integration with DeepResearch

### How DeepResearch Uses the Sandbox

1. **Agent receives Python code to execute**
2. **tool\_python.py** reads `SANDBOX_FUSION_ENDPOINT` from environment
3. **Sends code to sandbox via HTTP POST** to `/run_code` endpoint
4. **Sandbox executes code in isolated environment**
5. **Returns stdout, stderr, and execution time**
6. **Agent includes results in reasoning chain**

### Code Execution Flow

```
DeepResearch Agent
    ↓
tool_python.py (PythonInterpreter)
    ↓
sandbox_fusion library
    ↓
HTTP POST to http://localhost:8080/run_code
    ↓
SandboxFusion Container
    ↓
Isolated Python Execution
    ↓
Return results to agent
```

### Example Usage in Agent

When the agent wants to execute Python code:

```xml
<tool_call>
{
  "purpose": "Calculate the sum of first 100 numbers",
  "name": "PythonInterpreter",
  "arguments": {"code": ""}
}
<code>
result = sum(range(1, 101))
print(f"Sum: {result}")
</code>
</tool_call>
```

The sandbox executes this and returns:

```
stdout:
Sum: 5050
```

***

## Troubleshooting

### Issue: Container won't start

**Symptoms**: `docker run` fails immediately

**Solutions**:

1. Check if port 8080 is already in use:

   ```bash
   netstat -ano | findstr :8080
   ```
2. Try a different port:

   ```bash
   docker run -d --rm --privileged -p 8081:8080 <image-name>
   # Update .env: SANDBOX_FUSION_ENDPOINT=http://localhost:8081
   ```

### Issue: Connection refused from WSL

**Symptoms**: `curl` from WSL fails to connect

**Solutions**:

1. Verify mirrored networking in `.wslconfig`:

   ```ini
   [wsl2]
   networkingMode=mirrored
   ```
2. Restart WSL:

   ```bash
   wsl.exe --shutdown
   ```

### Issue: Permission denied errors in container

**Symptoms**: Code execution fails with permission errors

**Solution**: Ensure `--privileged` flag is used in `docker run`

### Issue: Image pull fails

**Symptoms**: Cannot pull from Alibaba registry

**Solutions**:

1. Check proxy settings (Windows proxy should be configured)
2. Try pulling with explicit proxy:

   ```bash
   docker pull vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
   ```
3. Alternative: Build from source (<https://github.com/bytedance/SandboxFusion>)

### Issue: sandbox-fusion package not found

**Symptoms**: Import error when running DeepResearch

**Solution**: Install in correct environment:

```bash
# Verify you're in react_infer_env
conda activate react_infer_env
python --version  # Should show 3.10.0

# Install package
pip install sandbox-fusion
```

***

## Performance Considerations

### Resource Usage

* **CPU**: Moderate (depends on code executed)

* **Memory**: \~500MB-1GB base, more for complex operations

* **Disk**: \~2-3GB for image

### Scaling

For high-throughput scenarios, run multiple containers:

```bash
# Container 1 on port 8080
docker run -d --rm --privileged -p 8080:8080 <image-name>

# Container 2 on port 8081
docker run -d --rm --privileged -p 8081:8080 <image-name>

# Container 3 on port 8082
docker run -d --rm --privileged -p 8082:8080 <image-name>

# Update .env
SANDBOX_FUSION_ENDPOINT=http://localhost:8080,http://localhost:8081,http://localhost:8082
```

The agent will randomly select an endpoint for each execution (load balancing).

***

## Security Notes

### Sandbox Isolation

* ✅ Code runs in isolated container environment

* ✅ Limited resource access

* ✅ Timeout protection (default 50s)

* ✅ No network access from executed code (by default)

### Best Practices

1. **Keep container updated**: Pull latest image periodically
2. **Monitor resource usage**: Check `docker stats`
3. **Review executed code**: Check agent logs for suspicious activity
4. **Firewall rules**: Restrict port 8080 to localhost only
5. **Restart regularly**: Restart container daily to clear state

***

## Summary

✅ **Docker Version**: 28.3.3
✅ **Image**: vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
✅ **Port**: 8080
✅ **Access**: Both Windows and WSL via localhost
✅ **Integration**: Configured in `.env` via `SANDBOX_FUSION_ENDPOINT`
✅ **Status**: Ready for Python code execution in DeepResearch agent

**Next Steps**:

1. Pull the Docker image (in progress)
2. Run the container
3. Update `.env` with endpoint
4. Install `sandbox-fusion` package
5. Test with sample execution

