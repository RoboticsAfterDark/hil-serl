# Troubleshooting

## Docker build times out downloading Nvidia CUDA wheels

**Problem:** `docker build` on `Dockerfile.isaac` fails through a `uv pip install` step with an error like:

```
error: Failed to download `nvidia-cublas-cu12==12.8.4.1`
cause: Request failed after 3 retries in 130.6s
cause: ... operation timed out
```

**Cause:** `uv`'s default HTTP read timeout is 30 seconds, which is too short for these on a slow or unstable connection.

**Fix:** set `UV_HTTP_TIMEOUT` to a higher value (in seconds) near the top of `Dockerfile.isaac`, before any `RUN uv pip install` line:

```dockerfile
ENV UV_HTTP_TIMEOUT=300
```
