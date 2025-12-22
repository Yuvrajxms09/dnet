# Debug Logging Guide for Memory Error Tracing

This guide explains how to enable debug logging to trace memory errors in the Outlines grammar-constrained generation system.

## Quick Start

### Enable Debug Mode

Set the `DNET_LOG` environment variable to `DEBUG`:

```bash
# For local development
export DNET_LOG=DEBUG

# Or in your .env file
echo "DNET_LOG=DEBUG" >> .env

# For remote servers (SSH)
ssh user@remote-server "export DNET_LOG=DEBUG && /path/to/dnet-shard"
```

### Log File Locations

Logs are automatically written to files in addition to stdout:

- **API Server**: `~/.dria/dnet/logs/dnet-api.log`
- **Shard Server**: `~/.dria/dnet/logs/dnet-shard-{PID}.log`
- **Default**: `~/.dria/dnet/logs/dnet.log`

You can customize the log directory with:

```bash
export DNET_LOG_DIR=/path/to/custom/logs
```

## Log Levels

| Level | Environment Variable | Description |
|-------|---------------------|-------------|
| `DEBUG` | `DNET_LOG=DEBUG` | All logs including detailed memory traces |
| `INFO` | `DNET_LOG=INFO` | Default - includes memory trace markers |
| `WARNING` | `DNET_LOG=WARNING` | Warnings and errors only |
| `ERROR` | `DNET_LOG=ERROR` | Errors only |

## Memory Trace Logging

All memory-related operations are tagged with `[MEMORY TRACE]` prefix for easy filtering:

### Key Log Markers

- `[MEMORY TRACE]` = Info/debug logs about memory operations
- `[MEMORY TRACE] ❌` = Memory errors
- `[MEMORY TRACE] ✅` = Successful operations
- `[MEMORY TRACE] ⚠️` = Warnings about large allocations

### What Gets Logged

1. **Grammar State Creation**
   - Vocab size and estimated bitmask size
   - Regex pattern building
   - Vocabulary creation
   - Index/Guide creation

2. **Bitmask Allocation**
   - Before allocation: vocab size, estimated size in GB
   - During allocation: allocation call
   - After allocation: actual size and shape
   - On failure: error details with context

3. **Grammar State Usage**
   - Before creation: nonce, model vocab size, logits shape
   - During sampling: logits shape, grammar state status
   - After sampling: token ID and termination status

4. **Cleanup Operations**
   - Before cleanup: nonce
   - During cleanup: bitmask size, vocab size, each step
   - After cleanup: completion with freed memory size

## Filtering Logs

### View Only Memory Traces

```bash
# From log file
grep "\[MEMORY TRACE\]" ~/.dria/dnet/logs/dnet-shard-*.log

# From live output
dnet-shard 2>&1 | grep "\[MEMORY TRACE\]"
```

### View Only Errors

```bash
grep "\[MEMORY TRACE\] ❌" ~/.dria/dnet/logs/dnet-shard-*.log
```

### View Bitmask Allocations

```bash
grep "bitmask" ~/.dria/dnet/logs/dnet-shard-*.log | grep -i "allocate\|size"
```

## Remote Server Debugging

### Method 1: SSH with Environment Variable

```bash
ssh user@remote-server << 'EOF'
export DNET_LOG=DEBUG
export DNET_LOG_DIR=/tmp/dnet-debug-logs
/path/to/dnet-shard &
PID=$!
echo "Started dnet-shard with PID $PID"
echo "Logs: /tmp/dnet-debug-logs/dnet-shard-$PID.log"
EOF
```

### Method 2: Systemd Service Override

If running as a systemd service, create an override:

```bash
# On remote server
sudo systemctl edit dnet-shard.service
```

Add:
```ini
[Service]
Environment="DNET_LOG=DEBUG"
Environment="DNET_LOG_DIR=/var/log/dnet"
```

Then restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart dnet-shard
```

### Method 3: Docker/Container

```bash
docker run -e DNET_LOG=DEBUG -e DNET_LOG_DIR=/logs \
  -v /host/logs:/logs your-dnet-image
```

### Method 4: Remote Log Collection

```bash
# On remote server - tail logs in real-time
tail -f ~/.dria/dnet/logs/dnet-shard-*.log | grep "\[MEMORY TRACE\]"

# Or copy logs to local machine
scp user@remote-server:~/.dria/dnet/logs/dnet-shard-*.log ./debug-logs/
```

## Example: Tracing a Memory Error

When a memory error occurs, you'll see logs like:

```
[MEMORY TRACE] Creating grammar state: vocab_size=128256, estimated_bitmask_size=0.49GB
[MEMORY TRACE] ⚠️ Large vocabulary size (128256) will require ~0.5GB for grammar bitmask
[MEMORY TRACE] Attempting to allocate grammar bitmask: vocab_size=128256, estimated_size=0.49GB
[MEMORY TRACE] Calling bitmask_allocator for vocab_size=128256
[MEMORY TRACE] ❌ Bitmask allocation FAILED: vocab_size=128256, error=Attempting to allocate ~10.5 GB, estimated_size=0.49GB
```

This shows:
1. The estimated size (0.49GB) vs actual allocation attempt (10.5GB)
2. The vocab size causing the issue
3. The exact point of failure

## Troubleshooting

### Logs Not Appearing

1. Check log level is set correctly:
   ```bash
   echo $DNET_LOG  # Should output "DEBUG"
   ```

2. Check log directory exists and is writable:
   ```bash
   ls -la ~/.dria/dnet/logs/
   ```

3. Check file permissions:
   ```bash
   chmod -R 755 ~/.dria/dnet/logs/
   ```

### Too Many Logs

If DEBUG mode produces too many logs, use INFO level and filter:

```bash
export DNET_LOG=INFO
# Then filter for memory traces only
tail -f logs/*.log | grep "\[MEMORY TRACE\]"
```

### Remote Log Access

If you can't SSH, check if logs are accessible via:
- Web interface (if configured)
- Log aggregation service (e.g., ELK, Splunk)
- Shared filesystem mount
- Container logs: `docker logs <container-id>`

## Next Steps

After collecting debug logs, look for:
1. **Large vocab sizes** (>100k) causing large allocations
2. **Multiple grammar states** not being cleaned up
3. **Bitmask allocation failures** and their context
4. **Memory cleanup** timing and effectiveness

Use this information to:
- Identify the exact point of memory failure
- Determine if it's a vocab size issue or cleanup issue
- Plan migration to LLGuidance if needed

