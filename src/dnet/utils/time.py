import time


def utc_epoch_now() -> int:
    """Get current UTC epoch time in milliseconds.

    High-resolution UTC epoch in milliseconds as int.
    Previous implementation used whole seconds, which quantized
    transport timing to ~1000 ms buckets and obscured latency.

    Returns:
        Current time in milliseconds since epoch
    """
    return int(time.time() * 1000)
