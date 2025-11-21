from dataclasses import dataclass


@dataclass
class ApiConfig:
    http_port: int
    grpc_port: int
    compression_pct: float = 0.0


@dataclass
class ClusterConfig:
    discovery_port: int = 0  # 0 means dynamic/default


@dataclass
class InferenceConfig:
    max_concurrent_requests: int = 100
