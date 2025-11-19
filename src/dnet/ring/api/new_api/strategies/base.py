from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dnet_p2p import DnetDeviceProperties, ThunderboltConnection
from distilp.common import DeviceProfile
from dnet.core.types.topology import TopologyInfo

@dataclass
class TokenResult:
    token_id: int
    logprob: float = 0.0
    top_logprobs: Dict[int, float] = field(default_factory=dict)

class ApiAdapterBase(ABC):
    """Abstract base class for API-Shard communication adapters."""
    
    def __init__(self) -> None:
        self.running = False

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    async def connect_first_shard(self, ip: str, port: int) -> None: ...

    @abstractmethod
    async def reset_cache(self) -> None: ...

    @abstractmethod
    async def send_tokens(
        self, 
        nonce: str, 
        tokens: bytes, 
        callback_addr: str,
        logprobs: bool = False,
        top_logprobs: int = 0
    ) -> None: ...

    @abstractmethod
    async def await_token(self, nonce: str, timeout_s: float) -> TokenResult: ...

    @abstractmethod
    def resolve_token(self, nonce: str, result: TokenResult) -> None: ...


class TopologySolver(ABC):
    """Abstract base class for topology solvers."""

    @abstractmethod
    async def solve(
        self,
        profiles: Dict[str, DeviceProfile],
        model_profile: Any,  # ModelProfile
        model_name: str,
        num_layers: int,
        kv_bits: str,
        shards: Dict[str, DnetDeviceProperties],
        thunderbolts: Dict[str, Dict[str, ThunderboltConnection]],
    ) -> TopologyInfo:
        """
        Computes the topology (layer assignments) for the given cluster and model.
        """
        pass


class Strategy(ABC):
    """
    Bundles a TopologySolver and an ApiAdapterBase for a specific execution strategy.
    """
    @property
    @abstractmethod
    def solver(self) -> TopologySolver: ...

    @property
    @abstractmethod
    def adapter(self) -> ApiAdapterBase: ...
