from typing import Optional
from hypercorn import Config
import hypercorn.asyncio as aio_hypercorn
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ....utils.logger import logger
from .shard import Shard
from ..models import (
    HealthResponse,
    MeasureLatencyRequest,
    MeasureLatencyResponse,
    ShardLoadModelRequest,
    ShardLoadModelResponse,
    ShardProfileRequest,
    ShardProfileResponse,
    ShardUnloadModelResponse,
)

class HTTPServer:
    """
    HTTP API server for shard node.
    """
    def __init__(self, http_port: int, shard: Shard) -> None:
        self.shard = shard
        self.http_port: int = http_port
        self.app = FastAPI()
        self.http_server: Optional[asyncio.Task] = None

    async def _start_http_server(self, shutdown_trigger: asyncio.Event) -> None:
        await self._setup_routes()

        # Start HTTP server in background
        config = Config.from_mapping(
            bind=f"0.0.0.0:{self.http_port}",
            log_level="info",
            log_config=None,
            use_reloader=False,
            h2c=False,
        )

        self.http_server = asyncio.create_task(
            aio_hypercorn.serve(self.app, config, shutdown_trigger=shutdown_trigger)  # type: ignore
        )

    async def _setup_routes(self) -> None:
        """Setup HTTP routes."""

        @self.app.get("/health")
        async def health() -> HealthResponse:
            try:
                instance = self.shard.discovery.instance_name()
            except Exception:
                instance = None
            return HealthResponse(
                status="ok",
                node_id=self.shard.node_id,
                running=self.shard.running,
                model_loaded=self.shard.runtime.model is not None,
                model_path=self.shard.runtime.model_path,
                assigned_layers=self.assigned_layers,
                queue_size=self.shard.runtime.queue_size(),
                grpc_port=self.shard.grpc_server.grpc_port,
                http_port=self.http_port,
                instance=instance,
            )

        @self.app.post("/profile")
        async def profile(req: ShardProfileRequest) -> ShardProfileResponse:
            logger.info("Received /profile request")
            try:
                device_profile = await self._profile_device(
                    req.repo_id, req.max_batch_exp
                )

                return ShardProfileResponse(profile=device_profile)
            except Exception as e:
                logger.error(f"Error in /profile endpoint: {e}")
                raise

        @self.app.post("/measure_latency")
        async def measure_latency(
            req: MeasureLatencyRequest,
        ) -> MeasureLatencyResponse:
            logger.info("Received /measure_latency request")
            try:
                # Measure latencies to other devices
                latency_results = await self._measure_latency_to_devices(
                    req.devices, req.thunderbolts, req.payload_sizes
                )

                return MeasureLatencyResponse(latency=latency_results)
            except Exception as e:
                logger.error(f"Error in /measure_latency endpoint: {e}")
                raise

        @self.app.post("/load_model")
        async def load_model_endpoint(
            req: ShardLoadModelRequest,
        ) -> ShardLoadModelResponse:
            """Load model with specified layers."""
            try:
                logger.info(
                    f"HTTP /load_model: model={req.model_path}, layers={req.layers}, "
                    f"next_node={req.next_node or 'none'}, window_size={req.window_size}, "
                    f"total_layers={req.total_layers}, kv_bits={req.kv_bits or 'default'}, "
                    f"api_callback={req.api_callback_address or 'none'}"
                )
                result = await self.shard.load_model(req)
                return result

            except Exception as e:
                logger.error(f"Error in /load_model endpoint: {e}")
                return ShardLoadModelResponse(
                    success=False,
                    message=f"Error: {str(e)}",
                    layers_loaded=[],
                    load_time_ms=0.0,
                )

        @self.app.post("/unload_model")
        async def unload_model_endpoint() -> ShardUnloadModelResponse:
            """Unload current model."""
            try:
                logger.info("HTTP /unload_model")
                result = await self.shard.unload_model()
                return result

            except Exception as e:
                logger.error(f"Error in /unload_model endpoint: {e}")
                return ShardUnloadModelResponse(
                    success=False,
                    message=f"Error: {str(e)}",
                )

        @self.app.post("/cleanup_repacked")
        async def cleanup_repacked(request: Request) -> JSONResponse:  # type: ignore
            """Delete repacked per-layer weights on this shard to free disk.

            Body JSON (all fields optional):
              - model_id: restrict cleanup to this model bucket
              - all: when true, remove the entire repack directory base
            """
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            model_id = (payload or {}).get("model_id")
            all_flag = bool((payload or {}).get("all", False))

            try:
                removed = delete_repacked_layers(
                    model_id=model_id,
                    all_flag=all_flag,
                    base_dir=os.getenv("DNET_REPACK_DIR", "repacked_models"),
                    current_model_path=self.model_path,
                )
                return JSONResponse(content={"removed": list(removed)})
            except Exception as e:
                logger.error("/cleanup_repacked failed: %s", e)
                return JSONResponse(status_code=500, content={"error": str(e)})