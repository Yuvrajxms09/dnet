from typing import Any

model_catalog: dict[str, list[dict[str, Any]]] = {
    "models": [
        {
            "id": "mlx-community/gpt-oss-20b-MXFP4-Q8",
            "arch": "gpt_oss",
            "quantization": "8bit",
            "alias": "gpt-oss-20b",
        },
        {
            "id": "mlx-community/gpt-oss-20b-MXFP4-Q4",
            "arch": "gpt_oss",
            "quantization": "4bit",
            "alias": "gpt-oss-20b",
        },
        {
            "id": "mlx-community/gpt-oss-120b-MXFP4-Q8",
            "arch": "gpt_oss",
            "quantization": "8bit",
            "alias": "gpt-oss-120b",
        },
        {
            "id": "mlx-community/gpt-oss-120b-MXFP4-Q4",
            "arch": "gpt_oss",
            "quantization": "4bit",
            "alias": "gpt-oss-120b",
        },
        {
            "id": "Qwen/Qwen3-4B-MLX-bf16",
            "arch": "qwen3",
            "quantization": "bf16",
            "alias": "qwen3-4b",
        },
        {
            "id": "Qwen/Qwen3-4B-MLX-8bit",
            "arch": "qwen3",
            "quantization": "8bit",
            "alias": "qwen3-4b",
        },
        {
            "id": "Qwen/Qwen3-4B-MLX-4bit",
            "arch": "qwen3",
            "quantization": "4bit",
            "alias": "qwen3-4b",
            "ci_test": True,
        },
        {
            "id": "Qwen/Qwen3-8B-MLX-bf16",
            "arch": "qwen3",
            "quantization": "bf16",
            "alias": "qwen3-8b",
        },
        {
            "id": "Qwen/Qwen3-8B-MLX-8bit",
            "arch": "qwen3",
            "quantization": "8bit",
            "alias": "qwen3-8b",
        },
        {
            "id": "Qwen/Qwen3-8B-MLX-4bit",
            "arch": "qwen3",
            "quantization": "4bit",
            "alias": "qwen3-8b",
        },
        {
            "id": "Qwen/Qwen3-14B-MLX-bf16",
            "arch": "qwen3",
            "quantization": "bf16",
            "alias": "qwen3-14b",
        },
        {
            "id": "Qwen/Qwen3-14B-MLX-8bit",
            "arch": "qwen3",
            "quantization": "8bit",
            "alias": "qwen3-14b",
        },
        {
            "id": "Qwen/Qwen3-14B-MLX-4bit",
            "arch": "qwen3",
            "quantization": "4bit",
            "alias": "qwen3-14b",
        },
        {
            "id": "Qwen/Qwen3-32B-MLX-bf16",
            "arch": "qwen3",
            "quantization": "bf16",
            "alias": "qwen3-32b",
        },
        {
            "id": "Qwen/Qwen3-32B-MLX-8bit",
            "arch": "qwen3",
            "quantization": "8bit",
            "alias": "qwen3-32b",
        },
        {
            "id": "Qwen/Qwen3-32B-MLX-4bit",
            "arch": "qwen3",
            "quantization": "4bit",
            "alias": "qwen3-32b",
        },
        {
            "id": "mlx-community/Llama-3.2-3B-Instruct",
            "arch": "llama",
            "quantization": "fp16",
            "alias": "llama-3.2-3b-instruct",
        },
        {
            "id": "mlx-community/Llama-3.2-3B-Instruct-8bit",
            "arch": "llama",
            "quantization": "8bit",
            "alias": "llama-3.2-3b-instruct",
        },
        {
            "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "llama-3.2-3b-instruct",
            "ci_test": True,
        },
        {
            "id": "mlx-community/Llama-3.1-8B-Instruct",
            "arch": "llama",
            "quantization": "fp16",
            "alias": "llama-3.1-8b-instruct",
        },
        {
            "id": "mlx-community/Llama-3.1-8B-Instruct-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "llama-3.1-8b-instruct",
        },
        {
            "id": "mlx-community/llama-3.3-70b-instruct-fp16",
            "arch": "llama",
            "quantization": "fp16",
            "alias": "llama-3.3-70b-instruct",
        },
        {
            "id": "mlx-community/Llama-3.3-70B-Instruct-8bit",
            "arch": "llama",
            "quantization": "8bit",
            "alias": "llama-3.3-70b-instruct",
        },
        {
            "id": "mlx-community/Llama-3.3-70B-Instruct-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "llama-3.3-70b-instruct",
        },
        {
            "id": "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "llama-3.1-70b-instruct",
        },
        {
            "id": "mlx-community/Hermes-4-70B-8bit",
            "arch": "llama",
            "quantization": "8bit",
            "alias": "hermes-4-70b",
        },
        {
            "id": "mlx-community/Hermes-4-70B-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "hermes-4-70b",
        },
        {
            "id": "mlx-community/Hermes-4-405B-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "hermes-4-405b",
        },
        {
            "id": "mlx-community/DeepSeek-V3-4bit",
            "arch": "deepseek_v3",
            "quantization": "4bit",
            "alias": "deepseek-v3",
        },
# DeepSeek V3.2 models disabled - requires mlx-lm>=0.30.0
# {
#     "id": "mlx-community/DeepSeek-V3.2_bf16",
#     "arch": "deepseek_v32",
#     "quantization": "bf16",
#     "alias": "deepseek-v3.2",
# },
# {
#     "id": "mlx-community/DeepSeek-V3.2-8bit",
#     "arch": "deepseek_v32",
#     "quantization": "8bit",
#     "alias": "deepseek-v3.2",
# },
# {
#     "id": "mlx-community/DeepSeek-V3.2-4bit",
#     "arch": "deepseek_v32",
#     "quantization": "4bit",
#     "alias": "deepseek-v3.2",
# },
        {
            "id": "mlx-community/OLMo-2-0325-32B-Instruct-4bit",
            "arch": "olmo2",
            "quantization": "4bit",
            "alias": "olmo-2-32b-instruct",
        },
        {
            "id": "mlx-community/OLMo-2-1124-13B-Instruct-8bit",
            "arch": "olmo2",
            "quantization": "8bit",
            "alias": "olmo-2-13b-instruct",
        },
        {
            "id": "mlx-community/OLMo-2-1124-13B-Instruct-4bit",
            "arch": "olmo2",
            "quantization": "4bit",
            "alias": "olmo-2-13b-instruct",
        },
        {
            "id": "mlx-community/OLMo-2-1124-7B-Instruct-8bit",
            "arch": "olmo2",
            "quantization": "8bit",
            "alias": "olmo-2-7b-instruct",
        },
        {
            "id": "mlx-community/OLMo-2-1124-7B-Instruct-4bit",
            "arch": "olmo2",
            "quantization": "4bit",
            "alias": "olmo-2-7b-instruct",
            "ci_test": True,
        },
        {
            "id": "mlx-community/GLM-4.7-8bit",
            "arch": "glm47",
            "quantization": "8bit",
            "alias": "glm-4.7",
        },
        {
            "id": "mlx-community/GLM-4.7-4bit",
            "arch": "glm47",
            "quantization": "4bit",
            "alias": "glm-4.7",
        },
        {
            "id": "mlx-community/MiniMax-M2.1-8bit",
            "arch": "minimax_21",
            "quantization": "8bit",
            "alias": "minimax-2.1",
        },
        {
            "id": "mlx-community/MiniMax-M2.1-4bit",
            "arch": "minimax_21",
            "quantization": "4bit",
            "alias": "minimax-2.1",
        },
    ]
}


def get_ci_test_models() -> list[dict[str, Any]]:
    """Return models marked for CI integration testing.

    These models are small enough to run on a single CI machine.
    """
    return [m for m in model_catalog["models"] if m.get("ci_test", False)]
