from pathlib import Path
from typing import Dict, Any, Optional

# Base paths
MODELS_DIR = Path("/home/lukas/projects/LLM_testing/webui/text-generation-webui-main/models")
TEMPLATES_DIR = Path("/home/lukas/projects/LLM_Canvas/chat_templates")

MODEL_CONFIGS = {
    "mistral-small": {
        "model_path": str(MODELS_DIR / "Mistral-Small-Instruct-2409-Q6_K_L.gguf"),
        "chat_template": str(TEMPLATES_DIR / "mistral_chat.jinja"),
        "max_model_len": 17000,
        "tensor_parallel_size": 2,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 24576,
        "default_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
        },
        "stop_tokens": ["</s>", "<|im_end|>"]
    },
    "llama3-70b": {
        "model_path": str(MODELS_DIR / "Llama-3.3-70B-Instruct-Q4_K_S.gguf"),
        "chat_template": str(TEMPLATES_DIR / "llama3_chat.jinja"),
        "max_model_len": 4096,
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.97,
        "enforce_eager": True,
        "default_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
        },
        "stop_tokens": ["</s>", "<|im_end|>", "<|endoftext|>"]
    },
    "qwen-14b": {
        "model_path": str(MODELS_DIR / "Qwen2.5-14B-Instruct-Q4_K_L.gguf"),
        "max_model_len": 4096,
        "tensor_parallel_size": 2,
        "default_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
        },
        "stop_tokens": ["<|endoftext|>", "<|im_end|>"]
    }
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]

def get_server_args(model_name: str) -> list[str]:
    """Convert model config to vLLM server arguments"""
    config = get_model_config(model_name)
    args = ["serve", config["model_path"]]
    
    # Add standard args
    if "tensor_parallel_size" in config:
        args.extend(["--tensor_parallel_size", str(config["tensor_parallel_size"])])
    if "max_model_len" in config:
        args.extend(["--max_model_len", str(config["max_model_len"])])
    if "chat_template" in config:
        args.extend(["--chat-template", config["chat_template"]])
    if "gpu_memory_utilization" in config:
        args.extend(["--gpu-memory-utilization", str(config["gpu_memory_utilization"])])
    if config.get("enforce_eager"):
        args.append("--enforce_eager")
        
    return args 