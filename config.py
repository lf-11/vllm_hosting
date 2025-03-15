"""
Configuration settings for the LLM Server Manager
"""

# Database configuration
DB_CONFIG = {
    'dbname': 'lm_hosting',  # Correct database name
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost'
}

# Server configuration
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 8080,
    'log_dir': 'logs'
}

# Path configuration
PATHS = {
    'models_dir': '/home/lukas/projects/LLM_testing/webui/text-generation-webui-main/models',  # Adjust this to your actual models directory
    'frontend_dir': 'frontend',
    'engines_dir': 'engines'
}

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    'tensor_parallel_size': 2,
    'gpu_memory_utilization': 0.97,
    'enforce_eager': False
} 