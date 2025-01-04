# Local LLM Hosting Guide

## Setup Requirements
- PostgreSQL database named 'vllm_hosting'
- vLLM framework installed
- Python packages: psycopg2, requests, tenacity

## Database Tables
1. `models` table: Stores model configurations
2. `generations` table: Tracks all generations with performance metrics

## Starting the Server
```bash
python server.py start --model MODEL_NAME
```

Optional parameters:
- `--max-model-len INT`: Set maximum sequence length
- `--speculative-decoding`: Enable speculative decoding

## Using the Client

### Basic Usage
```python
from client import VLLMClient
client = VLLMClient()

# Single generation
response = client.generate("Your prompt here", model="model_name")
print(response[0])

# Batch generation
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
responses = client.generate(prompts, model="model_name")  # batch_size is optional
```

### Parameters
- `model`: Name of the model to use
- `mode`: "completion" (default) or "chat"
- `batch_size`: (Optional) Maximum prompts to process simultaneously. If not specified, uses model's configured max_batch_size
- `params`: Dictionary of generation parameters:
  ```python
  params = {
      "temperature": 0.7,
      "max_tokens": 1024,
      "top_p": 0.95
  }
  ```

### Example with Parameters
```python
response = client.generate(
    "Your prompt",
    model="model_name",
    mode="chat",
    params={"temperature": 0.8, "max_tokens": 2048}
)
```

## Performance Tracking
- All generations are automatically logged in the database
- Tracks:
  - Input/output text
  - Tokens generated
  - Time taken
  - Model performance metrics

## Server Management
```bash
# Stop server
python3 server.py stop

# Restart server
python3 server.py restart --model MODEL_NAME

# Force kill (if needed)
python3 server.py force-kill