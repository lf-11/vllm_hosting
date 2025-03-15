#!/bin/bash
# This script runs the vLLM server and keeps the tmux session alive

echo 'Starting vLLM server with command: /home/lukas/miniconda3/bin/python3 /home/lukas/projects/LLM_server/engines/vllm/server.py start --model Mistral-Small --max-model-len 4096'
/home/lukas/miniconda3/bin/python3 /home/lukas/projects/LLM_server/engines/vllm/server.py start --model Mistral-Small --max-model-len 4096
EXIT_CODE=$?
echo ''
echo '================================================='
echo "Server process exited with code $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo 'ERROR: Server process failed!'
    echo 'Check the output above for error details.'
else
    echo 'Server process completed successfully.'
fi
echo '================================================='
echo 'Press Enter to close this session...'
read
