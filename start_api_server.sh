
#!/bin/bash
cd /home/lukas/projects/LLM_server
echo "Starting API server at $(date)" > /home/lukas/projects/LLM_server/logs/llm_manager_startup_20250315_113004.log
python3 /home/lukas/projects/LLM_server/api_server.py 2>&1 | tee -a /home/lukas/projects/LLM_server/logs/llm_manager_startup_20250315_113004.log
echo "API server exited with code $? at $(date)" >> /home/lukas/projects/LLM_server/logs/llm_manager_startup_20250315_113004.log
echo "Press Enter to close this session..."
read
