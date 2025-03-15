#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Define paths
PROJECT_DIR="/home/lukas/projects/LLM_server"
LLM_MANAGER_SCRIPT="$PROJECT_DIR/llm_manager.py"
BIN_DIR="/usr/local/bin"

echo "Installing LLM Manager CLI tools..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root (use sudo)${NC}"
  exit 1
fi

# Remove old scripts if they exist
rm -f "$BIN_DIR/vllm-start" "$BIN_DIR/vllm-stop" "$BIN_DIR/vllm-status"

# Create the llm-start script
cat > "$BIN_DIR/llm-start" << 'EOF'
#!/bin/bash
python3 /home/lukas/projects/LLM_server/llm_manager.py start
EOF

# Create the llm-stop script
cat > "$BIN_DIR/llm-stop" << 'EOF'
#!/bin/bash
python3 /home/lukas/projects/LLM_server/llm_manager.py stop
EOF

# Create the llm-status script
cat > "$BIN_DIR/llm-status" << 'EOF'
#!/bin/bash
python3 /home/lukas/projects/LLM_server/llm_manager.py status
EOF

# Create the llm-restart script
cat > "$BIN_DIR/llm-restart" << 'EOF'
#!/bin/bash
python3 /home/lukas/projects/LLM_server/llm_manager.py restart
EOF

# Create the llm-logs script
cat > "$BIN_DIR/llm-logs" << 'EOF'
#!/bin/bash
tmux attach -t llm-manager
EOF

# Make the scripts executable
chmod +x "$BIN_DIR/llm-start"
chmod +x "$BIN_DIR/llm-stop"
chmod +x "$BIN_DIR/llm-status"
chmod +x "$BIN_DIR/llm-restart"
chmod +x "$BIN_DIR/llm-logs"
chmod +x "$LLM_MANAGER_SCRIPT"

echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "You can now use the following commands from anywhere:"
echo "  llm-start   - Start the LLM Manager server"
echo "  llm-stop    - Stop the LLM Manager server"
echo "  llm-restart - Restart the LLM Manager server"
echo "  llm-status  - Check if the LLM Manager server is running"
echo "  llm-logs    - View the server logs (Ctrl+B, then D to detach)"
echo ""
echo "The server will continue running even if you close your terminal or SSH session."