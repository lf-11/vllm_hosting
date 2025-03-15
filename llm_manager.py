#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import socket
from pathlib import Path
from datetime import datetime
import time

# Path to the project directory
PROJECT_DIR = Path("/home/lukas/projects/LLM_server")
API_SCRIPT = PROJECT_DIR / "api_server.py"

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def start_server():
    """Start the vLLM manager API server in a detached tmux session"""
    # Check if tmux is installed
    if subprocess.call(["which", "tmux"], stdout=subprocess.DEVNULL) != 0:
        print("Error: tmux is not installed. Please install it with 'sudo apt install tmux'")
        return False
    
    # Check if the session already exists
    result = subprocess.run(
        ["tmux", "has-session", "-t", "llm-manager"], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    
    if result.returncode == 0:
        print("LLM Manager is already running!")
        show_server_info()
        return True
    
    # Create a new detached tmux session with proper error logging
    log_file = f"{PROJECT_DIR}/logs/llm_manager_startup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs(f"{PROJECT_DIR}/logs", exist_ok=True)
    
    # Create a script to run the API server with proper error handling
    startup_script = f"""
#!/bin/bash
cd {PROJECT_DIR}
echo "Starting API server at $(date)" > {log_file}
python3 {API_SCRIPT} 2>&1 | tee -a {log_file}
echo "API server exited with code $? at $(date)" >> {log_file}
echo "Press Enter to close this session..."
read
"""
    
    script_path = f"{PROJECT_DIR}/start_api_server.sh"
    with open(script_path, "w") as f:
        f.write(startup_script)
    
    os.chmod(script_path, 0o755)
    
    # Start the tmux session with the script
    subprocess.run([
        "tmux", "new-session", "-d", "-s", "llm-manager", script_path
    ])
    
    print("LLM Manager started successfully!")
    show_server_info()
    return True

def stop_server():
    """Stop the vLLM manager API server tmux session"""
    # Check if the session exists
    result = subprocess.run(
        ["tmux", "has-session", "-t", "llm-manager"], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    
    if result.returncode != 0:
        print("LLM Manager is not running.")
        return True
    
    # Kill the tmux session
    subprocess.run(["tmux", "kill-session", "-t", "llm-manager"])
    print("LLM Manager stopped successfully!")
    return True

def restart_server():
    """Restart the vLLM manager API server"""
    print("Restarting LLM Manager...")
    
    # First stop the server
    if stop_server():
        # Wait a moment to ensure everything is properly shut down
        time.sleep(2)
        
        # Then start it again
        if start_server():
            print("LLM Manager restarted successfully!")
            return True
    
    print("Failed to restart LLM Manager.")
    return False

def show_server_info():
    """Display server access information"""
    local_ip = get_local_ip()
    port = 8080
    
    print(f"\n{'='*50}")
    print(f"LLM Server Manager is running at:")
    print(f"- Local:   http://localhost:{port}/ui/")
    print(f"- Network: http://{local_ip}:{port}/ui/")
    print(f"{'='*50}")
    print("\nTo view logs: tmux attach -t llm-manager")
    print("To detach from logs: Press Ctrl+B, then D")
    print("Or simply run: llm-logs")
    print("To stop the server: llm-stop")

def main():
    parser = argparse.ArgumentParser(description="vLLM Server Manager CLI")
    parser.add_argument('action', choices=['start', 'stop', 'status', 'restart'], 
                        help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'start':
        start_server()
    elif args.action == 'stop':
        stop_server()
    elif args.action == 'restart':
        restart_server()
    elif args.action == 'status':
        # Check if the session exists
        result = subprocess.run(
            ["tmux", "has-session", "-t", "llm-manager"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        
        if result.returncode == 0:
            print("LLM Manager is running.")
            show_server_info()
        else:
            print("LLM Manager is not running.")

if __name__ == "__main__":
    main() 