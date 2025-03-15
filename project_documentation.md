# LLM Server Manager

## Project Overview

The LLM Server Manager is a comprehensive solution for hosting, managing, and interacting with large language models (LLMs) on a local Linux server. The project provides both a web-based interface and command-line tools to control LLM servers, monitor their performance, and make inference requests.

### Goals

1. **Simplified LLM Management**: Provide an easy way to start, stop, and monitor LLM servers without needing to remember complex commands.

2. **Remote Access**: Allow controlling the LLM server from any device on the network, including remote access from a MacBook to a Linux server.

3. **Persistent Operation**: Ensure the LLM server continues running even when SSH connections are closed or terminals are exited.

4. **Multiple Engine Support**: Support different LLM backends (currently vLLM, with plans to add llama.cpp).

5. **Performance Tracking**: Log and visualize model performance metrics for all generations.

### Current Functionality

- Web interface for controlling LLM servers
- Real-time terminal output streaming
- Database integration for model configuration and performance tracking
- Command-line utilities for server management from any directory
- Persistent server operation using tmux sessions
- Support for vLLM engine with configurable parameters

## Project Structure

```
/home/lukas/projects/LLM_server/
│
├── api_server.py                # FastAPI server for web interface
├── llm_manager.py               # CLI manager for starting/stopping services
├── install_cli.sh               # Installation script for CLI commands
├── init_db.py                   # Database initialization script
├── client.py                    # Client script for making requests to the LLM server
│
├── frontend/                    # Frontend web interface
│   ├── index.html               # Main HTML page
│   ├── js/
│   │   └── app.js               # JavaScript for the frontend
│   └── css/                     # (Optional) Custom CSS styles
│
├── engines/                     # Different LLM engine implementations
│   ├── vllm/                    # vLLM specific code
│   │   └── server.py            # vLLM server implementation (moved from root)
│   └── llama_cpp/               # Future llama.cpp implementation
│       └── server.py            # (Planned) llama.cpp server implementation
│
├── utils/                       # Shared utility functions
│   ├── db_utils.py              # Database utilities
│   └── logging_utils.py         # Logging utilities
│
└── logs/                        # Log files
    ├── vllm/                    # vLLM server logs
    └── api/                     # API server logs
```

## File Descriptions

### Core Files

- **api_server.py**: FastAPI server that provides API endpoints for the web interface and handles WebSocket connections for real-time terminal output.

- **llm_manager.py**: Command-line utility for managing the LLM server. Handles starting, stopping, and checking the status of the server in a tmux session.

- **install_cli.sh**: Installation script that sets up system-wide commands (`llm-start`, `llm-stop`, `llm-status`, `llm-logs`) for easy server management.

- **init_db.py**: Script to initialize the PostgreSQL database with the necessary tables and functions for tracking model configurations and performance metrics.

- **client.py**: Python script for making requests to the LLM server.

### Frontend

- **frontend/index.html**: Main HTML page for the web interface, providing controls for server management and a terminal output display.

- **frontend/js/app.js**: JavaScript code that handles the frontend logic, including API calls, WebSocket communication, and UI updates.

### Engines

- **engines/vllm/server.py**: Implementation of the vLLM server, handling model loading, inference requests, and performance tracking.

- **engines/llama_cpp/**: Directory for the future llama.cpp implementation.

### Utilities

- **utils/**: Directory containing shared utility functions used across different components of the project.

## Command-Line Interface

The project provides several command-line utilities that can be run from any directory:

- **llm-start**: Start the LLM Manager server in a detached tmux session.
- **llm-stop**: Stop the LLM Manager server.
- **llm-status**: Check if the LLM Manager server is running and display access URLs.
- **llm-logs**: View the server logs by attaching to the tmux session.

## Web Interface

The web interface is accessible at:
- Local: http://localhost:8080/ui/
- Network: http://192.168.178.61:8080/ui/

The interface provides:
- Server status monitoring
- Model selection and configuration
- Start/stop/restart controls
- Real-time terminal output display
- Model information and performance metrics

## Database Structure

The project uses a PostgreSQL database named 'vllm_hosting' with two main tables:

1. **models**: Stores model configurations, including paths, parameters, and performance statistics.
2. **generations**: Tracks all generation requests with performance metrics.

## Future Plans

1. Add support for llama.cpp as an alternative LLM engine
2. Enhance the web interface with more detailed analytics
3. Add user authentication for secure remote access
4. Implement model fine-tuning capabilities
5. Add support for model quantization options 