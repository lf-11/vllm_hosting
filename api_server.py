import os
import sys
import json
import asyncio
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
import psycopg2
from psycopg2.extras import DictCursor
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import socket
import time
from datetime import datetime
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/api_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger("vllm-api")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Models for request validation
class ServerStartRequest(BaseModel):
    model: str
    max_model_len: Optional[int] = None
    speculative_decoding: bool = False

class ModelConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    id: int
    model_name: str
    model_path: str
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: Optional[float]
    enforce_eager: bool
    chat_template: Optional[str]
    speculative_decoding: bool
    max_batch_size: Optional[int]
    generations_count: int
    tokens_generated_count: int
    avg_tokens_per_second: float

# Initialize FastAPI app
app = FastAPI(title="LLM Server Manager API")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
db_config = {
    'dbname': 'lm_hosting',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost'
}

# Path to server.py
SERVER_SCRIPT_PATH = Path(__file__).parent / "engines/vllm/server.py"

# Global variables
server_process = None
output_queue = asyncio.Queue()
connected_websockets = set()
server_error_detected = False

from utils import db_utils
from config import SERVER_CONFIG

def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(**db_config)

def is_server_running():
    """Check if vLLM server is running by checking the PID file"""
    pid_file = Path("vllm_server.pid")
    if not pid_file.exists():
        return False
    
    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())
        
        # Check if process exists
        os.kill(pid, 0)  # This will raise an exception if process doesn't exist
        return True
    except (ProcessLookupError, ValueError, FileNotFoundError, PermissionError):
        return False

def get_server_status():
    """Get detailed status of the server"""
    status = {
        "running": is_server_running(),
        "current_model": None,
        "start_time": None,
        "uptime_seconds": None
    }
    
    # Check for additional details if server is running
    if status["running"]:
        pid_file = Path("vllm_server.pid")
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            
            # Get process info
            import psutil
            process = psutil.Process(pid)
            status["start_time"] = process.create_time()
            status["uptime_seconds"] = time.time() - process.create_time()
            
            # Try to determine current model
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    # Get most recently used model
                    cur.execute("""
                        SELECT m.model_name 
                        FROM generations g
                        JOIN models m ON g.model_id = m.id
                        ORDER BY g.generated_at DESC
                        LIMIT 1
                    """)
                    result = cur.fetchone()
                    if result:
                        status["current_model"] = result[0]
        except Exception as e:
            logger.error(f"Error getting server details: {e}")
    
    return status

async def read_process_output(process):
    """Read output from the process and put it in the queue"""
    global server_error_detected
    
    # Buffer to collect error messages
    error_buffer = ""
    error_detected = False
    
    while True:
        if process.stdout.at_eof():
            break
        line = await process.stdout.readline()
        if not line:
            break
        line_str = line.decode('utf-8')
        await output_queue.put(line_str)
        
        # Check for error patterns
        if "error" in line_str.lower() or "exception" in line_str.lower() or "terminated" in line_str.lower():
            error_buffer += line_str
            error_detected = True
            
            # Check for specific error patterns
            if "RuntimeError: Server process terminated unexpectedly" in line_str:
                server_error_detected = True
                
                # Log the error buffer for debugging
                logger.info(f"Error detected: {error_buffer}")
                
                # Find matching errors in database
                matching_errors = db_utils.find_matching_error(error_buffer)
                
                # Log the matching result
                logger.info(f"Matching errors found: {len(matching_errors)}")
                
                if matching_errors:
                    # Known error found
                    for error in matching_errors:
                        logger.info(f"Sending known error: {error['error_message']}")
                        # Send error information to client
                        error_info = {
                            "type": "error",
                            "data": {
                                "known": True,  # This is a known error
                                "message": error["error_message"],
                                "solution": error["solution"]
                            }
                        }
                        # Forward to all connected websockets
                        for websocket in connected_websockets:
                            try:
                                await websocket.send_text(json.dumps(error_info))
                            except Exception:
                                pass
                else:
                    # New error - add to database with placeholder solution
                    # Extract a concise error message
                    import re
                    error_message = "Unknown error"
                    
                    # Try to extract the most specific error message
                    error_patterns = [
                        r"RuntimeError: (.+?)(?:\n|$)",
                        r"Error: (.+?)(?:\n|$)",
                        r"Exception: (.+?)(?:\n|$)"
                    ]
                    
                    for pattern in error_patterns:
                        match = re.search(pattern, error_buffer)
                        if match:
                            error_message = match.group(1).strip()
                            break
                    
                    # Create a pattern from the error message
                    error_pattern = re.escape(error_message[:100])  # Use first 100 chars as pattern
                    
                    # Add to database
                    try:
                        error_id = db_utils.add_new_error(
                            error_pattern,
                            error_message,
                            "This is a newly detected error. No solution has been provided yet."
                        )
                        
                        # Send new error information to client
                        error_info = {
                            "type": "error",
                            "data": {
                                "known": False,  # This is a new error
                                "message": error_message,
                                "error_id": error_id
                            }
                        }
                        # Forward to all connected websockets
                        for websocket in connected_websockets:
                            try:
                                await websocket.send_text(json.dumps(error_info))
                            except Exception:
                                pass
                    except Exception as e:
                        logger.error(f"Failed to add new error to database: {e}")
                
                # Update server status to failed
                status_update = {
                    "type": "status",
                    "data": {
                        "running": False,
                        "error": True,
                        "message": "Server failed to start"
                    }
                }
                
                # Forward to all connected websockets
                for websocket in connected_websockets:
                    try:
                        await websocket.send_text(json.dumps(status_update))
                    except Exception:
                        pass
        
        # Forward to all connected websockets
        for websocket in connected_websockets:
            try:
                await websocket.send_text(line_str)
            except Exception:
                # Will be removed in the connection handler
                pass
    
    # If we detected errors but didn't handle them specifically, check for matches at the end
    if error_detected and not server_error_detected and error_buffer:
        matching_errors = db_utils.find_matching_error(error_buffer)
        
        if matching_errors:
            # Known error found
            for error in matching_errors:
                # Send error information to client
                error_info = {
                    "type": "error",
                    "data": {
                        "known": True,
                        "message": error["error_message"],
                        "solution": error["solution"]
                    }
                }
                # Forward to all connected websockets
                for websocket in connected_websockets:
                    try:
                        await websocket.send_text(json.dumps(error_info))
                    except Exception:
                        pass

async def start_server_process(model, max_model_len=None, speculative_decoding=False):
    """Start the vLLM server process in the background"""
    global server_process
    
    await output_queue.put("Starting server process...")
    
    # Get model configuration from database
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # Find the model configuration that matches the requested parameters
            query = """
                SELECT * FROM models 
                WHERE model_name = %s
            """
            params = [model]
            
            if max_model_len:
                query += " AND max_model_len = %s"
                params.append(max_model_len)
            
            cur.execute(query, params)
            model_config = cur.fetchone()
            
            if not model_config:
                await output_queue.put(f"ERROR: No configuration found for model {model}")
                return
    
    # Create a unique session name
    tmux_session = f"vllm-server-{int(time.time())}"
    
    # Create a script to run the server with proper error handling
    script_content = f"""#!/bin/bash
cd {Path(__file__).parent}
echo "Starting vLLM server at $(date)"
echo "Model: {model}"
echo "Max model length: {max_model_len or 'default'}"
echo "Speculative decoding: {'enabled' if speculative_decoding else 'disabled'}"
echo "----------------------------------------"

# Run the server
python {SERVER_SCRIPT_PATH} start \\
    --model "{model}" \\
    {f'--max-model-len {max_model_len}' if max_model_len else ''} \\
    {'--speculative-decoding' if speculative_decoding else ''}

# Capture exit code
EXIT_CODE=$?

echo "----------------------------------------"
echo "vLLM server exited with code $EXIT_CODE at $(date)"
echo "Press Enter to close this session..."
read
"""
    
    # Write the script to a temporary file
    script_path = Path(f"/tmp/vllm_server_{tmux_session}.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    # Start the tmux session with the script
    await output_queue.put(f"Creating tmux session: {tmux_session}")
    
    process = await asyncio.create_subprocess_exec(
        "tmux", "new-session", "-d", "-s", tmux_session, str(script_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        await output_queue.put(f"ERROR: Failed to create tmux session. Return code: {process.returncode}")
        await output_queue.put(f"ERROR: stdout: {stdout.decode() if stdout else 'None'}")
        await output_queue.put(f"ERROR: stderr: {stderr.decode() if stderr else 'None'}")
        return
    
    await output_queue.put(f"Tmux session created successfully")
    
    # Write the tmux session name to a file for reference
    with open("vllm_tmux_session.txt", "w") as f:
        f.write(tmux_session)
    
    # Add tmux connection info to the output
    await output_queue.put(f"Server starting with model: {model}")
    await output_queue.put(f"To connect to the server session directly, run:")
    await output_queue.put(f"  tmux attach -t {tmux_session}")
    
    # Create a dummy process to keep track of
    server_process = await asyncio.create_subprocess_exec(
        "echo", "Server started in tmux session",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )
    
    # Start a task to monitor the tmux session
    asyncio.create_task(monitor_tmux_session(tmux_session))

async def monitor_tmux_session(tmux_session):
    """Monitor the tmux session and capture its output"""
    await output_queue.put(f"Starting to monitor tmux session: {tmux_session}")
    
    # Keep track of the last content
    last_content = ""
    
    # Wait a moment for the session to initialize
    await asyncio.sleep(2)
    
    # Flag to track if we've detected the server has exited
    server_exited = False
    
    # Buffer for collecting output
    output_buffer = ""
    
    # Flag to track if we're in the middle of a line
    in_partial_line = False
    
    while True:
        # Check if session exists
        process = await asyncio.create_subprocess_exec(
            "tmux", "has-session", "-t", tmux_session,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.communicate()
        
        if process.returncode != 0:
            # Flush any remaining buffer before exiting
            if output_buffer:
                await output_queue.put(output_buffer)
            await output_queue.put(f"Tmux session {tmux_session} has ended or doesn't exist.")
            break
        
        # Capture output from the session
        process = await asyncio.create_subprocess_exec(
            "tmux", "capture-pane", "-pt", tmux_session, "-S", "-1000",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        stdout, _ = await process.communicate()
        
        if process.returncode == 0 and stdout:
            current_content = stdout.decode('utf-8', errors='replace')
            
            # If content hasn't changed, skip processing
            if current_content == last_content:
                await asyncio.sleep(0.5)
                continue
            
            # Find new content
            if last_content:
                # Find the common prefix length
                i = 0
                min_len = min(len(last_content), len(current_content))
                while i < min_len and last_content[i] == current_content[i]:
                    i += 1
                
                # Get the new content
                new_content = current_content[i:]
                
                if new_content:
                    # Add to our buffer
                    output_buffer += new_content
                    
                    # Process the buffer line by line
                    lines = output_buffer.split('\n')
                    
                    # Process all complete lines (all except the last one)
                    for line in lines[:-1]:
                        if line.strip():
                            # Check for progress bars
                            if '%|' in line and ('█' in line or '▏' in line):
                                # This is a progress bar
                                await output_queue.put({"type": "progress", "data": line})
                            else:
                                # Regular line - if we were in a partial line, prepend with continuation marker
                                if in_partial_line:
                                    await output_queue.put(line)
                                    in_partial_line = False
                                else:
                                    await output_queue.put(line)
                            
                            # Check if the server has exited
                            if "vLLM server exited with code" in line and not server_exited:
                                server_exited = True
                                await output_queue.put("Server process has exited, but tmux session is still available for logs.")
                                await output_queue.put("You can reconnect to the session with: tmux attach -t " + tmux_session)
                    
                    # Keep the last (potentially incomplete) line in the buffer
                    output_buffer = lines[-1]
                    
                    # If the buffer is getting too large without a newline, 
                    # we need to send it as a partial line
                    if len(output_buffer) > 80:  # Typical terminal width
                        await output_queue.put(output_buffer)
                        output_buffer = ""
                        in_partial_line = True
            else:
                # First capture - process all lines
                lines = current_content.splitlines()
                for line in lines:
                    if line.strip():
                        await output_queue.put(line)
            
            # Update last content
            last_content = current_content
        
        # Wait before checking again
        await asyncio.sleep(0.5)

@app.get("/")
async def root():
    """Root endpoint that redirects to the frontend"""
    return {"message": "LLM Server Manager API"}

@app.get("/ui/")
async def ui_root():
    return FileResponse("frontend/index.html")

@app.get("/api/status")
async def get_status():
    """Get server status"""
    return get_server_status()

@app.get("/api/models")
async def get_models():
    """Get all models from the database"""
    try:
        models = db_utils.get_all_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/server/start")
async def start_server(request: ServerStartRequest, background_tasks: BackgroundTasks):
    """Start the vLLM server with the specified model"""
    if is_server_running():
        return JSONResponse(
            status_code=400,
            content={"detail": "Server is already running. Stop it first."}
        )
    
    try:
        # Start server in background
        background_tasks.add_task(
            start_server_process, 
            request.model, 
            request.max_model_len, 
            request.speculative_decoding
        )
        
        return {"status": "starting", "model": request.model}
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")

@app.post("/api/server/stop")
async def stop_server():
    """Stop the vLLM server"""
    global server_process
    
    if not is_server_running():
        return {"status": "not_running"}
    
    try:
        # Run the stop command
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(SERVER_SCRIPT_PATH), "stop",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        stdout, _ = await process.communicate()
        
        if server_process:
            try:
                server_process.terminate()
                await asyncio.wait_for(server_process.wait(), timeout=10)
            except asyncio.TimeoutError:
                server_process.kill()
            server_process = None
        
        return {"status": "stopped", "output": stdout.decode('utf-8')}
    except Exception as e:
        logger.error(f"Error stopping server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop server: {str(e)}")

@app.post("/api/server/restart")
async def restart_server(request: ServerStartRequest, background_tasks: BackgroundTasks):
    """Restart the vLLM server with the specified model"""
    try:
        # Stop server if running
        if is_server_running():
            await stop_server()
        
        # Start server with new parameters
        return await start_server(request, background_tasks)
    except Exception as e:
        logger.error(f"Error restarting server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restart server: {str(e)}")

@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    try:
        await websocket.accept()
        connected_websockets.add(websocket)
        
        logger.info("New WebSocket connection established")
        
        # Send initial status
        try:
            status = get_server_status()
            await websocket.send_text(json.dumps({
                "type": "status",
                "data": status
            }))
            
            # Send a welcome message to the terminal
            await output_queue.put("WebSocket connection established. Terminal ready.")
            
            if not is_server_running():
                await output_queue.put("No LLM server is currently running. Use the form on the left to start one.")
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
        
        # Keep the connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket message loop: {e}")
                break
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        try:
            connected_websockets.remove(websocket)
            logger.info("WebSocket connection removed from active connections")
        except KeyError:
            pass

@app.get("/api/logs/recent")
async def get_recent_logs(lines: int = 100):
    """Get recent log lines from the server log file"""
    log_dir = Path("logs/vllm")
    if not log_dir.exists():
        return {"logs": []}
    
    # Find most recent log file
    log_files = sorted(log_dir.glob("vllm_*.log"), reverse=True)
    if not log_files:
        return {"logs": []}
    
    recent_log = log_files[0]
    
    # Read last N lines
    try:
        with open(recent_log, 'r') as f:
            all_lines = f.readlines()
            return {"logs": all_lines[-lines:]}
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return {"logs": [], "error": str(e)}

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # This creates a socket that doesn't actually connect anywhere
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # This connects to an external IP (doesn't actually send anything)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"  # Fallback to localhost

@app.get("/api/gpu/stats")
async def get_gpu_stats():
    """Get GPU statistics using nvidia-smi"""
    try:
        # Run nvidia-smi to get GPU information in JSON format
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        stdout, _ = await process.communicate()
        
        if process.returncode != 0:
            return {"error": "Failed to run nvidia-smi", "gpus": []}
        
        # Parse the CSV output
        gpu_data = []
        for line in stdout.decode('utf-8').strip().split('\n'):
            if not line.strip():
                continue
                
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 8:
                gpu_data.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "temperature": float(parts[2]),
                    "utilization": float(parts[3]),
                    "memory": {
                        "used": float(parts[4]),
                        "total": float(parts[5])
                    },
                    "power": {
                        "draw": float(parts[6]),
                        "limit": float(parts[7])
                    }
                })
        
        # Get process information
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-compute-apps=gpu_index,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        stdout, _ = await process.communicate()
        
        processes = []
        if process.returncode == 0:
            process_output = stdout.decode('utf-8').strip()
            if process_output:  # Only process if there's output
                for line in process_output.split('\n'):
                    if not line.strip():
                        continue
                        
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 4:
                        processes.append({
                            "gpu_id": int(parts[0]),
                            "pid": int(parts[1]),
                            "name": parts[2],
                            "memory_usage": float(parts[3]),
                            "type": "C"  # Compute process
                        })
        
        return {
            "gpus": gpu_data,
            "processes": processes
        }
        
    except Exception as e:
        logger.error(f"Error getting GPU stats: {e}")
        return {"error": str(e), "gpus": []}

@app.get("/api/errors")
async def get_errors():
    """Get all known errors from the database"""
    try:
        errors = db_utils.get_known_errors()
        return {"errors": errors}
    except Exception as e:
        logger.error(f"Error getting known errors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/errors")
async def add_error(error: dict):
    """Add a new error to the database"""
    try:
        error_id = db_utils.add_new_error(
            error["pattern"],
            error["message"],
            error["solution"]
        )
        return {"id": error_id}
    except Exception as e:
        logger.error(f"Error adding new error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test/error/{type}")
async def test_error(type: str):
    """Test endpoint to simulate error handling"""
    if type == "known":
        # Simulate a known error
        error_info = {
            "type": "error",
            "data": {
                "known": True,
                "message": "Test known error",
                "solution": "This is a test solution for a known error."
            }
        }
        
        # Send to all connected websockets
        for websocket in connected_websockets:
            try:
                await websocket.send_text(json.dumps(error_info))
            except Exception:
                pass
                
        return {"status": "sent", "error_type": "known"}
        
    elif type == "new":
        # Simulate a new error
        error_info = {
            "type": "error",
            "data": {
                "known": False,
                "message": "Test new error",
                "error_id": 9999
            }
        }
        
        # Send to all connected websockets
        for websocket in connected_websockets:
            try:
                await websocket.send_text(json.dumps(error_info))
            except Exception:
                pass
                
        return {"status": "sent", "error_type": "new"}
        
    return {"status": "error", "message": "Invalid error type"}

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests"""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

@app.get("/api/server/tmux-info")
async def get_tmux_info():
    """Get information about tmux sessions"""
    try:
        # Get llm-manager session info
        llm_manager_exists = False
        process = await asyncio.create_subprocess_exec(
            "tmux", "has-session", "-t", "llm-manager",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.communicate()
        llm_manager_exists = process.returncode == 0
        
        # Get vLLM server session info
        vllm_session = None
        try:
            with open("vllm_tmux_session.txt", "r") as f:
                vllm_session = f.read().strip()
                
            # Check if this session exists
            process = await asyncio.create_subprocess_exec(
                "tmux", "has-session", "-t", vllm_session,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.communicate()
            if process.returncode != 0:
                vllm_session = None
        except FileNotFoundError:
            pass
        
        # Get all tmux sessions
        all_sessions = []
        process = await asyncio.create_subprocess_exec(
            "tmux", "list-sessions", "-F", "#{session_name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        stdout, _ = await process.communicate()
        
        if process.returncode == 0:
            all_sessions = stdout.decode('utf-8').strip().split('\n')
            # Filter out empty strings
            all_sessions = [s for s in all_sessions if s]
        
        return {
            "llm_manager": {
                "exists": llm_manager_exists,
                "name": "llm-manager" if llm_manager_exists else None
            },
            "vllm_server": {
                "exists": vllm_session is not None,
                "name": vllm_session
            },
            "all_sessions": all_sessions
        }
    except Exception as e:
        logger.error(f"Error getting tmux info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/server/hostname")
async def get_hostname():
    """Get the server hostname for SSH connections"""
    try:
        # Get hostname
        hostname = socket.gethostname()
        
        # Get username (for SSH command)
        username = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
        
        return {
            "hostname": hostname,
            "username": username,
            "fqdn": socket.getfqdn(),
            "ip": get_local_ip()
        }
    except Exception as e:
        logger.error(f"Error getting hostname: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/server/check-tmux")
async def check_tmux():
    """Check tmux sessions and display in terminal"""
    try:
        # List all tmux sessions
        process = await asyncio.create_subprocess_exec(
            "tmux", "list-sessions",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        stdout, _ = await process.communicate()
        
        if process.returncode == 0:
            sessions = stdout.decode('utf-8').strip()
            if sessions:
                await output_queue.put("Current tmux sessions:")
                for line in sessions.splitlines():
                    await output_queue.put(f"  {line}")
            else:
                await output_queue.put("No tmux sessions found.")
        else:
            await output_queue.put("Error listing tmux sessions.")
        
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error checking tmux: {e}")
        await output_queue.put(f"Error checking tmux: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/server/reconnect-tmux")
async def reconnect_tmux():
    """Reconnect to the tmux session if it exists"""
    try:
        # Check if the vLLM server tmux session exists
        vllm_session = None
        try:
            with open("vllm_tmux_session.txt", "r") as f:
                vllm_session = f.read().strip()
        except FileNotFoundError:
            await output_queue.put("No vLLM tmux session file found.")
            return {"status": "error", "message": "No session file found"}
        
        # Check if the session exists
        process = await asyncio.create_subprocess_exec(
            "tmux", "has-session", "-t", vllm_session,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.communicate()
        
        if process.returncode != 0:
            await output_queue.put(f"Tmux session {vllm_session} does not exist.")
            return {"status": "error", "message": f"Session {vllm_session} not found"}
        
        # Start monitoring the session again
        asyncio.create_task(monitor_tmux_session(vllm_session))
        
        await output_queue.put(f"Reconnected to tmux session: {vllm_session}")
        return {"status": "ok", "session": vllm_session}
    except Exception as e:
        logger.error(f"Error reconnecting to tmux: {e}")
        await output_queue.put(f"Error reconnecting to tmux: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/server/debug-tmux")
async def debug_tmux():
    """Debug tmux sessions"""
    try:
        # Run tmux ls directly
        process = await asyncio.create_subprocess_exec(
            "tmux", "ls",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        stdout, _ = await process.communicate()
        
        # Check if vllm_tmux_session.txt exists
        session_file_exists = os.path.exists("vllm_tmux_session.txt")
        session_name = None
        if session_file_exists:
            with open("vllm_tmux_session.txt", "r") as f:
                session_name = f.read().strip()
        
        return {
            "tmux_ls_output": stdout.decode() if stdout else "",
            "tmux_ls_returncode": process.returncode,
            "session_file_exists": session_file_exists,
            "session_name": session_name
        }
    except Exception as e:
        logger.error(f"Error debugging tmux: {e}")
        return {"error": str(e)}

@app.get("/api/server/check-process")
async def check_process():
    """Check if the vLLM server process is still running inside the tmux session"""
    try:
        # Check if the vLLM server tmux session exists
        vllm_session = None
        try:
            with open("vllm_tmux_session.txt", "r") as f:
                vllm_session = f.read().strip()
        except FileNotFoundError:
            return {"status": "not_found", "message": "No session file found"}
        
        # Check if the session exists
        process = await asyncio.create_subprocess_exec(
            "tmux", "has-session", "-t", vllm_session,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.communicate()
        
        if process.returncode != 0:
            return {"status": "not_running", "message": f"Session {vllm_session} not found"}
        
        # Check if the Python process is still running inside the session
        # This uses pgrep to find Python processes with SERVER_SCRIPT_PATH in their command line
        process = await asyncio.create_subprocess_exec(
            "tmux", "capture-pane", "-pt", vllm_session, "-S", "-1000",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        stdout, _ = await process.communicate()
        
        if process.returncode == 0 and stdout:
            output = stdout.decode('utf-8', errors='replace')
            
            # Check for error indicators in the output
            error_indicators = [
                "CUDA error",
                "RuntimeError",
                "Error:",
                "Exception:",
                "Traceback (most recent call last)",
                "Server process exited with code"
            ]
            
            errors_found = []
            for indicator in error_indicators:
                if indicator in output:
                    errors_found.append(indicator)
            
            if errors_found:
                return {
                    "status": "error",
                    "message": f"Errors detected in session {vllm_session}",
                    "errors": errors_found,
                    "session_alive": True
                }
            
            # Check if the session is waiting for input (our script's "Press Enter to close" state)
            if "Press Enter to close this session" in output:
                return {
                    "status": "waiting",
                    "message": f"Session {vllm_session} is waiting for input (server process has ended)",
                    "session_alive": True
                }
            
            return {
                "status": "running",
                "message": f"Session {vllm_session} is running",
                "session_alive": True
            }
        
        return {
            "status": "unknown",
            "message": f"Session {vllm_session} exists but couldn't capture output",
            "session_alive": True
        }
    except Exception as e:
        logger.error(f"Error checking process: {e}")
        return {"status": "error", "message": str(e)}

async def output_queue_consumer():
    """Consume messages from the output queue and broadcast to all connected websockets"""
    try:
        logger.info("Starting output queue consumer")
        while True:
            try:
                message = await output_queue.get()
                
                # Check if this is a special message type
                if isinstance(message, dict) and "type" in message:
                    # Already formatted as a message
                    json_message = message
                else:
                    # Create a JSON message to send to clients
                    json_message = {
                        "type": "output",
                        "data": message
                    }
                
                # Broadcast to all connected websockets
                if connected_websockets:
                    for websocket in connected_websockets.copy():
                        try:
                            await websocket.send_text(json.dumps(json_message, ensure_ascii=False))
                        except Exception as e:
                            logger.error(f"Error sending to websocket: {e}")
                
                # Mark the task as done
                output_queue.task_done()
            except Exception as e:
                logger.error(f"Error in output queue consumer: {e}")
                await asyncio.sleep(1)
    except Exception as e:
        logger.critical(f"Output queue consumer crashed: {e}", exc_info=True)

@app.post("/api/server/test-tmux-output")
async def test_tmux_output():
    """Test tmux output streaming by sending some test messages"""
    try:
        # Check if the vLLM server tmux session exists
        vllm_session = None
        try:
            with open("vllm_tmux_session.txt", "r") as f:
                vllm_session = f.read().strip()
        except FileNotFoundError:
            await output_queue.put("No vLLM tmux session file found.")
            return {"status": "error", "message": "No session file found"}
        
        # Check if the session exists
        process = await asyncio.create_subprocess_exec(
            "tmux", "has-session", "-t", vllm_session,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.communicate()
        
        if process.returncode != 0:
            await output_queue.put(f"Tmux session {vllm_session} does not exist.")
            return {"status": "error", "message": f"Session {vllm_session} not found"}
        
        # Send some test messages to the output queue
        await output_queue.put(f"=== TEST: Beginning tmux output test for session {vllm_session} ===")
        
        # Get the current content of the tmux session
        process = await asyncio.create_subprocess_exec(
            "tmux", "capture-pane", "-pt", vllm_session,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        stdout, _ = await process.communicate()
        
        if process.returncode == 0 and stdout:
            content = stdout.decode('utf-8', errors='replace')
            lines = content.splitlines()
            
            await output_queue.put(f"Found {len(lines)} lines in tmux session")
            
            # Send the last 10 lines
            await output_queue.put("Last 10 lines from tmux session:")
            for line in lines[-10:]:
                if line.strip():
                    await output_queue.put(f"TMUX: {line}")
        
        # Send a command to the tmux session to generate some output
        process = await asyncio.create_subprocess_exec(
            "tmux", "send-keys", "-t", vllm_session, "echo 'Test message from API server'", "C-m",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.communicate()
        
        await output_queue.put("=== TEST: End of tmux output test ===")
        
        return {"status": "ok", "session": vllm_session}
    except Exception as e:
        logger.error(f"Error testing tmux output: {e}")
        await output_queue.put(f"Error testing tmux output: {e}")
        return {"status": "error", "message": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    try:
        # Start the output queue consumer
        app.output_consumer_task = asyncio.create_task(output_queue_consumer())
        logger.info("Output queue consumer started")
    except Exception as e:
        logger.error(f"Error starting output queue consumer: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    try:
        # Cancel the output queue consumer task if it exists
        if hasattr(app, "output_consumer_task"):
            app.output_consumer_task.cancel()
            try:
                await app.output_consumer_task
            except asyncio.CancelledError:
                pass
            logger.info("Output queue consumer stopped")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.post("/api/server/kill-tmux-session")
async def kill_tmux_session(request: dict):
    """Kill a tmux session"""
    try:
        session_name = request.get("session")
        if not session_name:
            return {"status": "error", "message": "No session name provided"}
        
        # Check if the session exists
        process = await asyncio.create_subprocess_exec(
            "tmux", "has-session", "-t", session_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.communicate()
        
        if process.returncode != 0:
            return {"status": "error", "message": f"Session {session_name} not found"}
        
        # Kill the session
        process = await asyncio.create_subprocess_exec(
            "tmux", "kill-session", "-t", session_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        stdout, _ = await process.communicate()
        
        if process.returncode == 0:
            await output_queue.put(f"Killed tmux session: {session_name}")
            return {"status": "ok", "message": f"Session {session_name} killed successfully"}
        else:
            error_message = stdout.decode() if stdout else "Unknown error"
            await output_queue.put(f"Error killing tmux session {session_name}: {error_message}")
            return {"status": "error", "message": error_message}
    except Exception as e:
        logger.error(f"Error killing tmux session: {e}")
        return {"status": "error", "message": str(e)}

def main():
    """Main function to start the API server"""
    try:
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Set up logging to capture any startup errors
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"logs/api_startup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        
        logger.info("Starting LLM Server Manager API")
        
        # Mount the frontend static files
        frontend_dir = Path(__file__).parent / "frontend"
        if frontend_dir.exists():
            app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
            logger.info(f"Mounted frontend at /ui from {frontend_dir}")
            
            # Mount admin interface
            admin_dir = frontend_dir / "admin"
            if not admin_dir.exists():
                admin_dir.mkdir(exist_ok=True)
            app.mount("/admin", StaticFiles(directory=str(admin_dir), html=True), name="admin")
            logger.info("Mounted admin interface at /admin")
        else:
            logger.warning(f"Frontend directory not found at {frontend_dir}")
        
        # Get local IP for display
        local_ip = get_local_ip()
        port = 8080
        
        print(f"\n{'='*50}")
        print(f"LLM Server Manager API starting on:")
        print(f"- Local:   http://localhost:{port}/ui/")
        print(f"- Network: http://{local_ip}:{port}/ui/")
        print(f"- Admin:   http://localhost:{port}/admin/errors.html")
        print(f"{'='*50}\n")
        
        # Start the server with a startup event handler to initialize the output queue consumer
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        logger.critical(f"Fatal error starting API server: {e}", exc_info=True)
        print(f"ERROR: Failed to start API server: {e}")

if __name__ == "__main__":
    main() 