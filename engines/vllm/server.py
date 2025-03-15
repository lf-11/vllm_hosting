import subprocess
import sys
import time
import signal
import logging
import re
import atexit
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import os
import psutil
import psycopg2
from psycopg2.extras import DictCursor

logger = logging.getLogger(__name__)

class VLLMServer:
    """Manages vLLM server lifecycle with PostgreSQL integration"""
    
    def __init__(self, host: str = "localhost", port: int = 8000, 
                 db_config: Dict[str, Any] = None):
        self.process: Optional[subprocess.Popen] = None
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.max_concurrency: Optional[int] = None
        self.current_model: Optional[str] = None
        
        # Database configuration
        self.db_config = db_config or {
            'dbname': 'postgres',
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost'
        }
        
        # Store PID in a file for recovery
        self.pid_file = Path("vllm_server.pid")
        self.log_file: Optional[Path] = None
        
        # Clean up any existing server
        self._cleanup_existing()
        
        # Register cleanup handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, lambda signo, frame: self.shutdown())
        signal.signal(signal.SIGINT, lambda signo, frame: self.shutdown())
        
        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging directory and file"""
        log_dir = Path("logs/vllm")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"vllm_{time.strftime('%Y%m%d')}.log"

    def _get_db_connection(self):
        """Create and return a database connection"""
        return psycopg2.connect(**self.db_config)

    def _get_model_config(self, model_name: str, custom_params: Optional[Dict] = None) -> Dict:
        """Get model configuration from database"""
        with self._get_db_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                if custom_params:
                    # Check if config with custom params exists
                    query = """
                        SELECT * FROM models 
                        WHERE model_name = %s 
                        AND max_model_len = %s 
                        AND speculative_decoding = %s
                        LIMIT 1
                    """
                    cur.execute(query, (
                        model_name,
                        custom_params.get('max_model_len'),
                        custom_params.get('speculative_decoding', False)
                    ))
                else:
                    # Get default config (lowest ID)
                    query = "SELECT * FROM models WHERE model_name = %s ORDER BY id LIMIT 1"
                    cur.execute(query, (model_name,))
                
                result = cur.fetchone()
                
                if not result and custom_params:
                    # Create new config if custom params specified
                    insert_query = """
                        INSERT INTO models (
                            model_name, model_path, tensor_parallel_size, 
                            max_model_len, gpu_memory_utilization, enforce_eager,
                            chat_template, speculative_decoding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING *
                    """
                    # Get base config for required fields
                    cur.execute("SELECT * FROM models WHERE model_name = %s LIMIT 1", (model_name,))
                    base_config = cur.fetchone()
                    
                    if not base_config:
                        raise ValueError(f"No base configuration found for model {model_name}")
                    
                    cur.execute(insert_query, (
                        model_name,
                        base_config['model_path'],
                        base_config['tensor_parallel_size'],
                        custom_params.get('max_model_len', base_config['max_model_len']),
                        base_config['gpu_memory_utilization'],
                        base_config['enforce_eager'],
                        base_config['chat_template'],
                        custom_params.get('speculative_decoding', False)
                    ))
                    conn.commit()
                    result = cur.fetchone()
                
                if not result:
                    raise ValueError(f"No configuration found for model {model_name}")
                
                return dict(result)

    def start(self, model_name: str, **custom_params) -> None:
        """Start vLLM server with specified model and parameters"""
        if self.process:
            if self.current_model == model_name:
                logger.info("Server already running with requested model")
                return
            self.shutdown()
        
        logger.info(f"Starting vLLM server with model: {model_name}")
        
        # Get model configuration
        model_config = self._get_model_config(model_name, custom_params if custom_params else None)
        
        # Construct command with correct format
        command = [
            "vllm",
            "serve",
            model_config['model_path'],  # Model path directly after 'serve'
            "--tensor-parallel-size", str(model_config['tensor_parallel_size']),
            "--host", self.host,
            "--port", str(self.port),
            "--max-model-len", str(model_config['max_model_len'])
        ]
        if model_config['gpu_memory_utilization'] is not None:
            command.extend(["--gpu-memory-utilization", str(model_config['gpu_memory_utilization'])])
            
        if model_config['enforce_eager']:
            command.append("--enforce-eager")
        
        if model_config['speculative_decoding']:
            command.append("--enable-speculative-decoding")
        
        # Add chat template if specified
        if model_config['chat_template']:
            command.extend(["--chat-template", model_config['chat_template']])
        
        logger.info(f"Running command: {' '.join(command)}")
        
        try:
            # Start server process WITH output capture
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )
            
            with open(self.pid_file, 'w') as f:
                f.write(str(self.process.pid))
            
            # Wait for server to be ready with output monitoring
            max_attempts = 20
            for attempt in range(max_attempts):
                # Read output while waiting
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        break
                        
                    print(line, end='')  # Show output in terminal
                    logger.debug(f"Processing line: {line.strip()}")  # Debug log
                    
                    # Look for concurrency information
                    match = re.search(r"INFO .* Maximum concurrency for (\d+) tokens per request: ([\d.]+)x", line)
                    if match:
                        logger.info(f"Found concurrency line: {line.strip()}")  # Debug log
                        tokens = int(match.group(1))
                        max_concurrency = int(float(match.group(2)))  # Convert to float first, then truncate to int
                        logger.info(f"Parsed values: tokens={tokens}, max_concurrency={max_concurrency}")  # Debug log
                        
                        if model_config['max_batch_size'] is None:
                            # Update in database
                            with self._get_db_connection() as conn:
                                with conn.cursor() as cur:
                                    cur.execute("""
                                        UPDATE models 
                                        SET max_batch_size = %s 
                                        WHERE id = %s
                                    """, (max_concurrency, model_config['id']))
                                    conn.commit()
                                    logger.info(f"Updated database with max_concurrency={max_concurrency}")  # Debug log
                            self.max_concurrency = max_concurrency
                    
                # Check if process has terminated
                if self.process.poll() is not None:
                    exit_code = self.process.returncode
                    self.shutdown()
                    raise RuntimeError(f"Server process terminated unexpectedly with exit code {exit_code}")
                
                if self.is_running():
                    logger.info("vLLM server is ready!")
                    break
                    
                time.sleep(10)
                logger.info(f"Waiting for server... (attempt {attempt + 1}/{max_attempts})")
            else:
                self.shutdown()
                raise RuntimeError("Server failed to start within timeout")
            
            self.current_model = model_name
            
        except Exception as e:
            self.shutdown()
            raise

        # Start a thread to continuously read and display output
        def _output_reader():
            while self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    print(line, end='')
        
        import threading
        threading.Thread(target=_output_reader, daemon=True).start()

    def is_running(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def _cleanup_existing(self):
        """Clean up any existing vLLM server process"""
        if self.pid_file.exists():
            try:
                with open(self.pid_file) as f:
                    old_pid = int(f.read().strip())
                try:
                    process = psutil.Process(old_pid)
                    if "vllm" in process.name().lower() or "vllm" in " ".join(process.cmdline()).lower():
                        logger.warning(f"Found existing vLLM server (PID: {old_pid}). Terminating...")
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except psutil.TimeoutExpired:
                            process.kill()
                except psutil.NoSuchProcess:
                    pass
            except Exception as e:
                logger.error(f"Error cleaning up existing server: {e}")
            self.pid_file.unlink()
    
    def shutdown(self) -> None:
        """Shutdown vLLM server"""
        if self.process:
            logger.info("Shutting down vLLM server...")
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)  # Reduced timeout to 10 seconds
                except subprocess.TimeoutExpired:
                    logger.warning("Server not responding to SIGTERM, using SIGKILL")
                    self.process.kill()
                    self.process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
                # Try harder to kill it
                try:
                    self.process.kill()
                    self.process.wait(timeout=5)
                except:
                    pass
            
            self.process = None
            self.current_model = None
            self.max_concurrency = None
            
        # Clean up PID file
        try:
            self.pid_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error removing PID file: {e}")
    
    

def get_db_config():
    """Get fixed database configuration"""
    return {
        'dbname': 'lm_hosting',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost'
    }

def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Manage vLLM server with database integration')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'force-kill'], 
                      help='Action to perform')
    parser.add_argument('--model', type=str, help='Model name to load')
    parser.add_argument('--max-model-len', type=int, 
                      help='Maximum sequence length for the model')
    parser.add_argument('--speculative-decoding', action='store_true',
                      help='Enable speculative decoding')
    parser.add_argument('--tensor-parallel-size', type=int, 
                      help='Tensor parallel size for model loading')
    parser.add_argument('--gpu-memory-utilization', type=float,
                      help='GPU memory utilization (0.0 to 1.0)')
    parser.add_argument('--enforce-eager', action='store_true',
                      help='Enforce eager execution')
    parser.add_argument('--chat-template', type=str,
                      help='Chat template to use')
    parser.add_argument('--max-batch-size', type=int,
                      help='Maximum batch size for inference')
    parser.add_argument('--use-tmux', action='store_true',
                      help='Run the server in a tmux session')
    
    args = parser.parse_args()
    
    # Initialize server with fixed database configuration
    server = VLLMServer(db_config=get_db_config())
    
    # Prepare custom parameters if any are specified
    custom_params = {}
    if args.max_model_len:
        custom_params['max_model_len'] = args.max_model_len
    if args.speculative_decoding:
        custom_params['speculative_decoding'] = True
    if args.tensor_parallel_size:
        custom_params['tensor_parallel_size'] = args.tensor_parallel_size
    if args.gpu_memory_utilization:
        custom_params['gpu_memory_utilization'] = args.gpu_memory_utilization
    if args.enforce_eager:
        custom_params['enforce_eager'] = True
    if args.chat_template:
        custom_params['chat_template'] = args.chat_template
    if args.max_batch_size:
        custom_params['max_batch_size'] = args.max_batch_size
    
    try:
        if args.action == 'start':
            if not args.model:
                parser.error("--model is required for start action")
                
            if args.use_tmux:
                # Start in tmux session
                import subprocess
                
                # Create a unique tmux session name based on the model
                tmux_session = f"vllm_server_{args.model.replace('/', '_')}"
                
                # Check if session already exists
                check_cmd = ["tmux", "has-session", "-t", tmux_session]
                if subprocess.run(check_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                    print(f"Session {tmux_session} already exists. Killing it...")
                    subprocess.run(["tmux", "kill-session", "-t", tmux_session])
                
                # Construct the command to run inside tmux
                cmd_parts = [sys.executable, __file__, "start", "--model", args.model]
                
                # Add all the other arguments except --use-tmux
                if args.max_model_len:
                    cmd_parts.extend(["--max-model-len", str(args.max_model_len)])
                if args.speculative_decoding:
                    cmd_parts.append("--speculative-decoding")
                if args.tensor_parallel_size:
                    cmd_parts.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
                if args.gpu_memory_utilization:
                    cmd_parts.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
                if args.enforce_eager:
                    cmd_parts.append("--enforce-eager")
                if args.chat_template:
                    cmd_parts.extend(["--chat-template", args.chat_template])
                if args.max_batch_size:
                    cmd_parts.extend(["--max-batch-size", str(args.max_batch_size)])
                
                # Convert command parts to a single string
                cmd_str = " ".join(cmd_parts)
                
                # Start the tmux session
                tmux_cmd = ["tmux", "new-session", "-d", "-s", tmux_session, cmd_str]
                subprocess.run(tmux_cmd)
                
                print(f"vLLM server started in tmux session: {tmux_session}")
                print(f"To attach to this session, run: tmux attach -t {tmux_session}")
                
                # Write the tmux session name to a file for reference
                with open("vllm_tmux_session.txt", "w") as f:
                    f.write(tmux_session)
                
                # Exit this process as the real server is now running in tmux
                return
            else:
                # Start normally
                server.start(args.model, **custom_params)
                # Keep script running until interrupted
                while True:
                    time.sleep(1)
                
        elif args.action == 'stop':
            server.shutdown()
            
            # Also try to kill any tmux session
            try:
                with open("vllm_tmux_session.txt", "r") as f:
                    tmux_session = f.read().strip()
                    
                import subprocess
                subprocess.run(["tmux", "kill-session", "-t", tmux_session], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Killed tmux session: {tmux_session}")
            except FileNotFoundError:
                pass
            
        elif args.action == 'restart':
            if not args.model:
                parser.error("--model is required for restart action")
            server.shutdown()
            server.start(args.model, **custom_params)
            # Keep script running until interrupted
            while True:
                time.sleep(1)
                
        elif args.action == 'force-kill':
            server._cleanup_existing()
            
            # Also try to kill any tmux session
            try:
                with open("vllm_tmux_session.txt", "r") as f:
                    tmux_session = f.read().strip()
                    
                import subprocess
                subprocess.run(["tmux", "kill-session", "-t", tmux_session], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Force-killed tmux session: {tmux_session}")
            except FileNotFoundError:
                pass
            
    except KeyboardInterrupt:
        print("\nReceived interrupt, shutting down...")
        server.shutdown()

if __name__ == '__main__':
    main()
    
    
