import argparse
import logging
import time
from server import VLLMServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_db_config():
    """Get fixed database configuration"""
    return {
        'dbname': 'vllm_hosting',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost'
    }

def main():
    parser = argparse.ArgumentParser(description='Manage vLLM server with database integration')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'force-kill'], 
                      help='Action to perform')
    parser.add_argument('--model', type=str, help='Model name to load')
    
    # Add custom parameter options
    parser.add_argument('--max-model-len', type=int, 
                      help='Maximum sequence length for the model')
    parser.add_argument('--speculative-decoding', action='store_true',
                      help='Enable speculative decoding')
    
    args = parser.parse_args()
    
    # Initialize server with fixed database configuration
    server = VLLMServer(db_config=get_db_config())
    
    # Prepare custom parameters if any are specified
    custom_params = {}
    if args.max_model_len:
        custom_params['max_model_len'] = args.max_model_len
    if args.speculative_decoding:
        custom_params['speculative_decoding'] = True
    
    if args.action == 'start':
        if not args.model:
            parser.error("--model is required for start action")
        server.start(args.model, **custom_params)
        # Keep script running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nReceived interrupt, shutting down...")
            server.shutdown()
            
    elif args.action == 'stop':
        server.shutdown()
        
    elif args.action == 'restart':
        if not args.model:
            parser.error("--model is required for restart action")
        server.shutdown()
        server.start(args.model, **custom_params)
        # Keep script running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nReceived interrupt, shutting down...")
            server.shutdown()
            
    elif args.action == 'force-kill':
        server._cleanup_existing()

if __name__ == '__main__':
    main() 