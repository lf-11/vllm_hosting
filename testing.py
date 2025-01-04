import logging
import requests
from client import VLLMClient
from time import sleep
from typing import Optional
import psycopg2
from psycopg2.extras import DictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection with fixed configuration"""
    return psycopg2.connect(
        dbname='vllm_hosting',
        user='postgres',
        password='postgres',
        host='localhost'
    )

def get_current_model() -> Optional[str]:
    """Get the currently loaded model from server"""
    try:
        response = requests.get(
            "http://localhost:8000/v1/models",
            timeout=5
        )
        response.raise_for_status()
        models_data = response.json()
        
        if models_data.get("data") and len(models_data["data"]) > 0:
            model_path = models_data["data"][0]["id"]  # Get first model's path
            
            # Query database to get model name from path
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute("SELECT model_name FROM models WHERE model_path = %s", (model_path,))
                    result = cur.fetchone()
                    if result:
                        model_name = result['model_name']
                        logger.info(f"Found loaded model: {model_name}")
                        return model_name
                    else:
                        logger.warning(f"Model path {model_path} not found in database")
                        return None
        else:
            logger.info("No models found in server response")
            return None
            
    except Exception as e:
        logger.error(f"Error getting current model: {e}")
        return None

def get_model_info(model_name: str) -> Optional[dict]:
    """Get model configuration from database"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT * FROM models 
                    WHERE model_name = %s 
                    ORDER BY id LIMIT 1
                """, (model_name,))
                result = cur.fetchone()
                return dict(result) if result else None
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return None

def test_basic_completion(model_name: str):
    client = VLLMClient()
    prompts = [
        "Explain the concept of recursion in programming.",
        "Write a haiku about artificial intelligence.",
        "What are the three laws of thermodynamics?",
        "Describe the taste of an orange to someone who has never tasted one."
    ]
    
    # Test single random prompt
    from random import choice
    test_prompt = choice(prompts)
    print("\nSingle completion test:")
    print(f"Prompt: {test_prompt}")
    result = client.generate(test_prompt, model=model_name)
    print(f"Response: {result[0]}")

def test_batch_completion(model_name: str):
    client = VLLMClient()
    model_info = get_model_info(model_name)
    
    if not model_info:
        logger.error(f"Could not find configuration for model {model_name}")
        return
        
    max_batch_size = model_info.get('max_batch_size')
    if not max_batch_size:
        logger.warning("Max batch size not set in database, using default of 2")
        max_batch_size = 2
    
    # Diverse set of prompts testing different capabilities
    prompts = [
        # Creative writing
        "Write a short story about a robot discovering emotions.",
        
        # Analytical thinking
        "Compare and contrast renewable and non-renewable energy sources.",
        
        # Technical explanation
        "Explain how a blockchain works to a 10-year-old.",
        
        # Problem solving
        "A train leaves Paris at 3pm traveling at 200 km/h towards Berlin (1000km away). Another train leaves Berlin at 4pm traveling at 250 km/h towards Paris. At what time will they meet?",
        
        # Abstract thinking
        "If colors had sounds, what would blue sound like?",
        
        # Factual knowledge
        "What are the main differences between Python and JavaScript?",
        
        # Summarization
        "Summarize the theory of evolution in three sentences."
    ]
    
    # Test batch completion mode with random subset of prompts
    from random import sample
    batch_size = min(3, max_batch_size)  # Use smaller of 3 or max_batch_size
    test_prompts = sample(prompts, batch_size)
    
    print(f"\nBatch completion results (batch_size={batch_size}):")
    results = client.generate(test_prompts, model=model_name, batch_size=batch_size)
    for prompt, result in zip(test_prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")
    
    # Test batch chat mode with same prompts
    if model_info.get('chat_template'):
        print("\nBatch chat results:")
        chat_results = client.generate(test_prompts, model=model_name, mode="chat", batch_size=batch_size)
        for prompt, result in zip(test_prompts, chat_results):
            print(f"\nPrompt: {prompt}")
            print(f"Response: {result}")
    else:
        print("\nSkipping chat mode test - no chat template configured for model")

if __name__ == '__main__':
    # Get currently loaded model
    model_name = get_current_model()
    if not model_name:
        print("No model currently loaded. Please start the server first.")
        exit(1)
    
    print(f"Testing with model: {model_name}")
    
    print("Running single completion test...")
    test_basic_completion(model_name)
    sleep(1)
    
    print("\nRunning batch tests...")
    test_batch_completion(model_name) 