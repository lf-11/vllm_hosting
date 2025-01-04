import logging
import requests
import time
from typing import List, Dict, Any, Union, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import psycopg2
from psycopg2.extras import DictCursor
import json

logger = logging.getLogger(__name__)

class VLLMClient:
    """Client for making requests to vLLM server"""
    
    def __init__(self, server_url: str = "http://localhost:8000", max_batch_size: Optional[int] = None):
        self.server_url = server_url.rstrip('/')
        self.max_batch_size = max_batch_size
        self.current_model: Optional[str] = None
        self.current_model_id: Optional[int] = None
        
        # Database configuration
        self.db_config = {
            'dbname': 'vllm_hosting',
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost'
        }
    
    def _get_db_connection(self):
        """Create and return a database connection"""
        return psycopg2.connect(**self.db_config)
    
    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model configuration from database"""
        with self._get_db_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT * FROM models 
                    WHERE model_name = %s 
                    ORDER BY id LIMIT 1
                """, (model_name,))
                result = cur.fetchone()
                if not result:
                    raise ValueError(f"Model {model_name} not found in database")
                
                # Convert database row to dict and add default parameters
                model_config = dict(result)
                self.current_model_id = model_config['id']
                model_config['default_params'] = {
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "top_p": 0.95,
                }
                return model_config
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to server with retries"""
        try:
            response = requests.post(
                f"{self.server_url}/v1/{endpoint}",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error making request to {endpoint}: {e}")
            raise

    def generate(
        self,
        prompts: Union[str, List[str]],
        model: str,
        mode: str = "completion",
        params: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None
    ) -> List[str]:

        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Get model config from database and merge parameters
        model_config = self._get_model_config(model)
        request_params = model_config["default_params"].copy()
        if params:
            request_params.update(params)
            
        # Use model path for API requests
        model_path = model_config["model_path"]
        self.current_model = model_path
            
        # Determine batch size
        if batch_size is None:
            batch_size = model_config.get('max_batch_size', 1)
        effective_batch_size = min(batch_size, self.max_batch_size) if self.max_batch_size else batch_size
        
        # Process prompts in batches
        all_completions = []
        for i in range(0, len(prompts), effective_batch_size):
            batch = prompts[i:i + effective_batch_size]
            
            if mode == "chat" and model_config.get('chat_template'):
                completions = self._process_chat_batch(batch, request_params)
            else:
                completions = self._process_completion_batch(batch, request_params)
                
            all_completions.extend(completions)
            
        return all_completions
    
    def _store_generation(
        self,
        input_text: str,
        output_text: str,
        tokens_generated: int,
        time_taken: float,
        parameters: Dict[str, Any]
    ) -> None:
        """Store generation details in the database"""
        with self._get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO generations 
                    (input_text, output_text, model_id, tokens_generated, time_taken, parameters)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    input_text,
                    output_text,
                    self.current_model_id,
                    tokens_generated,
                    time_taken,
                    json.dumps(parameters)
                ))
            conn.commit()

    def _process_completion_batch(
        self, 
        prompts: List[str],
        params: Dict[str, Any]
    ) -> List[str]:
        """Process a batch of completion requests"""
        payload = {
            "prompt": prompts,
            "model": self.current_model,
            **params
        }
        
        start_time = time.time()
        response = self._make_request("completions", payload)
        time_taken = time.time() - start_time
        
        completions = []
        for idx, choice in enumerate(response["choices"]):
            text = choice["text"].strip()
            completions.append(text)
            
            # Get token count from response
            tokens_generated = choice.get("usage", {}).get("completion_tokens", 0)
            if not tokens_generated:  # Fallback if usage not provided
                tokens_generated = len(choice.get("text", "").split())
            
            self._store_generation(
                input_text=prompts[idx],
                output_text=text,
                tokens_generated=tokens_generated,
                time_taken=time_taken / len(prompts),
                parameters=params
            )
            
        if len(completions) != len(prompts):
            raise RuntimeError(f"Got {len(completions)} completions for {len(prompts)} prompts")
            
        return completions
    
    def _process_chat_batch(
        self,
        prompts: List[str],
        params: Dict[str, Any]
    ) -> List[str]:
        """Process a batch of chat requests"""
        messages_list = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            messages_list.append(messages)
            
        completions = []
        for idx, messages in enumerate(messages_list):
            payload = {
                "model": self.current_model,
                "messages": messages,
                **params
            }
            
            start_time = time.time()
            response = self._make_request("chat/completions", payload)
            time_taken = time.time() - start_time
            
            text = response["choices"][0]["message"]["content"].strip()
            completions.append(text)
            
            # Get token count from response
            tokens_generated = response.get("usage", {}).get("completion_tokens", 0)
            if not tokens_generated:  # Fallback if usage not provided
                tokens_generated = len(text.split())
            
            self._store_generation(
                input_text=prompts[idx],
                output_text=text,
                tokens_generated=tokens_generated,
                time_taken=time_taken,
                parameters=params
            )
            
        return completions