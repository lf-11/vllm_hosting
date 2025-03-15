import psycopg2
import psycopg2.extras
from config import DB_CONFIG
import logging

logger = logging.getLogger(__name__)

def get_db_connection():
    """Get a connection to the database"""
    return psycopg2.connect(**DB_CONFIG)

def get_all_models():
    """Get all models from the database"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    cur.execute("""
        SELECT * FROM models
        ORDER BY model_name, max_model_len
    """)
    
    models = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return [dict(model) for model in models]

def get_model_by_id(model_id):
    """Get a model by its ID"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    cur.execute("""
        SELECT * FROM models
        WHERE id = %s
    """, (model_id,))
    
    model = cur.fetchone()
    
    cur.close()
    conn.close()
    
    return dict(model) if model else None

def get_model_configurations(model_name):
    """Get all configurations for a specific model name"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    cur.execute("""
        SELECT * FROM models
        WHERE model_name = %s
        ORDER BY max_model_len
    """, (model_name,))
    
    configs = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return [dict(config) for config in configs]

def add_model(model_data):
    """Add a new model to the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO models (
            model_name, model_path, tensor_parallel_size, max_model_len,
            gpu_memory_utilization, enforce_eager, chat_template,
            speculative_decoding, max_batch_size
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id
    """, (
        model_data['model_name'],
        model_data['model_path'],
        model_data['tensor_parallel_size'],
        model_data['max_model_len'],
        model_data['gpu_memory_utilization'],
        model_data['enforce_eager'],
        model_data.get('chat_template'),
        model_data.get('speculative_decoding', False),
        model_data.get('max_batch_size')
    ))
    
    model_id = cur.fetchone()[0]
    
    conn.commit()
    cur.close()
    conn.close()
    
    return model_id 

def get_known_errors():
    """Get all known errors from the database"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    cur.execute("""
        SELECT * FROM error_tracking
        ORDER BY occurrence_count DESC
    """)
    
    errors = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return [dict(error) for error in errors]

def find_matching_error(error_text):
    """Find a known error that matches the given error text"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Get all error patterns
    cur.execute("SELECT * FROM error_tracking")
    errors = cur.fetchall()
    
    matching_errors = []
    for error in errors:
        # Use regex to match error pattern
        import re
        try:
            if re.search(error['error_pattern'], error_text, re.IGNORECASE):
                matching_errors.append(dict(error))
                
                # Update occurrence count and last_seen
                cur.execute("""
                    UPDATE error_tracking 
                    SET occurrence_count = occurrence_count + 1,
                        last_seen = now()
                    WHERE id = %s
                """, (error['id'],))
                
                # Log that we found a match
                logger.info(f"Found matching error pattern: {error['error_pattern']}")
        except Exception as e:
            logger.error(f"Error matching pattern '{error['error_pattern']}': {e}")
    
    conn.commit()
    cur.close()
    conn.close()
    
    # Log the result
    if matching_errors:
        logger.info(f"Found {len(matching_errors)} matching errors")
    else:
        logger.info("No matching errors found")
        
    return matching_errors

def add_new_error(error_pattern, error_message, solution):
    """Add a new error to the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO error_tracking (error_pattern, error_message, solution)
        VALUES (%s, %s, %s)
        RETURNING id
    """, (error_pattern, error_message, solution))
    
    error_id = cur.fetchone()[0]
    
    conn.commit()
    cur.close()
    conn.close()
    
    return error_id 