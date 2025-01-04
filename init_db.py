import psycopg2

def init_db():
    # Database connection parameters
    db_params = {
        'dbname': 'vllm_hosting',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost'
    }
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_params)
    conn.autocommit = True
    cur = conn.cursor()
    
    # Create sequence if not exists
    cur.execute("""
        CREATE SEQUENCE IF NOT EXISTS models_id_seq
        INCREMENT 1
        START 1
        MINVALUE 1
        MAXVALUE 2147483647
        CACHE 1;
    """)
    
    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.models
        (
            id integer NOT NULL DEFAULT nextval('models_id_seq'::regclass),
            model_name text COLLATE pg_catalog."default" NOT NULL,
            generations_count bigint DEFAULT 0,
            tokens_generated_count bigint DEFAULT 0,
            avg_tokens_per_second double precision DEFAULT 0.0,
            model_path text COLLATE pg_catalog."default" NOT NULL,
            tensor_parallel_size integer NOT NULL,
            max_model_len integer NOT NULL,
            gpu_memory_utilization double precision,
            enforce_eager boolean NOT NULL,
            chat_template text COLLATE pg_catalog."default",
            speculative_decoding boolean DEFAULT false,
            created_at timestamp without time zone DEFAULT now(),
            max_batch_size integer,
            CONSTRAINT models_pkey PRIMARY KEY (id)
        )
    """)
    
    # Close connection
    cur.close()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!") 