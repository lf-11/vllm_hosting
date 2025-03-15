import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config import DB_CONFIG

def init_db():
    """Initialize the LM hosting database with all required tables and functions"""
    
    # First connect to default postgres database to create our database if it doesn't exist
    conn = psycopg2.connect(
        dbname='postgres',
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        host=DB_CONFIG['host']
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # Create database if it doesn't exist
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_CONFIG['dbname']}'")
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
    
    # Close connection to postgres database
    cur.close()
    conn.close()
    
    # Connect to our database
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    
    # Create sequences
    cur.execute("""
        CREATE SEQUENCE IF NOT EXISTS models_id_seq
        INCREMENT 1
        START 1
        MINVALUE 1
        MAXVALUE 2147483647
        CACHE 1;
        
        CREATE SEQUENCE IF NOT EXISTS generations_id_seq
        INCREMENT 1
        START 1
        MINVALUE 1
        MAXVALUE 2147483647
        CACHE 1;
    """)
    
    # Create models table
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
        );
    """)
    
    # Create generations table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.generations
        (
            id integer NOT NULL DEFAULT nextval('generations_id_seq'::regclass),
            input_text text COLLATE pg_catalog."default" NOT NULL,
            output_text text COLLATE pg_catalog."default" NOT NULL,
            model_id integer NOT NULL,
            tokens_generated integer NOT NULL,
            time_taken double precision NOT NULL,
            parameters jsonb NOT NULL,
            generated_at timestamp without time zone DEFAULT now(),
            CONSTRAINT generations_pkey PRIMARY KEY (id),
            CONSTRAINT generations_model_id_fkey FOREIGN KEY (model_id)
                REFERENCES public.models (id) MATCH SIMPLE
                ON UPDATE NO ACTION
                ON DELETE NO ACTION
        );
    """)
    
    # Create function for updating model stats
    cur.execute("""
        CREATE OR REPLACE FUNCTION public.update_model_stats()
            RETURNS trigger
            LANGUAGE 'plpgsql'
            COST 100
            VOLATILE NOT LEAKPROOF
        AS $BODY$
        BEGIN
            -- Update the statistics in the models table
            UPDATE models
            SET 
                generations_count = (
                    SELECT COUNT(*) 
                    FROM generations 
                    WHERE model_id = NEW.model_id
                ),
                tokens_generated_count = (
                    SELECT COALESCE(SUM(tokens_generated), 0)
                    FROM generations 
                    WHERE model_id = NEW.model_id
                ),
                avg_tokens_per_second = (
                    SELECT COALESCE(
                        SUM(tokens_generated) / NULLIF(SUM(time_taken), 0),
                        0
                    )
                    FROM generations 
                    WHERE model_id = NEW.model_id
                )
            WHERE id = NEW.model_id;
            
            RETURN NEW;
        END;
        $BODY$;
    """)
    
    # Create trigger
    cur.execute("""
        DROP TRIGGER IF EXISTS update_model_stats_trigger ON public.generations;
        
        CREATE TRIGGER update_model_stats_trigger
            AFTER INSERT
            ON public.generations
            FOR EACH ROW
            EXECUTE FUNCTION public.update_model_stats();
    """)
    
    # Set ownership
    cur.execute("""
        ALTER TABLE IF EXISTS public.models OWNER to postgres;
        ALTER TABLE IF EXISTS public.generations OWNER to postgres;
        ALTER FUNCTION public.update_model_stats() OWNER TO postgres;
    """)
    
    # Close connection
    cur.close()
    conn.close()
    
    print("Database initialized successfully!")
    print("Created:")
    print("- Database: vllm_hosting")
    print("- Tables: models, generations")
    print("- Function: update_model_stats()")
    print("- Trigger: update_model_stats_trigger")

if __name__ == "__main__":
    try:
        init_db()
    except Exception as e:
        print(f"Error initializing database: {e}") 