import os
import pymysql
import pymysql.cursors
from contextlib import contextmanager
from dotenv import load_dotenv

# Absolute path resolution to strictly load .env from agent/customer/.env
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    # Fallback to standard load if not found locally
    load_dotenv()

# Strict Environment Variable Validation (No hardcoded fallback defaults in code)
DB_HOST = os.getenv("DB_HOST")
DB_PORT_STR = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

missing_vars = []
if not DB_HOST: missing_vars.append("DB_HOST")
if not DB_PORT_STR: missing_vars.append("DB_PORT")
if not DB_USER: missing_vars.append("DB_USER")
if not DB_PASSWORD: missing_vars.append("DB_PASSWORD")
if not DB_NAME: missing_vars.append("DB_NAME")

if missing_vars:
    raise ValueError(
        f"Database configuration error: The following required environment variables "
        f"are missing from the .env file: {', '.join(missing_vars)}. "
        f"Please verify your local .env file at {env_path}"
    )

DB_PORT = int(DB_PORT_STR)

@contextmanager
def get_db_connection():
    """
    Context manager that yields a database connection.
    Automatically handles commits and rollbacks.
    """
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        yield connection
        connection.commit()
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        connection.close()

@contextmanager
def get_db_cursor():
    """
    Context manager that yields a cursor from a database connection.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            yield cursor
