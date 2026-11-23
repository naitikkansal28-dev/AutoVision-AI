"""
AutoVision AI - Database Connection Manager
Handles connections for both MySQL (primary) and SQLite (automatic local fallback).
Provides uniform query execution and Pandas DataFrame extraction.
"""

import sqlite3
import re
import pandas as pd
from contextlib import contextmanager
import config

# Global flag to track active database backend
_ACTIVE_BACKEND = None


def test_mysql_connection():
    """Test if MySQL server is reachable with configured credentials."""
    try:
        import MySQLdb
        conn = MySQLdb.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            passwd=config.DB_PASSWORD,
            connect_timeout=2
        )
        cursor = conn.cursor()
        # Ensure database exists
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{config.DB_NAME}` CHARACTER SET utf8mb4;")
        conn.select_db(config.DB_NAME)
        cursor.close()
        conn.close()
        return True
    except Exception:
        return False


def get_backend():
    """Detect and cache whether MySQL or SQLite is active."""
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is not None:
        return _ACTIVE_BACKEND

    if test_mysql_connection():
        _ACTIVE_BACKEND = "mysql"
        print(f"[AutoVision AI] Connected to MySQL database '{config.DB_NAME}' at {config.DB_HOST}:{config.DB_PORT}")
    else:
        _ACTIVE_BACKEND = "sqlite"
        print(f"[AutoVision AI] MySQL unavailable. Using SQLite fallback: {config.SQLITE_DB_PATH}")

    return _ACTIVE_BACKEND


def get_connection():
    """Return an active connection object for the configured backend."""
    backend = get_backend()
    if backend == "mysql":
        import MySQLdb
        conn = MySQLdb.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            passwd=config.DB_PASSWORD,
            db=config.DB_NAME,
            charset="utf8mb4"
        )
        try:
            cur = conn.cursor()
            cur.execute("SET SESSION sql_mode = CONCAT(@@sql_mode, ',PIPES_AS_CONCAT');")
            cur.close()
        except Exception:
            pass
        return conn
    else:
        conn = sqlite3.connect(str(config.SQLITE_DB_PATH), timeout=15)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn


@contextmanager
def get_db_cursor(commit=False):
    """Context manager for acquiring a cursor and auto-closing/committing."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        if commit:
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def normalize_query(sql, backend):
    """
    Normalizes SQL parameter placeholders and date functions across dialects:
    SQLite uses '?' placeholders, while MySQLdb uses '%s'.
    """
    if backend == "sqlite":
        # If query has %s, convert to ?
        return sql.replace("%s", "?")
    else:
        # Convert ? to %s for MySQL
        normalized = sql.replace("?", "%s")
        # Adapt SQLite DATE('now', '-X day') to MySQL DATE_SUB(CURDATE(), INTERVAL X DAY)
        normalized = re.sub(r"DATE\('now',\s*'-(\d+)\s*day'\)", r"DATE_SUB(CURDATE(), INTERVAL \1 DAY)", normalized, flags=re.IGNORECASE)
        return normalized


def execute_query(sql, params=None, commit=True):
    """
    Executes an INSERT, UPDATE, or DELETE query safely.
    Returns the lastrowid or rowcount.
    """
    backend = get_backend()
    normalized_sql = normalize_query(sql, backend)
    
    conn = get_connection()
    cursor = conn.cursor()
    try:
        if params:
            cursor.execute(normalized_sql, params)
        else:
            cursor.execute(normalized_sql)
            
        if commit:
            conn.commit()
            
        last_id = getattr(cursor, "lastrowid", None)
        return last_id if last_id else cursor.rowcount
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def execute_script(sql_script):
    """Executes a multi-statement SQL script."""
    backend = get_backend()
    conn = get_connection()
    
    try:
        if backend == "sqlite":
            # Adapt MySQL dialect to SQLite
            adapted = sql_script
            adapted = re.sub(r'AUTO_INCREMENT', 'AUTOINCREMENT', adapted, flags=re.IGNORECASE)
            conn.executescript(adapted)
        else:
            cursor = conn.cursor()
            # MySQL split by semicolon
            statements = [s.strip() for s in sql_script.split(";") if s.strip()]
            for stmt in statements:
                cursor.execute(stmt)
            conn.commit()
            cursor.close()
    finally:
        conn.close()


def fetch_dataframe(sql, params=None):
    """
    Executes a SELECT query and returns the results as a Pandas DataFrame.
    """
    backend = get_backend()
    normalized_sql = normalize_query(sql, backend)
    conn = get_connection()
    try:
        if params:
            df = pd.read_sql_query(normalized_sql, conn, params=params)
        else:
            df = pd.read_sql_query(normalized_sql, conn)
        return df
    finally:
        conn.close()


def init_database():
    """Initializes tables using schema.sql."""
    schema_path = config.DATABASE_DIR / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")
        
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
        
    execute_script(schema_sql)
    print(f"[AutoVision AI] Database initialized successfully ({get_backend().upper()} mode).")


if __name__ == "__main__":
    init_database()
