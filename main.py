from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import List, Dict, Any, Optional, Union
from nl2sql import NL2SQLConverter
import os
from urllib.parse import urlparse
import time
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced PostgreSQL Query API with Auto-AI",
    version="3.0.0",
    description="AI-powered natural language to SQL converter with enhanced features"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
nl2sql_converter = None
connection_pool: Dict[str, psycopg2.extensions.connection] = {}
last_activity: Dict[str, float] = {}
executor = ThreadPoolExecutor(max_workers=4)

# ——— Pydantic Models ——————————————————————————————————————————————

class DatabaseConfig(BaseModel):
    host: str
    port: int = 5432
    database: str
    username: str
    password: str

class DatabaseURL(BaseModel):
    url: str

    @validator('url')
    def validate_postgresql_url(cls, v):
        if not v.startswith(('postgresql://', 'postgres://')):
            raise ValueError('URL must start with postgresql:// or postgres://')
        return v

class ConnectionRequest(BaseModel):
    connection_type: str  # "credentials" or "url"
    credentials: Optional[DatabaseConfig] = None
    url_connection: Optional[DatabaseURL] = None

class NaturalQueryRequest(BaseModel):
    natural_query: str
    connection_request: ConnectionRequest

class QueryExecutionRequest(BaseModel):
    sql_query: str
    connection_request: ConnectionRequest

class EnhancedConnectionResponse(BaseModel):
    status: str
    message: str
    schema_data: Optional[List[Dict[str, Any]]] = None
    connection_info: Optional[Dict[str, Any]] = None
    sample_queries: Optional[List[str]] = None

class ColumnMetadata(BaseModel):
    type_code: str
    display_size: Optional[int]
    internal_size: Optional[int]

class EnhancedQueryResponse(BaseModel):
    status: str
    message: str
    sql_query: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    explanation: Optional[str] = None
    visualization_suggestions: Optional[List[Dict[str, Any]]] = None
    column_info: Optional[Dict[str, ColumnMetadata]] = None
    execution_time: Optional[float] = None
    from_cache: Optional[bool] = None

# ——— Startup & Shutdown ————————————————————————————————————————

def initialize_ai_on_startup():
    global nl2sql_converter
    try:
        nl2sql_converter = NL2SQLConverter()
        logger.info("✅ AI Converter initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ AI initialization failed: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    logger.info("Starting API...")
    initialize_ai_on_startup()
    asyncio.create_task(cleanup_inactive_connections())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down")
    cleanup_all_connections()
    executor.shutdown(wait=True)

# ——— Connection Cleanup ————————————————————————————————————————

async def cleanup_inactive_connections():
    while True:
        now = time.time()
        for key, last in list(last_activity.items()):
            if now - last > 300:
                conn = connection_pool.get(key)
                if conn:
                    conn.close()
                connection_pool.pop(key, None)
                last_activity.pop(key, None)
                logger.info(f"🧹 Cleaned expired connection: {key}")
        await asyncio.sleep(60)

def cleanup_all_connections():
    for key, conn in connection_pool.items():
        try:
            conn.close()
            logger.info(f"🧹 Closed connection: {key}")
        except:
            pass
    connection_pool.clear()
    last_activity.clear()

# ——— Utility Functions —————————————————————————————————————————

def parse_database_url(url: str) -> DatabaseConfig:
    try:
        p = urlparse(url)
        if p.scheme not in ['postgresql', 'postgres']:
            raise ValueError("Invalid URL scheme")
        return DatabaseConfig(
            host=p.hostname or 'localhost',
            port=p.port or 5432,
            database=(p.path or '').lstrip('/'),
            username=p.username or '',
            password=p.password or ''
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid URL: {e}")

def get_connection_key(cfg: DatabaseConfig) -> str:
    return f"{cfg.username}@{cfg.host}:{cfg.port}/{cfg.database}"

def create_or_get_connection(cr: ConnectionRequest):
    try:
        if cr.connection_type == "url":
            cfg = parse_database_url(cr.url_connection.url)
        elif cr.connection_type == "credentials":
            cfg = cr.credentials
        else:
            raise HTTPException(status_code=400, detail="Invalid connection_type")
        if not cfg.database or not cfg.username:
            raise HTTPException(status_code=400, detail="DB name and user required")

        key = get_connection_key(cfg)
        if key in connection_pool:
            try:
                cur = connection_pool[key].cursor()
                cur.execute("SELECT 1")
                cur.close()
                last_activity[key] = time.time()
                logger.info(f"Reusing connection {key}")
                return connection_pool[key], cfg
            except:
                connection_pool[key].close()
                connection_pool.pop(key, None)
                last_activity.pop(key, None)

        conn = psycopg2.connect(
            host=cfg.host, port=cfg.port,
            database=cfg.database,
            user=cfg.username, password=cfg.password,
            connect_timeout=15,
            application_name="AI_SQL_Query_App"
        )
        cur = conn.cursor()
        cur.execute("SELECT version()")
        ver = cur.fetchone()[0]
        cur.close()
        connection_pool[key] = conn
        last_activity[key] = time.time()
        logger.info(f"✅ New connection {key}, Postgres: {ver}")
        return conn, cfg

    except psycopg2.OperationalError as e:
        msg = str(e)
        if "authentication failed" in msg:
            raise HTTPException(status_code=401, detail="Authentication failed")
        elif "does not exist" in msg:
            raise HTTPException(status_code=404, detail="Database not found")
        elif "could not connect to server" in msg:
            raise HTTPException(status_code=503, detail="Cannot connect to server")
        else:
            raise HTTPException(status_code=400, detail=f"Connection error: {msg}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Connection unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

def get_enhanced_database_schema(conn) -> tuple:
    """Fetch comprehensive schema information with enhanced structure and summary."""
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Fetch all public base table names
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        schema_data = []
        total_columns = 0

        for tbl in tables:
            table_name = tbl['table_name']

            # Fetch columns info for each table
            cursor.execute("""
                SELECT column_name,
                       data_type,
                       is_nullable,
                       column_default,
                       character_maximum_length,
                       numeric_precision,
                       numeric_scale
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                ORDER BY ordinal_position;
            """, (table_name,))

            columns = cursor.fetchall()
            total_columns += len(columns)

            # Organize and include in schema_data
            schema_data.append({
                'table_name': table_name,
                'columns': columns  # raw dicts from RealDictCursor
            })

        summary = {
            'total_tables': len(tables),
            'total_columns': total_columns
        }

        cursor.close()
        return schema_data, summary

    except Exception as e:
        logger.error(f"Error fetching enhanced schema: {e}")
        # Return empty but valid structure instead of None
        return [], {'total_tables': 0, 'total_columns': 0}

    ...

def execute_sql_with_timing(connection, sql_query: str) -> tuple:
    start = time.time()
    try:
        cur = connection.cursor(cursor_factory=RealDictCursor)
        cur.execute("SET statement_timeout = '30s'")
        cur.execute(sql_query)
        rows = cur.fetchall()

        column_info: Dict[str, Dict[str, Any]] = {}
        if cur.description:
            for desc in cur.description:
                column_info[desc.name] = {
                    'type_code': str(desc.type_code),
                    'display_size': desc.display_size,
                    'internal_size': desc.internal_size
                }

        cur.close()
        exec_time = time.time() - start
        logger.info(f"Query executed in {exec_time:.3f}s, rows: {len(rows)}")
        return rows, column_info, exec_time

    except psycopg2.errors.QueryCanceled:
        raise HTTPException(status_code=408, detail="Query timeout (>30s)")
    except psycopg2.errors.SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"SQL Syntax Error: {e}")
    except psycopg2.errors.UndefinedTable as e:
        raise HTTPException(status_code=404, detail=f"Table Not Found: {e}")
    except psycopg2.errors.UndefinedColumn as e:
        raise HTTPException(status_code=404, detail=f"Column Not Found: {e}")
    except psycopg2.Error as e:
        raise HTTPException(status_code=400, detail=f"DB Error [{e.pgcode}]: {e}")
    except Exception as e:
        logger.error(f"Query unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

# ——— API Endpoints —————————————————————————————————————————————

@app.post("/connect", response_model=EnhancedConnectionResponse)
async def connect_database(request: ConnectionRequest):
    conn, cfg = create_or_get_connection(request)
    schema_data, summary = get_enhanced_database_schema(conn)
    examples = (
        nl2sql_converter.get_sample_queries_from_schema(schema_data)
        if nl2sql_converter else [
            "Show all rows from the largest table",
            "Count rows per table",
            "List column names"
        ]
    )
    info = {
        "host": cfg.host, "port": cfg.port,
        "database": cfg.database, "username": cfg.username,
        "connection_type": request.connection_type,
        "schema_summary": summary
    }
    return EnhancedConnectionResponse(
        status="success",
        message=f"Connected to {cfg.database}: {summary['total_tables']} tables, {summary['total_columns']} cols",
        schema_data=schema_data,
        connection_info=info,
        sample_queries=examples
    )

@app.post("/natural-query", response_model=EnhancedQueryResponse)
async def process_enhanced_natural_query(request: NaturalQueryRequest):
    if not nl2sql_converter:
        try:
            initialize_ai_on_startup()
        except:
            raise HTTPException(status_code=503, detail="AI unavailable")

    conn, _ = create_or_get_connection(request.connection_request)
    schema_data, _ = get_enhanced_database_schema(conn)

    conv = nl2sql_converter.convert_to_sql(request.natural_query, schema_data)
    if not conv.get('success', False):
        return EnhancedQueryResponse(
            status="error",
            message=conv.get('error'),
            sql_query=conv.get('sql_query'),
            from_cache=conv.get('from_cache', False),
        )

    sql_q = conv['sql_query']
    try:
        rows, col_info, qt = execute_sql_with_timing(conn, sql_q)
    except HTTPException as e:
        return EnhancedQueryResponse(status="error", message=e.detail, sql_query=sql_q)

    explanation = nl2sql_converter.explain_query_results(
        request.natural_query, sql_q, rows, col_info
    )
    viz = nl2sql_converter.suggest_visualization_options(rows, request.natural_query)
    total_time = time.time() - (conv.get('start_time', time.time()))

    return EnhancedQueryResponse(
        status="success",
        message=f"Query executed in {qt:.3f}s, {len(rows)} rows.",
        sql_query=sql_q,
        results=rows,
        explanation=explanation,
        visualization_suggestions=viz,
        column_info=col_info,
        execution_time=total_time,
        from_cache=conv.get('from_cache', False)
    )

@app.post("/execute-sql", response_model=EnhancedQueryResponse)
async def execute_direct_sql(request: QueryExecutionRequest):
    if nl2sql_converter:
        valid, err = nl2sql_converter._validate_sql_query(request.sql_query)
        if not valid:
            raise HTTPException(status_code=400, detail=err)

    conn, _ = create_or_get_connection(request.connection_request)
    rows, col_info, qt = execute_sql_with_timing(conn, request.sql_query)

    viz = nl2sql_converter.suggest_visualization_options(rows, "Direct SQL")

    return EnhancedQueryResponse(
        status="success",
        message=f"SQL executed in {qt:.3f}s, {len(rows)} rows.",
        sql_query=request.sql_query,
        results=rows,
        explanation=f"Direct SQL execution.",
        visualization_suggestions=viz,
        column_info=col_info,
        execution_time=qt,
        from_cache=False
    )

@app.get("/health")
async def enhanced_health_check():
    ai_stat = "healthy" if nl2sql_converter else "uninitialized"
    active = len(connection_pool)
    ai_test = None
    if nl2sql_converter:
        try:
            nl2sql_converter.get_connection_test_query()
            ai_test = "passed"
        except:
            ai_test = "failed"
    return {
        "status": "healthy",
        "ai_status": ai_stat,
        "ai_test": ai_test,
        "active_connections": active,
        "cache_size": len(getattr(nl2sql_converter, "query_cache", {}))
    }

@app.post("/test-connection")
async def test_database_connection(request: ConnectionRequest):
    conn, cfg = create_or_get_connection(request)
    cur = conn.cursor()
    cur.execute("SELECT 1 as test, CURRENT_TIMESTAMP as now")
    row = cur.fetchone()
    cur.close()
    return {
        "status": "success",
        "message": f"Connection to {cfg.database} successful",
        "test_result": row[0],
        "server_time": row[1].isoformat()
    }

@app.get("/")
async def root():
    ai_stat = "initialized" if nl2sql_converter else "not initialized"
    active = len(connection_pool)
    return {
        "message": "API is running",
        "status": "healthy",
        "ai_status": ai_stat,
        "active_connections": active,
        "version": "3.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
