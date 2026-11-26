from prometheus_client import Counter, Histogram
from prometheus_client import Gauge

DB_POOL_SIZE = Gauge(
    "syntrix_db_pool_size",
    "Configured SQLAlchemy connection pool size",
)

DB_POOL_CONNECTIONS_OPEN = Gauge(
    "syntrix_db_pool_connections_open",
    "Number of currently open database connections",
)

DB_POOL_CONNECTIONS_CHECKED_OUT = Gauge(
    "syntrix_db_pool_connections_checked_out",
    "Number of database connections currently checked out",
)

DB_POOL_OVERFLOW = Gauge(
    "syntrix_db_pool_overflow",
    "Number of connections beyond the configured pool size",
)

LLM_REQUESTS_TOTAL = Counter(
    "syntrix_llm_requests_total",
    "Total number of LLM requests",
    ["provider", "model", "status"],
)

LLM_REQUEST_DURATION = Histogram(
    "syntrix_llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["provider", "model"],
)

LLM_TOKEN_USAGE = Counter(
    "syntrix_llm_tokens_total",
    "Total tokens consumed by LLM requests",
    ["provider", "model", "token_type"],
)

LLM_STREAM_REQUESTS_TOTAL = Counter(
    "syntrix_llm_stream_requests_total",
    "Total number of LLM streaming requests",
    ["provider", "model", "status"],
)

LLM_STREAM_DURATION = Histogram(
    "syntrix_llm_stream_duration_seconds",
    "Total LLM streaming duration in seconds",
    ["provider", "model"],
)

LLM_TIME_TO_FIRST_TOKEN = Histogram(
    "syntrix_llm_time_to_first_token_seconds",
    "Time taken to receive the first streaming chunk",
    ["provider", "model"],
)