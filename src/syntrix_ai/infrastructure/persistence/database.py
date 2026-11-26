
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

from syntrix_ai.core.config import get_settings

from sqlalchemy import event

from syntrix_ai.infrastructure.observability.metrics import (
    DB_POOL_SIZE,
    DB_POOL_CONNECTIONS_OPEN,
    DB_POOL_CONNECTIONS_CHECKED_OUT,
    DB_POOL_OVERFLOW,
)


settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
)

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

_pool_metrics_configured = False

def configure_database_pool_metrics() -> None:
    """Collect SQLAlchemy connection pool metrics."""

    global _pool_metrics_configured

    if _pool_metrics_configured:
        return

    _pool_metrics_configured = True

    pool = engine.sync_engine.pool

    DB_POOL_SIZE.set(pool.size())

    def update_overflow():
        DB_POOL_OVERFLOW.set(max(pool.overflow(), 0))

    @event.listens_for(pool, "connect")
    def on_connect(dbapi_connection, connection_record):
        DB_POOL_CONNECTIONS_OPEN.inc()
        update_overflow()

    @event.listens_for(pool, "close")
    def on_close(dbapi_connection, connection_record):
        DB_POOL_CONNECTIONS_OPEN.dec()
        update_overflow()

    @event.listens_for(pool, "checkout")
    def on_checkout(
        dbapi_connection,
        connection_record,
        connection_proxy,
    ):
        DB_POOL_CONNECTIONS_CHECKED_OUT.inc()
        update_overflow()

    @event.listens_for(pool, "checkin")
    def on_checkin(dbapi_connection, connection_record):
        DB_POOL_CONNECTIONS_CHECKED_OUT.dec()
        update_overflow()
    

_database_instrumented = False
def configure_database_instrumentation() -> None:
    """Enable OpenTelemetry instrumentation for SQLAlchemy."""

    global _database_instrumented

    if not settings.otel_enabled or _database_instrumented:
        return

    SQLAlchemyInstrumentor().instrument(
        engine=engine.sync_engine,
        enable_commenter=False,
    )

    _database_instrumented = True


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionFactory() as session:
        yield session


async def close_database() -> None:
    await engine.dispose()
