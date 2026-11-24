
import pytest
from sqlalchemy import text

from syntrix_ai.infrastructure.persistence.database import (
    AsyncSessionFactory,
)


@pytest.mark.asyncio
async def test_database_connection():

    async with AsyncSessionFactory() as session:
        result = await session.execute(text("SELECT 1"))

        assert result.scalar_one() == 1
