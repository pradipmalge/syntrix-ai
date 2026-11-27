import asyncio

from sqlalchemy import text

from syntrix_ai.infrastructure.persistence.database import (
    AsyncSessionFactory,
)


async def main():
    async with AsyncSessionFactory() as session:
        result = await session.execute(
            text("""
                SELECT
                    current_database() AS database_name,
                    current_user AS username,
                    inet_server_addr() AS server_address,
                    inet_server_port() AS server_port,
                    current_setting('data_directory') AS data_directory
            """)
        )

        print("Application connection:", result.mappings().one())

        result = await session.execute(
            text("SELECT COUNT(*) FROM documents")
        )
        print("Document count:", result.scalar_one())


if __name__ == "__main__":
    asyncio.run(main())