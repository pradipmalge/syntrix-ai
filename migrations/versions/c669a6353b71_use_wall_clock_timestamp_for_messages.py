"""use wall clock timestamp for messages

Revision ID: c669a6353b71
Revises: 48c701a1b343
Create Date: 2026-11-24 11:43:02.609458

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c669a6353b71'
down_revision: Union[str, Sequence[str], None] = '48c701a1b343'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "messages",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        server_default=sa.text("clock_timestamp()"),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "messages",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        server_default=sa.text("now()"),
        existing_nullable=False,
    )
