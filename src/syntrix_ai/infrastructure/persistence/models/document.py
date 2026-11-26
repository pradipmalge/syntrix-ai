
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from syntrix_ai.infrastructure.persistence.base import Base


class DocumentRecord(Base):

    __tablename__ = "documents"

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'indexed', 'failed', 'deleted')",
            name="ck_documents_status",
        ),
        Index(
            "ix_documents_tenant_knowledge_base",
            "tenant_id",
            "knowledge_base_id",
        ),
        Index(
            "ix_documents_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_documents_created_at",
            "created_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    knowledge_base_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    content_type: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    source_uri: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        server_default="pending",
    )

    document_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.clock_timestamp(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.clock_timestamp(),
        onupdate=func.clock_timestamp(),
    )
