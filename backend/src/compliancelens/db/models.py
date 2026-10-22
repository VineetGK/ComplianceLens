"""ORM models. The initial Alembic migration mirrors these definitions.

Vectors are stored in an untyped ``vector`` column next to ``embedding_model``;
per-model partial HNSW indexes (phase 2) cast to the model's dimension, so vectors
from different embedding models can never be compared with each other.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from compliancelens.db.base import Base


def _enum(enum_cls: type[StrEnum], name: str) -> SAEnum:
    """VARCHAR + CHECK enum storing the string values (easy to extend in migrations)."""
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda e: [m.value for m in e],
    )


class DocumentKind(StrEnum):
    RFP = "rfp"
    PROPOSAL = "proposal"
    ATTACHMENT = "attachment"


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PARSED = "parsed"
    INDEXED = "indexed"
    FAILED = "failed"


class Modality(StrEnum):
    SHALL = "shall"
    MUST = "must"
    WILL = "will"
    SHOULD = "should"
    MAY = "may"


class Category(StrEnum):
    TECHNICAL = "technical"
    MANAGEMENT = "management"
    PAST_PERFORMANCE = "past_performance"
    PRICING = "pricing"
    ADMIN = "admin"
    SECURITY = "security"
    REPORTING = "reporting"


class RequirementStatus(StrEnum):
    AUTO_ACCEPTED = "auto_accepted"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class CoverageStatus(StrEnum):
    ADDRESSED = "addressed"
    PARTIALLY_ADDRESSED = "partially_addressed"
    NOT_ADDRESSED = "not_addressed"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="reviewer")
    created_at: Mapped[datetime] = _created_at()


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    agency: Mapped[str | None] = mapped_column(String(200))
    solicitation_number: Mapped[str | None] = mapped_column(String(100))
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = _created_at()

    documents: Mapped[list[Document]] = relationship(back_populates="project")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("project_id", "sha256", name="uq_documents_project_sha"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[DocumentKind] = mapped_column(_enum(DocumentKind, "document_kind"), nullable=False)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[DocumentStatus] = mapped_column(
        _enum(DocumentStatus, "document_status"), nullable=False, default=DocumentStatus.UPLOADED
    )
    created_at: Mapped[datetime] = _created_at()

    project: Mapped[Project] = relationship(back_populates="documents")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    heading: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    # [{"page": 3, "bbox": [x0, y0, x1, y1]}, ...] in PDF points
    bboxes: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    embedding: Mapped[list[float] | None] = mapped_column(Vector())
    embedding_model: Mapped[str | None] = mapped_column(String(200), index=True)
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(heading, '') || ' ' || text)", persisted=True),
    )


class Regulation(Base):
    __tablename__ = "regulations"
    __table_args__ = (Index("ix_regulations_tsv", "tsv", postgresql_using="gin"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    source: Mapped[str] = mapped_column(String(32), nullable=False)  # "FAR" | "2CFR200"
    citation: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    part: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    effective_date: Mapped[date | None] = mapped_column(Date)
    embedding: Mapped[list[float] | None] = mapped_column(Vector())
    embedding_model: Mapped[str | None] = mapped_column(String(200), index=True)
    fetched_at: Mapped[datetime] = _created_at()
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', citation || ' ' || title || ' ' || text)", persisted=True),
    )


class Requirement(Base):
    __tablename__ = "requirements"
    __table_args__ = (
        UniqueConstraint("project_id", "req_key", name="uq_requirements_project_key"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    req_key: Mapped[str] = mapped_column(String(32), nullable=False)  # e.g. "REQ-0042"
    text: Mapped[str] = mapped_column(Text, nullable=False)
    verbatim_quote: Mapped[str] = mapped_column(Text, nullable=False)
    modality: Mapped[Modality] = mapped_column(_enum(Modality, "modality"), nullable=False)
    category: Mapped[Category] = mapped_column(_enum(Category, "category"), nullable=False)
    section_ref: Mapped[str | None] = mapped_column(String(200))
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox: Mapped[list[float] | None] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    grounding_score: Mapped[float | None] = mapped_column(Float)
    flags: Mapped[list[str]] = mapped_column(ARRAY(String(32)), nullable=False, default=list)
    status: Mapped[RequirementStatus] = mapped_column(
        _enum(RequirementStatus, "requirement_status"),
        nullable=False,
        default=RequirementStatus.NEEDS_REVIEW,
    )
    duplicate_of: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("requirements.id"))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _created_at()


class RegulationMatch(Base):
    __tablename__ = "regulation_matches"

    id: Mapped[uuid.UUID] = _uuid_pk()
    requirement_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("requirements.id", ondelete="CASCADE"), nullable=False, index=True
    )
    regulation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("regulations.id"), nullable=False)
    relevance_rationale: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class CoverageResult(Base):
    __tablename__ = "coverage_results"

    id: Mapped[uuid.UUID] = _uuid_pk()
    requirement_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("requirements.id", ondelete="CASCADE"), nullable=False, index=True
    )
    proposal_document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[CoverageStatus] = mapped_column(
        _enum(CoverageStatus, "coverage_status"), nullable=False
    )
    evidence_quote: Mapped[str | None] = mapped_column(Text)
    evidence_page: Mapped[int | None] = mapped_column(Integer)
    evidence_grounded: Mapped[bool | None] = mapped_column()
    suggested_fix: Mapped[str | None] = mapped_column(Text)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class ReviewItem(Base):
    __tablename__ = "review_items"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    requirement_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("requirements.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reasons: Mapped[list[str]] = mapped_column(ARRAY(String(64)), nullable=False)
    status: Mapped[ReviewStatus] = mapped_column(
        _enum(ReviewStatus, "review_status"),
        nullable=False,
        default=ReviewStatus.PENDING,
        index=True,
    )
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    decision_reason: Mapped[str | None] = mapped_column(Text)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created_at()
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    """Append-only: a trigger rejects UPDATE and DELETE (see migration 0001)."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_project_created", "project_id", "created_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    event_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # llm_call|retrieval|decision
    actor: Mapped[str] = mapped_column(String(320), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        _enum(JobStatus, "job_status"), nullable=False, default=JobStatus.QUEUED
    )
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
