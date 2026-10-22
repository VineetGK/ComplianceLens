"""Initial schema: core tables, pgvector extension, append-only audit trigger.

Revision ID: 0001
Revises:
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = [
    "review_items",
    "regulation_matches",
    "coverage_results",
    "requirements",
    "chunks",
    "jobs",
    "documents",
    "projects",
    "users",
    "regulations",
    "audit_events",
]

APPEND_ONLY_FN = """
CREATE OR REPLACE FUNCTION audit_events_block_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_events is append-only (% blocked)', TG_OP
        USING ERRCODE = 'insufficient_privilege';
END;
$$ LANGUAGE plpgsql;
"""


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "audit_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor", sa.String(length=320), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_events")),
    )
    op.create_table(
        "regulations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("citation", sa.String(length=64), nullable=False),
        sa.Column("part", sa.String(length=16), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("embedding", Vector(), nullable=True),
        sa.Column("embedding_model", sa.String(length=200), nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('english', citation || ' ' || title || ' ' || text)", persisted=True
            ),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_regulations")),
        sa.UniqueConstraint("citation", name=op.f("uq_regulations_citation")),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
    )
    op.create_table(
        "projects",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=300), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("agency", sa.String(length=200), nullable=True),
        sa.Column("solicitation_number", sa.String(length=100), nullable=True),
        sa.Column("created_by", sa.UUID(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], name=op.f("fk_projects_created_by_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_projects")),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "rfp",
                "proposal",
                "attachment",
                name="document_kind",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("filename", sa.String(length=500), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("storage_path", sa.String(length=1000), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "uploaded",
                "parsed",
                "indexed",
                "failed",
                name="document_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_documents_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("project_id", "sha256", name="uq_documents_project_sha"),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=True),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "queued",
                "running",
                "succeeded",
                "failed",
                name="job_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_jobs_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("section_path", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("heading", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("bboxes", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("embedding", Vector(), nullable=True),
        sa.Column("embedding_model", sa.String(length=200), nullable=True),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('english', coalesce(heading, '') || ' ' || text)", persisted=True
            ),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
    )
    op.create_table(
        "requirements",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("req_key", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("verbatim_quote", sa.Text(), nullable=False),
        sa.Column(
            "modality",
            sa.Enum(
                "shall",
                "must",
                "will",
                "should",
                "may",
                name="modality",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "category",
            sa.Enum(
                "technical",
                "management",
                "past_performance",
                "pricing",
                "admin",
                "security",
                "reporting",
                name="category",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("section_ref", sa.String(length=200), nullable=True),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("bbox", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("grounding_score", sa.Float(), nullable=True),
        sa.Column("flags", postgresql.ARRAY(sa.String(length=32)), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "auto_accepted",
                "needs_review",
                "approved",
                "edited",
                "rejected",
                name="requirement_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("duplicate_of", sa.UUID(), nullable=True),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_requirements_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["duplicate_of"],
            ["requirements.id"],
            name=op.f("fk_requirements_duplicate_of_requirements"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_requirements_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_requirements")),
        sa.UniqueConstraint("project_id", "req_key", name="uq_requirements_project_key"),
    )
    op.create_table(
        "coverage_results",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("requirement_id", sa.UUID(), nullable=False),
        sa.Column("proposal_document_id", sa.UUID(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "addressed",
                "partially_addressed",
                "not_addressed",
                name="coverage_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("evidence_quote", sa.Text(), nullable=True),
        sa.Column("evidence_page", sa.Integer(), nullable=True),
        sa.Column("evidence_grounded", sa.Boolean(), nullable=True),
        sa.Column("suggested_fix", sa.Text(), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["proposal_document_id"],
            ["documents.id"],
            name=op.f("fk_coverage_results_proposal_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["requirement_id"],
            ["requirements.id"],
            name=op.f("fk_coverage_results_requirement_id_requirements"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_coverage_results")),
    )
    op.create_table(
        "regulation_matches",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("requirement_id", sa.UUID(), nullable=False),
        sa.Column("regulation_id", sa.UUID(), nullable=False),
        sa.Column("relevance_rationale", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["regulation_id"],
            ["regulations.id"],
            name=op.f("fk_regulation_matches_regulation_id_regulations"),
        ),
        sa.ForeignKeyConstraint(
            ["requirement_id"],
            ["requirements.id"],
            name=op.f("fk_regulation_matches_requirement_id_requirements"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_regulation_matches")),
    )
    op.create_table(
        "review_items",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("requirement_id", sa.UUID(), nullable=False),
        sa.Column("reasons", postgresql.ARRAY(sa.String(length=64)), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "approved",
                "edited",
                "rejected",
                name="review_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("reviewer_id", sa.UUID(), nullable=True),
        sa.Column("decision_reason", sa.Text(), nullable=True),
        sa.Column("before", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("after", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_review_items_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["requirement_id"],
            ["requirements.id"],
            name=op.f("fk_review_items_requirement_id_requirements"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["reviewer_id"], ["users.id"], name=op.f("fk_review_items_reviewer_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_review_items")),
    )
    op.create_index(
        "ix_audit_events_project_created",
        "audit_events",
        ["project_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_regulations_tsv", "regulations", ["tsv"], unique=False, postgresql_using="gin"
    )
    op.create_index(op.f("ix_regulations_part"), "regulations", ["part"], unique=False)
    op.create_index(
        op.f("ix_regulations_embedding_model"), "regulations", ["embedding_model"], unique=False
    )
    op.create_index(op.f("ix_documents_project_id"), "documents", ["project_id"], unique=False)
    op.create_index(op.f("ix_jobs_project_id"), "jobs", ["project_id"], unique=False)
    op.create_index(op.f("ix_chunks_document_id"), "chunks", ["document_id"], unique=False)
    op.create_index(op.f("ix_chunks_embedding_model"), "chunks", ["embedding_model"], unique=False)
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], unique=False, postgresql_using="gin")
    op.create_index(
        op.f("ix_requirements_project_id"), "requirements", ["project_id"], unique=False
    )
    op.create_index(
        op.f("ix_coverage_results_requirement_id"),
        "coverage_results",
        ["requirement_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_regulation_matches_requirement_id"),
        "regulation_matches",
        ["requirement_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_review_items_requirement_id"), "review_items", ["requirement_id"], unique=False
    )
    op.create_index(op.f("ix_review_items_status"), "review_items", ["status"], unique=False)
    op.create_index(
        op.f("ix_review_items_project_id"), "review_items", ["project_id"], unique=False
    )

    # Append-only audit log: block UPDATE/DELETE per row and TRUNCATE per statement.
    op.execute(APPEND_ONLY_FN)
    op.execute(
        "CREATE TRIGGER audit_events_no_update_delete BEFORE UPDATE OR DELETE ON audit_events "
        "FOR EACH ROW EXECUTE FUNCTION audit_events_block_mutation()"
    )
    op.execute(
        "CREATE TRIGGER audit_events_no_truncate BEFORE TRUNCATE ON audit_events "
        "FOR EACH STATEMENT EXECUTE FUNCTION audit_events_block_mutation()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS audit_events_no_truncate ON audit_events")
    op.execute("DROP TRIGGER IF EXISTS audit_events_no_update_delete ON audit_events")
    op.execute("DROP FUNCTION IF EXISTS audit_events_block_mutation()")
    for table in TABLES:
        op.drop_table(table)
