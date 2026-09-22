"""identify uploaded files by a content hash

Re-uploading the same file to a survey stored a second copy and ran the whole
pipeline again, producing a duplicate set of detections that reached the
registry as if it were a second independent sighting.

Revision ID: e5a01d9c26f4
Revises: d3b8c15a7e91
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5a01d9c26f4"
down_revision: str | Sequence[str] | None = "d3b8c15a7e91"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable: rows written before this have no digest, and there is nothing
    # to backfill them from without re-reading every stored file.
    op.add_column("files", sa.Column("content_sha256", sa.String(length=64),
                                     nullable=True))
    op.create_index("ix_files_survey_content", "files",
                    ["survey_id", "content_sha256"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_files_survey_content", table_name="files")
    op.drop_column("files", "content_sha256")
