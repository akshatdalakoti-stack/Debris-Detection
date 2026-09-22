"""link each detection to the registry hazard it was folded into

Reconciliation already works out which hazard a detection belongs to and used
to discard it, so there was no way to get from a registry entry back to the
surveys and boxes that produced it.

Revision ID: d3b8c15a7e91
Revises: c7f2a9b41e08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d3b8c15a7e91"
down_revision: str | Sequence[str] | None = "c7f2a9b41e08"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # batch_alter_table so SQLite gets the foreign key too; it cannot ADD
    # CONSTRAINT and has to rebuild the table instead.
    with op.batch_alter_table("detections", schema=None) as batch_op:
        batch_op.add_column(sa.Column("registry_entry_id", sa.Integer(), nullable=True))
        batch_op.create_index("ix_detections_registry_entry_id",
                              ["registry_entry_id"], unique=False)
        batch_op.create_foreign_key("fk_detections_registry_entry",
                                    "registry_entries", ["registry_entry_id"], ["id"])


def downgrade() -> None:
    with op.batch_alter_table("detections", schema=None) as batch_op:
        batch_op.drop_constraint("fk_detections_registry_entry", type_="foreignkey")
        batch_op.drop_index("ix_detections_registry_entry_id")
        batch_op.drop_column("registry_entry_id")
