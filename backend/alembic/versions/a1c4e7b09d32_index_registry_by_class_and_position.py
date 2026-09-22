"""index registry entries by class and position

Reconciliation looks for "an entry of this class within 25 m of this point"
once per detection, which was a full scan of registry_entries every time. The
service now narrows with a lat/lon bounding box first; this is the index that
makes that narrowing cheap.

Revision ID: a1c4e7b09d32
Revises: fb6720e37081
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a1c4e7b09d32"
down_revision: str | Sequence[str] | None = "fb6720e37081"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_registry_entries_class_lat_lon",
        "registry_entries",
        ["class", "lat", "lon"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_registry_entries_class_lat_lon", table_name="registry_entries")
