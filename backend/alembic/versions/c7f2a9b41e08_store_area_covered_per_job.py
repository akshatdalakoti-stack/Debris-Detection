"""store the seabed area each job searched

The API reported area_covered as a hardcoded 0.0. The pipeline now derives it
from the file's own geometry - swath from slant range and altitude, track
length from the distance between consecutive fixes - so there is a real number
to keep. Null where the file carried no navigation.

Revision ID: c7f2a9b41e08
Revises: a1c4e7b09d32
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7f2a9b41e08"
down_revision: str | Sequence[str] | None = "a1c4e7b09d32"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("area_covered_m2", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "area_covered_m2")
