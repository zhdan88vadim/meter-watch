"""add is_anomaly field

Revision ID: b4aa748a704c
Revises: bd12d9e991df
Create Date: 2026-07-23 14:42:25.748165

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b4aa748a704c'
down_revision: Union[str, Sequence[str], None] = 'bd12d9e991df'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create enum types first - using uppercase to match your enum values
    source_enum = postgresql.ENUM('METER', 'PERSON_DETECTOR', name='sourceenum', create_type=True)
    event_enum = postgresql.ENUM('READING', 'PERSON_DETECTED', 'PERSON_LEFT', name='eventtypeenum', create_type=True)
    
    # Create the enum types in the database
    source_enum.create(op.get_bind(), checkfirst=True)
    event_enum.create(op.get_bind(), checkfirst=True)
    
    # Now alter columns to use the enum types
    op.alter_column('activity_logs', 'source',
               existing_type=sa.VARCHAR(length=50),
               type_=source_enum,
               nullable=False,
               postgresql_using="source::text::sourceenum")
    
    op.alter_column('activity_logs', 'event_type',
               existing_type=sa.VARCHAR(length=50),
               type_=event_enum,
               nullable=False,
               postgresql_using="event_type::text::eventtypeenum")
    
    # Drop the person_detected column if it exists
    op.drop_column('activity_logs', 'person_detected')
    
    # Add the is_anomaly column
    op.add_column('meter_readings', sa.Column('is_anomaly', sa.Boolean(), nullable=True))
    
    # Ensure value column is not nullable
    op.alter_column('meter_readings', 'value',
               existing_type=sa.DOUBLE_PRECISION(precision=53),
               nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Revert value column to nullable
    op.alter_column('meter_readings', 'value',
               existing_type=sa.DOUBLE_PRECISION(precision=53),
               nullable=True)
    
    # Drop the is_anomaly column
    op.drop_column('meter_readings', 'is_anomaly')
    
    # Add back person_detected column
    op.add_column('activity_logs', sa.Column('person_detected', sa.BOOLEAN(), autoincrement=False, nullable=True))
    
    # Change enum columns back to VARCHAR
    op.alter_column('activity_logs', 'event_type',
               existing_type=sa.Enum('READING', 'PERSON_DETECTED', 'PERSON_LEFT', name='eventtypeenum'),
               type_=sa.VARCHAR(length=50),
               nullable=True)
    
    op.alter_column('activity_logs', 'source',
               existing_type=sa.Enum('METER', 'PERSON_DETECTOR', name='sourceenum'),
               type_=sa.VARCHAR(length=50),
               nullable=True)
    
    # Drop the enum types
    op.execute("DROP TYPE IF EXISTS eventtypeenum")
    op.execute("DROP TYPE IF EXISTS sourceenum")