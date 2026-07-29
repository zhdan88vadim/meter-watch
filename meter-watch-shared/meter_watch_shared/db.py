# db.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from urllib.parse import urlparse
from meter_watch_shared.config import config
from alembic.config import Config as alembic_config
from alembic import command

DATABASE_URL = config.DATABASE_URL
DATABASE_URL_ASYNC = config.DATABASE_URL_ASYNC

print("DATABASE_URL", DATABASE_URL)
print("DATABASE_URL_ASYNC", DATABASE_URL_ASYNC)

def create_database_if_not_exists():
    """Create PostgreSQL database if it doesn't exist"""
    try:
        # Parse the database URL
        parsed = urlparse(DATABASE_URL_ASYNC)
        db_name = parsed.path[1:]  # Remove leading '/'
        
        # Connect to default 'postgres' database
        # Your URL: postgresql://tracker_user:pg_secure_password_here@postgres:5432/person_tracker
        default_db_url = f"{parsed.scheme}://{parsed.netloc}/postgres"
        
        # Create engine for default database with autocommit
        default_engine = create_engine(default_db_url, isolation_level="AUTOCOMMIT")
        
        with default_engine.connect() as conn:
            # Check if target database exists
            result = conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
            )
            exists = result.scalar() is not None
            
            if not exists:
                # Create the database
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                print(f"✅ Database '{db_name}' created successfully")
            else:
                print(f"ℹ️ Database '{db_name}' already exists")
                
    except Exception as e:
        print(f"⚠️ Error creating database: {e}")
        # Continue anyway - database might already exist or be created by migrations

def run_migrations():
    """Run Alembic migrations"""
    try:       
        alembic_cfg = alembic_config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        print("✅ Migrations applied successfully")
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise

def init_database():
    """Initialize database on first start"""
    print("🔄 Initializing database...")
    create_database_if_not_exists()
    run_migrations()
    print("✅ Database initialization complete")

# Create engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()