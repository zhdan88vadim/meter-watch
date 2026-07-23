from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from meter_watch_shared.config import config

DATABASE_URL = config.DATABASE_URL
DATABASE_URL_ASYNC = config.DATABASE_URL_ASYNC

print("DATABASE_URL", DATABASE_URL)
print("DATABASE_URL_ASYNC", DATABASE_URL_ASYNC)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()