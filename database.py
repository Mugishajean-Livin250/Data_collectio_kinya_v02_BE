from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import Base  # Use the Base from models.py

DATABASE_URL = "postgresql://postgres:mugisha%40250@localhost/auth_demo"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"connect_timeout": 10}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# No need to redefine Base here, use models.Base
