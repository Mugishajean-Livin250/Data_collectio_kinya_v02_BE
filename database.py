from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# URL decode the password if needed
DATABASE_URL = "postgresql://postgress:3OX8iLdkYmSQZRd1mmXji8txhCii9cR2@dpg-d345lv3uibrs73b4hps0-a.oregon-postgres.render.com/auth_demo_gc8s"

# Add connection pooling and timeout settings
engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True,  # Validates connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
    connect_args={"connect_timeout": 10}  # 10 second timeout
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Single Base for the entire project
Base = declarative_base()

# Test database connection
def test_connection():
    try:
        conn = engine.connect()
        conn.close()
        print("✅ Database connection successful!")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

# Test on import
if __name__ == "__main__":
    test_connection()