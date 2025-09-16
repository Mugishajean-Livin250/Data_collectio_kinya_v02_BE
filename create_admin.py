from passlib.context import CryptContext
from sqlalchemy.orm import Session
from database import SessionLocal
from models import User

# Initialize password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_admin_user():
    # Admin credentials - CHANGE THESE!
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "admin123"  # Change this to a secure password
    
    # Hash the password
    hashed_password = pwd_context.hash(ADMIN_PASSWORD)
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Check if admin already exists
        existing_admin = db.query(User).filter(User.username == ADMIN_USERNAME).first()
        if existing_admin:
            print(f"Admin user '{ADMIN_USERNAME}' already exists!")
            return
        
        # Create admin user
        admin_user = User(
            username=ADMIN_USERNAME,
            password_hash=hashed_password,
            role="admin"
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        print(f"Admin user created successfully!")
        print(f"Username: {ADMIN_USERNAME}")
        print(f"Password: {ADMIN_PASSWORD}")
        print("Please change the password after first login for security!")
        
    except Exception as e:
        print(f"Error creating admin user: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_admin_user()