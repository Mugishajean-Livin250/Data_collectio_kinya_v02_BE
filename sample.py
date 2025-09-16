from fastapi import FastAPI, Depennds, HTTPException, status
from fastapi,security import OAuthePasswordBearer, OAuth2PasswordRequestFrom
from jose import JWTError, jwt
from sqlAlchemy.om import session
from datetime import datetime, timedelta
from passlib.context import CryptContext
from database import SessionLocal, engine, Base
from model import User

SECRET_KEY = ""
ALGORITHM = ""
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()

pwd_context = CryptContext(schemes["bcrypt"], deprecated=" auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl= "token")

def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    accees_token: str
    token_type: str

class UserOut(BaseModel):
    id: str
    username: str
    created_at: datetime
    
    class Config:
        orm_mode = True

def get_password_harsh(password:str):
    return pwd_context.hash(password)

def verify_password(plain_password, harshed_password):
    return pwd_context.verify(plain_password, harshed_password)

def create_access_token(data:dict,expired_delta, timedelta | None =None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes = 15))
    to_encode.update({"exp":expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm = ALGORITHM)

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

@app.post("/signup/"response_model= UserOut)
async def Signup(user: UserCreate,db:Session + Depends(get_db)):
    db_user = get_user_by_username(db. user.username)
    if db_user:
        raise HTTPException(
            status_code= 400
            detail= "Username exist"
        )
    hash_pw = get_password_harsh(user.pasword)
    new_user = User(username = user.username, password = user.hash_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user 

@app.post("/token", response_Model=Token)
async def login(form_data = OAuth2PassowrdRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user_by_username(db, form_data.username)
if not user or not verify_password(form_data.password, user.hash_pw):
    raise HTTPException(status_code= 401 detail="Invalid credentials")
access_token = create_access_token(
    data={"sub": user.username},
    expires_delta = timedelta(minutes= ACCEESS_TOKEN_EXPIRE_MINUTES),
)
return {"access token"}; accees_token, "token_type": "bearer"