import os
import shutil
import json
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_silence
import pyloudnorm as pyln
import whisper
import asyncio

from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional,List
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator

# Database imports
from database import engine, SessionLocal
from models import User, Audio, AudioStatus, Base

# Setup
Base.metadata.create_all(bind=engine)
app = FastAPI(title="Audio Processing Server", version="1.0.0")
app.mount("/static", StaticFiles(directory="uploads"), name="static")

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173",  # React dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
UPLOAD_DIR = "uploads/audios"
PROCESSED_DIR = "uploads/processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

MIN_DURATION, MAX_DURATION = 15, 25
MAX_SILENCE_DURATION = 2
TARGET_LUFS, MIN_LUFS, MAX_LUFS = -23.0, -40.0, -6.0

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)
def hash_password(password): return pwd_context.hash(password)

def authenticate_user(db, username, password):
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.password_hash):
        return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    
    if expires_delta is not None:  # only set exp if expiry is given
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    except jwt.JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return user

class AudioValidator:
    @staticmethod
    def validate_audio_quality(path: str):
        try:
            audio, sr = librosa.load(path, sr=None)
            duration = len(audio) / sr
            segment = AudioSegment.from_file(path)
            silence_ranges = detect_silence(segment, min_silence_len=500, silence_thresh=segment.dBFS - 16)
            total_silence = sum(end - start for start, end in silence_ranges) / 1000
            max_silence = max([(end - start) / 1000 for start, end in silence_ranges], default=0)
            rms_energy = np.sqrt(np.mean(audio ** 2))
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            clipping_pct = (np.abs(audio) > 0.95).mean() * 100

            valid = True
            issues = []
            if duration < MIN_DURATION: 
                valid = False
                issues.append("Audio too short")
            elif duration > MAX_DURATION: 
                valid = False
                issues.append("Audio too long")
            if max_silence > MAX_SILENCE_DURATION: 
                valid = False
                issues.append("Silence gap too long")
            if rms_energy < 0.001: 
                valid = False
                issues.append("Audio too quiet")
            if loudness < MIN_LUFS or loudness > MAX_LUFS: 
                valid = False
                issues.append("Audio loudness out of range")
            if clipping_pct > 1.0: 
                valid = False
                issues.append("Audio has clipping")

            metrics = {
                "duration": float(duration),
                "total_silence": float(total_silence),
                "max_silence": float(max_silence),
                "RMS_energy": float(rms_energy),
                "lufs": float(loudness),
                "clipping_percentage": float(clipping_pct),
            }


            return {"valid": valid, "issues": issues, "metrics": metrics}
        except Exception as e:
            print(f"Audio validation error: {e}")
            return {"valid": False, "issues": [f"Validation error: {str(e)}"], "metrics": {}}

    @staticmethod
    def convert_to_wav(input_path: str, output_path: str) -> bool:
        try:
            audio, _ = librosa.load(input_path, sr=16000)
            loudness = pyln.Meter(16000).integrated_loudness(audio)
            if not np.isnan(loudness) and abs(loudness - TARGET_LUFS) > 1.0:
                audio = pyln.normalize.loudness(audio, loudness, TARGET_LUFS)
            sf.write(output_path, audio, 16000, subtype='PCM_16')
            return True
        except Exception as e:
            print(f"WAV conversion error: {e}")
            return False

class AITranscriptionService:
    model = whisper.load_model("small")

    @staticmethod
    async def transcribe_and_translate(file_path: str):
        """Transcribe Kinyarwanda audio locally using Whisper, then translate to English"""
        import functools  # Ensure functools is imported

        try:
            print(f"Starting local whisper transcription for: {file_path}")

            # Run Whisper transcription in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    AITranscriptionService.model.transcribe,
                    file_path,   
                    task="transcribe"
                )
            )

            kinyarwanda_text = result.get("text", "").strip()
            if not kinyarwanda_text:
                print("Warning: Whisper returned empty transcription")
                return "", ""

            print(f"Kinyarwanda transcription length: {len(kinyarwanda_text)}")

            # Translate Kinyarwanda to English using deep-translator
            try:
                english_text = GoogleTranslator(source='auto', target='en').translate(kinyarwanda_text)
                print(f"Translation successful, length: {len(english_text)}")
            except Exception as e:
                print(f"Translation failed: {str(e)}")
                english_text = ""

            return kinyarwanda_text, english_text

        except Exception as e:
            print(f"Local Whisper transcription error: {str(e)}")
            raise e

# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str
    role: str

class AudioAssign(BaseModel):
    topic: str
    collector_id: int
    transcriber_id: int
    validator_id: Optional[int] = None

class TranscriptionSubmit(BaseModel):
    transcription: str

class ValidationSubmit(BaseModel):
    final_transcription: str
    human_transcription_quality: int
    ai_transcription_quality: int
    notes: Optional[str] = None

class AudioResponse(BaseModel):
    id: int
    topic: str
    status: str
    created_at: datetime
    updated_at: datetime
    download_url: str
    filename: str
    duration: Optional[float] = None
    quality_issues: Optional[List[str]] = None
    ai_transcription: Optional[str] = None
    human_transcription: Optional[str] = None
    final_transcription: Optional[str] = None
    ai_transcription_english: Optional[str] = None
    class Config:
        from_attributes = True

async def process_audio_background(audio_id: int, file_path: str):
    db = SessionLocal()
    try:
        audio = db.query(Audio).get(audio_id)
        if not audio:
            print(f"Audio {audio_id} not found in database")
            return

        print(f"Processing audio {audio_id}: {file_path}")

        validation = AudioValidator.validate_audio_quality(file_path)
        audio.quality_metrics = json.dumps(validation["metrics"])
        audio.quality_issues = json.dumps(validation["issues"])

        if not validation["valid"]:
            print(f"Audio {audio_id} failed validation: {validation['issues']}")
            audio.status = AudioStatus.rejected
            audio.updated_at = datetime.utcnow()
            db.commit()
            return

        wav_path = os.path.join(PROCESSED_DIR, f"{audio_id}_processed.wav")
        if AudioValidator.convert_to_wav(file_path, wav_path):
            audio.processed_file_path = wav_path
            print(f"Audio converted to WAV: {wav_path}")
        else:
            print(f"WAV conversion failed for audio {audio_id}")

        # Use AITranscriptionService to transcribe and translate
        try:
            transcription_file = wav_path if os.path.exists(wav_path) else file_path
            print(f"Attempting transcription with file: {transcription_file}")

            kinyarwanda_text, english_text = await AITranscriptionService.transcribe_and_translate(transcription_file)

            if kinyarwanda_text:
                audio.ai_transcription = kinyarwanda_text
                audio.ai_transcription_english = english_text
                print(f"AI transcription and translation successful for audio {audio_id}")
            else:
                print(f"AI transcription returned empty result for audio {audio_id}")

        except Exception as e:
            print(f"AI transcription/translation failed for audio {audio_id}: {str(e)}")
            audio.ai_transcription = f"Transcription failed: {str(e)}"

        audio.status = AudioStatus.ready_for_transcription
        audio.updated_at = datetime.utcnow()
        db.commit()
        print(f"Audio {audio_id} processing completed")

    except Exception as e:
        print(f"Background processing failed for audio {audio_id}: {str(e)}")
        if audio:
            audio.status = AudioStatus.rejected
            audio.updated_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()

@app.get("/")
def root():
    return {"status": "Audio Processing Server running ✅", "version": "1.0.0"}

@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    token = create_access_token({"sub": user.username}, expires_delta=None)
    return {f"access_token": token,
             "token_type": "bearer", 
             "user":{user.username}, 
             "role": {user.role}}

@app.post("/create-user")
def create_user(user: UserCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "admin" or user.role not in {"transcriber", "datacollector", "validator", "admin"}:
        raise HTTPException(status_code=403, detail="Unauthorized or invalid role")
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = User(username=user.username, password_hash=hash_password(user.password), role=user.role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": f"User '{user.username}' created"}

@app.post("/assign-audio")
def assign_audio(data: AudioAssign, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can assign audios")
    collector = db.query(User).get(data.collector_id)
    transcriber = db.query(User).get(data.transcriber_id)
    validator = db.query(User).get(data.validator_id) if data.validator_id else None
    if not (collector and collector.role == "datacollector"):
        raise HTTPException(status_code=400, detail="Invalid collector")
    if not (transcriber and transcriber.role == "transcriber"):
        raise HTTPException(status_code=400, detail="Invalid transcriber")
    if data.validator_id and (not validator or validator.role != "validator"):
        raise HTTPException(status_code=400, detail="Invalid validator")
    audio = Audio(topic=data.topic, file_path="", assigned_collector_id=collector.id,
                  assigned_transcriber_id=transcriber.id,
                  assigned_validator_id=validator.id if validator else None,
                  status=AudioStatus.pending)
    db.add(audio)
    db.commit()
    db.refresh(audio)
    return {"msg": "Audio assigned", "audio_id": audio.id}

@app.post("/submit-audio/{audio_id}")
async def submit_audio(audio_id: int, file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks(),
                       current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    audio = db.query(Audio).get(audio_id)
    if not audio or audio.assigned_collector_id != current_user.id or current_user.role != "datacollector":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    file_path = os.path.join(UPLOAD_DIR, f"{audio_id}_{file.filename}")
    
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Verify file was written
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    audio.file_path = file_path
    audio.original_filename = file.filename
    audio.status = AudioStatus.processing
    audio.updated_at = datetime.utcnow()
    db.commit()
    background_tasks.add_task(process_audio_background, audio_id, file_path)
    return {"msg": "Audio submitted and processing started", "file_path": file_path}

@app.get("/assigned-audios")
def get_assigned_audios(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    role = current_user.role
    if role not in {"transcriber", "datacollector", "validator", "admin"}:
        raise HTTPException(status_code=403, detail="Access restricted")

    query = db.query(Audio)
    if role == "transcriber":
        audios = query.filter(
        Audio.assigned_transcriber_id == current_user.id,
        Audio.status == AudioStatus.ready_for_transcription
    ).all()
    elif role == "validator":
        audios = query.filter(
        Audio.assigned_validator_id == current_user.id,
        Audio.status == AudioStatus.ready_for_validation
    ).all()
    elif role == "datacollector":
        audios = query.filter(
        Audio.assigned_collector_id == current_user.id
    ).all()
    elif role == "admin":
        audios = query.all()   # admin sees all


    def parse_json(raw):
        try:
            return json.loads(raw) if raw else None
        except Exception:
            return None

    response = []
    for audio in audios:
        file_path = audio.processed_file_path or audio.file_path
        file_exists = file_path and os.path.exists(file_path)
        quality_issues = parse_json(audio.quality_issues) or []
        quality_metrics = parse_json(audio.quality_metrics) or {}
        response.append(AudioResponse(
            id=audio.id,
            topic=audio.topic,
            status=audio.status.value,
            created_at=audio.created_at,
            updated_at=audio.updated_at,
            download_url=f"/download-audio/{audio.id}" if file_exists else "",
            filename=os.path.basename(file_path) if file_exists else "File not found",
            duration=quality_metrics.get("duration"),
            quality_issues=quality_issues,
            ai_transcription=audio.ai_transcription,
            ai_transcription_english=audio.ai_transcription_english,
            human_transcription=audio.transcription,
            final_transcription=audio.final_transcription,
        ))
    return response

@app.get("/download-audio/{audio_id}")
def download_audio(audio_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    audio = db.query(Audio).get(audio_id)
    if not audio:
        raise HTTPException(status_code=404, detail="Audio not found")
    authorized = (current_user.role == "admin" or
                  (current_user.role == "transcriber" and audio.assigned_transcriber_id == current_user.id) or
                  (current_user.role == "validator" and audio.assigned_validator_id == current_user.id))
    if not authorized:
        raise HTTPException(status_code=403, detail="Not authorized")
    file_path = audio.processed_file_path or audio.file_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    media_type = 'audio/wav' if file_path.endswith('.wav') else 'audio/mpeg'
    return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type=media_type)

@app.post("/submit-transcription/{audio_id}")
def submit_transcription(audio_id: int, data: TranscriptionSubmit,
                         current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    audio = db.query(Audio).get(audio_id)
    if not audio or audio.assigned_transcriber_id != current_user.id or current_user.role != "transcriber":
        raise HTTPException(status_code=403, detail="Not authorized")
    audio.transcription = data.transcription
    audio.status = AudioStatus.ready_for_validation if audio.assigned_validator_id else AudioStatus.ready_for_review
    audio.updated_at = datetime.utcnow()
    db.commit()
    return {"msg": "Transcription submitted"}

@app.post("/force-transcription/{audio_id}")
async def force_transcription(
    audio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Force re-transcription using AI"""
    if current_user.role not in {"admin", "validator"}:
        raise HTTPException(status_code=403, detail="Only admin or validator can force transcription")

    audio = db.query(Audio).get(audio_id)
    if not audio:
        raise HTTPException(status_code=404, detail="Audio not found")

    file_path = audio.processed_file_path or audio.file_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    try:
        print(f"🔄 Force transcription requested for audio {audio_id}")
        kinyarwanda_text, english_text = await AITranscriptionService.transcribe_and_translate(file_path)
        
        if not kinyarwanda_text:
            return {
                "msg": "Transcription completed but returned empty result", 
                "transcription": "[No speech detected]"
            }
            
        audio.ai_transcription = kinyarwanda_text
        audio.ai_transcription_english = english_text
        audio.updated_at = datetime.utcnow()
        db.commit()
        
        print(f"✅ Force transcription completed for audio {audio_id}")
        return {"msg": "AI transcription updated successfully!", "transcription": kinyarwanda_text}
        
    except Exception as e:
        error_msg = f"AI transcription failed: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/submit-validation/{audio_id}")
def submit_validation(audio_id: int, data: ValidationSubmit,
                      current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "validator":
        raise HTTPException(status_code=403, detail="Only validators can submit")
    audio = db.query(Audio).get(audio_id)
    if not audio or audio.assigned_validator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    audio.final_transcription = data.final_transcription
    audio.human_transcription_quality = data.human_transcription_quality
    audio.ai_transcription_quality = data.ai_transcription_quality
    audio.validation_notes = data.notes
    audio.status = AudioStatus.ready_for_review
    audio.updated_at = datetime.utcnow()
    db.commit()
    return {"msg": "Validation submitted"}

@app.post("/review-transcription/{audio_id}")
def review_transcription(audio_id: int, approve: bool,
                         current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can review")
    audio = db.query(Audio).get(audio_id)
    if not audio or audio.status != AudioStatus.ready_for_review:
        raise HTTPException(status_code=400, detail="Transcription not ready")
    audio.status = AudioStatus.approved if approve else AudioStatus.rejected
    audio.updated_at = datetime.utcnow()
    db.commit()
    return {"msg": f"Transcription {'approved' if approve else 'rejected'}"}

@app.get("/audio-quality-report/{audio_id}")
def get_audio_quality_report(audio_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    audio = db.query(Audio).get(audio_id)
    if not audio:
        raise HTTPException(status_code=404, detail="Audio not found")
    authorized = (current_user.role in {"admin", "validator"} or
                  (current_user.role == "transcriber" and audio.assigned_transcriber_id == current_user.id) or
                  (current_user.role == "datacollector" and audio.assigned_collector_id == current_user.id))
    if not authorized:
        raise HTTPException(status_code=403, detail="Not authorized")
    quality_metrics = json.loads(audio.quality_metrics) if audio.quality_metrics else {}
    quality_issues = json.loads(audio.quality_issues) if audio.quality_issues else []
    return {
        "audio_id": audio.id,
        "topic": audio.topic,
        "status": audio.status.value,
        "quality_metrics": quality_metrics,
        "quality_issues": quality_issues,
        "ai_transcription": audio.ai_transcription,
        "human_transcription": audio.transcription,
        "final_transcription": audio.final_transcription,
        "validation_notes": audio.validation_notes,
        "quality_ratings": {
            "human_transcription": audio.human_transcription_quality,
            "ai_transcription": audio.ai_transcription_quality,
        },
    }

@app.on_event("startup")
async def startup_event():
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.role == "admin").first():
            default_admin = User(username="admin", password_hash=hash_password("admin123"), role="admin")
            db.add(default_admin)
            db.commit()
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)