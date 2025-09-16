from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

# --- Audio Status Enum ---
class AudioStatus(enum.Enum):
    pending = "pending"
    processing = "processing"
    ready_for_transcription = "ready_for_transcription"
    ready_for_validation = "ready_for_validation"
    ready_for_review = "ready_for_review"
    approved = "approved"
    rejected = "rejected"
    completed = "completed"

# --- User Table ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # admin, datacollector, transcriber, validator
    created_at = Column(DateTime, default=datetime.utcnow)
   
    # Relationships
    collected_audios = relationship("Audio", back_populates="collector", foreign_keys="Audio.assigned_collector_id")
    transcribed_audios = relationship("Audio", back_populates="transcriber", foreign_keys="Audio.assigned_transcriber_id")
    validated_audios = relationship("Audio", back_populates="validator", foreign_keys="Audio.assigned_validator_id")

# --- Audio Table ---
class Audio(Base):
    __tablename__ = "audios"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, nullable=False)
    file_path = Column(String, nullable=True)
    processed_file_path = Column(String, nullable=True)
    original_filename = Column(String, nullable=True)
    
    # File metadata
    file_size = Column(Integer, nullable=True)  # File size in bytes
    duration = Column(Float, nullable=True)  # Duration in seconds
    
    # Assignment fields
    assigned_collector_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    assigned_transcriber_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    assigned_validator_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Status and processing
    status = Column(Enum(AudioStatus), default=AudioStatus.pending)
    quality_metrics = Column(Text, nullable=True)  # JSON string
    quality_issues = Column(Text, nullable=True)  # JSON string
    
    # AI Transcription (single column, no duplicates)
    ai_transcription = Column(Text, nullable=True)  # AI transcription result (Kinyarwanda)
    ai_transcription_english = Column(Text, nullable=True)  # English translation
    gemini_processing_time = Column(Float, nullable=True)  # Time taken for AI transcription
    gemini_file_uri = Column(String, nullable=True)  # File URI for tracking
    
    # Human transcription workflow
    transcription = Column(Text, nullable=True)  # Manual transcription by transcriber (this was missing!)
    final_transcription = Column(Text, nullable=True)  # Final validated transcription
    
    # Quality ratings (1-10 scale)
    human_transcription_quality = Column(Integer, nullable=True)
    ai_transcription_quality = Column(Integer, nullable=True)  # AI quality rating
    validation_notes = Column(Text, nullable=True)
    
    # Error handling
    retry_count = Column(Integer, default=0)  # Track retry attempts
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    transcription_started_at = Column(DateTime, nullable=True)
    transcription_completed_at = Column(DateTime, nullable=True)

    # Relationships
    collector = relationship("User", back_populates="collected_audios", foreign_keys=[assigned_collector_id])
    transcriber = relationship("User", back_populates="transcribed_audios", foreign_keys=[assigned_transcriber_id])
    validator = relationship("User", back_populates="validated_audios", foreign_keys=[assigned_validator_id])

    def __repr__(self):
        return f"<Audio(id={self.id}, topic='{self.topic}', status='{self.status.value}')>"

    @property 
    def transcription_text(self):
        """Returns the best available transcription"""
        return self.final_transcription or self.transcription or self.ai_transcription

# --- Audio Processing Log Table ---
class AudioProcessingLog(Base):
    __tablename__ = "audio_processing_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    audio_id = Column(Integer, ForeignKey("audios.id"), nullable=False)
    stage = Column(String, nullable=False)  # "upload", "validation", "ai_transcription", "human_transcription"
    status = Column(String, nullable=False)  # "started", "completed", "failed"
    message = Column(Text, nullable=True)  # Details about the processing step
    processing_time = Column(Float, nullable=True)  # Time taken for this step
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    audio = relationship("Audio", backref="processing_logs")

    def __repr__(self):
        return f"<ProcessingLog(audio_id={self.audio_id}, stage='{self.stage}', status='{self.status}')>"