from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, JSON, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from .database import Base

class MessageType(enum.Enum):
    USER = "user"
    AI = "ai"

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    original_filename = Column(String)
    file_type = Column(String)
    file_size = Column(Integer)  # Size in bytes
    s3_key = Column(String, unique=True, index=True)
    user_id = Column(String, index=True)  # Will store Clerk user ID
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # PDF specific fields
    is_pdf = Column(Boolean, default=False)
    pdf_text = Column(Text, nullable=True)  # Store extracted text
    pdf_metadata = Column(Text, nullable=True)  # Store PDF metadata as JSON string
    pdf_page_count = Column(Integer, nullable=True)  # Number of pages in PDF
    
    # Vector store fields
    vector_store_ids = Column(JSON, nullable=True)  # Store ChromaDB document IDs as JSON array 

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # Will store Clerk user ID
    title = Column(String, nullable=True)  # Optional conversation title
    file_id = Column(Integer, nullable=True, index=True)  # Associate with a file
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"))
    message_type = Column(Enum(MessageType))
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Optional fields for RAG
    referenced_documents = Column(JSON, nullable=True)  # Store IDs of referenced documents
    embedding = Column(JSON, nullable=True)  # Store message embedding for semantic search
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages") 