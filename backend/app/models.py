from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.sql import func
from .database import Base

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