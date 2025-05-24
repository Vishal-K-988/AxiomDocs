from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, List, Union
from enum import Enum

class FileBase(BaseModel):
    filename: str
    original_filename: str
    file_type: str
    file_size: int

class FileCreate(FileBase):
    pass

class PDFInfo(BaseModel):
    page_count: Optional[int] = None
    metadata: Optional[Dict] = None
    text: Optional[str] = None

class File(FileBase):
    id: int
    s3_key: str
    user_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_pdf: bool = False
    pdf_info: Optional[PDFInfo] = None

    class Config:
        orm_mode = True

class FileResponse(File):
    download_url: Optional[str] = None

class FileRenameRequest(BaseModel):
    new_filename: str

class MessageType(str, Enum):
    USER = "user"
    AI = "ai"

class MessageBase(BaseModel):
    content: str
    message_type: MessageType
    referenced_documents: Optional[List[int]] = None

class MessageCreate(MessageBase):
    conversation_id: int

class Message(MessageBase):
    id: int
    conversation_id: int
    created_at: datetime
    embedding: Optional[List[float]] = None

    model_config = {
        "from_attributes": True
    }

class ConversationBase(BaseModel):
    title: str
    file_id: Optional[int] = None

class ConversationCreate(ConversationBase):
    user_id: str

class Conversation(ConversationBase):
    id: int
    user_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    messages: List[Message] = []

    model_config = {
        "from_attributes": True
    }

class ConversationList(BaseModel):
    conversations: List[Conversation]

class ChatResponse(BaseModel):
    message: Message
    conversation: Conversation 