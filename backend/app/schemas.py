from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class FileBase(BaseModel):
    filename: str
    original_filename: str
    file_type: str
    file_size: int

class FileCreate(FileBase):
    pass

class File(FileBase):
    id: int
    s3_key: str
    user_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class FileResponse(File):
    download_url: Optional[str] = None

class FileRenameRequest(BaseModel):
    new_filename: str 