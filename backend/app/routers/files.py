from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header, Query
from typing import List, Optional
from ..s3 import upload_file_to_s3, delete_file_from_s3, rename_file_in_s3
from ..database import get_db
from ..models import File
from ..schemas import FileRenameRequest
from sqlalchemy.orm import Session
from datetime import datetime

router = APIRouter(
    prefix="/files",
    tags=["files"]
)

async def get_user_id(x_user_id: Optional[str] = Header(None)):
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User ID is required")
    return x_user_id

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    try:
        # Upload to S3
        s3_url = upload_file_to_s3(file.file, file.filename, user_id)
        
        # Create database entry
        db_file = File(
            filename=file.filename,
            original_filename=file.filename,
            file_type=file.content_type,
            file_size=file.size,
            s3_key=s3_url,
            user_id=user_id
        )
        
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        
        return {
            "id": db_file.id,
            "name": db_file.filename,
            "s3_url": db_file.s3_key,
            "upload_time": db_file.created_at.isoformat(),
            "size": db_file.file_size
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def list_files(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    files = db.query(File).filter(File.user_id == user_id).all()
    
    return [
        {
            "id": file.id,
            "name": file.filename,
            "s3_url": file.s3_key,
            "upload_time": file.created_at.isoformat(),
            "size": file.file_size
        }
        for file in files
    ]

@router.delete("/{file_id}")
async def delete_file(
    file_id: int,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    file = db.query(File).filter(File.id == file_id, File.user_id == user_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Delete from S3
        delete_file_from_s3(file.s3_key)
        
        # Delete from database
        db.delete(file)
        db.commit()
        
        return {"message": "File deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{file_id}/rename")
async def rename_file(
    file_id: int,
    rename_request: FileRenameRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    file = db.query(File).filter(File.id == file_id, File.user_id == user_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Rename in S3
        new_s3_url = rename_file_in_s3(file.s3_key, rename_request.new_filename, user_id)
        
        # Update in database
        file.filename = rename_request.new_filename
        file.s3_key = new_s3_url
        db.commit()
        
        return {
            "id": file.id,
            "name": file.filename,
            "s3_url": file.s3_key
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e)) 