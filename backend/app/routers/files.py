from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header, Query
from typing import List, Optional, Dict, Any
from ..s3 import upload_file_to_s3, delete_file_from_s3, rename_file_in_s3
from ..database import get_db
from ..models import File as DBFile
from ..schemas import FileRenameRequest
from ..pdf_processor import extract_text_from_pdf, is_pdf_file
from ..utils.embeddings import prepare_documents_for_vector_store
from ..utils.gemini_embeddings import get_query_embedding
from ..db.vector_store import vector_store
from sqlalchemy.orm import Session
from datetime import datetime
import json
import logging
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

class PDFInfo(BaseModel):
    page_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    text: Optional[str] = None

class FileResponse(BaseModel):
    id: int
    name: str
    s3_url: str
    upload_time: str
    size: int
    is_pdf: bool
    pdf_info: Optional[PDFInfo] = None

class FileListResponse(BaseModel):
    files: List[FileResponse]

class FileDeleteResponse(BaseModel):
    message: str

router = APIRouter(
    prefix="/files",
    tags=["files"]
)

def get_user_id(x_user_id: str = Header(...)) -> str:
    return x_user_id

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    try:
        print(f"Starting file upload for {file.filename}")
        
        # Read file content
        file_content = await file.read()
        print(f"File content read successfully, size: {len(file_content)} bytes")
        
        # Upload to S3
        try:
            s3_url = upload_file_to_s3(file_content, file.filename, user_id)
            print(f"File uploaded to S3 successfully: {s3_url}")
        except Exception as s3_error:
            print(f"Error uploading to S3: {str(s3_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to upload file to S3: {str(s3_error)}")
        
        # Create database entry
        try:
            db_file = DBFile(
                filename=file.filename,
                original_filename=file.filename,
                file_type=file.content_type,
                file_size=len(file_content),
                s3_key=s3_url,
                user_id=user_id
            )
            db.add(db_file)
            db.flush()  # Get the ID without committing
            print(f"Database entry created with ID: {db_file.id}")
        except Exception as db_error:
            print(f"Error creating database entry: {str(db_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to create database entry: {str(db_error)}")
        
        # Process PDF if it's a PDF file
        if is_pdf_file(file.filename):
            try:
                print("Processing PDF file...")
                pdf_data = extract_text_from_pdf(file_content)
                print(f"PDF processed successfully. Page count: {pdf_data['page_count']}")
                
                db_file.is_pdf = True
                db_file.pdf_text = pdf_data["text"]
                db_file.pdf_metadata = json.dumps(pdf_data["metadata"])
                db_file.pdf_page_count = pdf_data["page_count"]
                
                # Prepare documents for vector store
                base_metadata = {
                    "file_id": db_file.id,
                    "filename": db_file.filename,
                    "user_id": user_id,
                    "is_pdf": True,
                    "page_count": pdf_data["page_count"],
                    "pdf_metadata": pdf_data["metadata"]
                }
                
                try:
                    print("Preparing documents for vector store...")
                    chunks, embeddings, chunk_metadata = prepare_documents_for_vector_store(
                        text=pdf_data["text"],
                        metadata=base_metadata,
                        chunk_size=1000,  # Using Langchain's default chunk size
                        chunk_overlap=200  # Using Langchain's default overlap
                    )
                    print(f"Created {len(chunks)} chunks for vector store")
                    
                    # Add to vector store
                    if chunks:
                        try:
                            print("Adding documents to vector store...")
                            vector_store_ids = vector_store.add_documents(chunks, embeddings, chunk_metadata)
                            db_file.vector_store_ids = vector_store_ids
                            print("Documents added to vector store successfully")
                        except Exception as vs_error:
                            print(f"Error adding to vector store: {str(vs_error)}")
                            # Continue even if vector store fails
                            db_file.vector_store_ids = []
                except Exception as prep_error:
                    print(f"Error preparing documents for vector store: {str(prep_error)}")
                    # Continue with file upload even if vector store preparation fails
                    db_file.vector_store_ids = []
                
            except Exception as pdf_error:
                print(f"Error processing PDF: {str(pdf_error)}")
                # Continue with file upload even if PDF processing fails
                db_file.is_pdf = False
                db_file.pdf_text = None
                db_file.pdf_metadata = None
                db_file.pdf_page_count = None
                db_file.vector_store_ids = []
        
        try:
            db.commit()
            db.refresh(db_file)
            print("Database transaction committed successfully")
        except Exception as commit_error:
            print(f"Error committing to database: {str(commit_error)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to commit database transaction: {str(commit_error)}")
        
        return {
            "id": db_file.id,
            "name": db_file.filename,
            "s3_url": db_file.s3_key,
            "upload_time": db_file.created_at.isoformat(),
            "size": db_file.file_size,
            "is_pdf": db_file.is_pdf,
            "pdf_info": {
                "page_count": db_file.pdf_page_count,
                "metadata": json.loads(db_file.pdf_metadata) if db_file.pdf_metadata else None,
                "text": db_file.pdf_text
            } if db_file.is_pdf else None,
            "vector_store_ids": db_file.vector_store_ids
        }
    except Exception as e:
        print(f"Unexpected error in upload_file: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=FileListResponse)
async def list_files(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    files = db.query(DBFile).filter(DBFile.user_id == user_id).all()
    
    return FileListResponse(
        files=[
            FileResponse(
                id=file.id,
                name=file.filename,
                s3_url=file.s3_key,
                upload_time=file.created_at.isoformat(),
                size=file.file_size,
                is_pdf=file.is_pdf,
                pdf_info=PDFInfo(
                    page_count=file.pdf_page_count,
                    metadata=json.loads(file.pdf_metadata) if file.pdf_metadata else None,
                    text=file.pdf_text
                ) if file.is_pdf else None
            )
            for file in files
        ]
    )

@router.get("/{file_id}", response_model=FileResponse)
async def get_file(
    file_id: int,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    file = db.query(DBFile).filter(DBFile.id == file_id, DBFile.user_id == user_id).first()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        id=file.id,
        name=file.filename,
        s3_url=file.s3_key,
        upload_time=file.created_at.isoformat(),
        size=file.file_size,
        is_pdf=file.is_pdf,
        pdf_info=PDFInfo(
            page_count=file.pdf_page_count,
            metadata=json.loads(file.pdf_metadata) if file.pdf_metadata else None,
            text=file.pdf_text
        ) if file.is_pdf else None
    )

@router.delete("/{file_id}", response_model=FileDeleteResponse)
async def delete_file(
    file_id: int,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    file = db.query(DBFile).filter(DBFile.id == file_id, DBFile.user_id == user_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Delete from S3
        delete_file_from_s3(file.s3_key)
        
        # Delete from database
        db.delete(file)
        db.commit()
        
        return FileDeleteResponse(message="File deleted successfully")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{file_id}/rename", response_model=FileResponse)
async def rename_file(
    file_id: int,
    rename_request: FileRenameRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    file = db.query(DBFile).filter(DBFile.id == file_id, DBFile.user_id == user_id).first()
    
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Rename in S3
        new_s3_url = rename_file_in_s3(file.s3_key, rename_request.new_filename, user_id)
        
        # Update in database
        file.filename = rename_request.new_filename
        file.s3_key = new_s3_url
        db.commit()
        
        return FileResponse(
            id=file.id,
            name=file.filename,
            s3_url=file.s3_key,
            upload_time=file.created_at.isoformat(),
            size=file.file_size,
            is_pdf=file.is_pdf,
            pdf_info=PDFInfo(
                page_count=file.pdf_page_count,
                metadata=json.loads(file.pdf_metadata) if file.pdf_metadata else None,
                text=file.pdf_text
            ) if file.is_pdf else None
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{file_id}/search")
async def search_pdf_content(
    file_id: int,
    query: str,
    n_results: int = 5,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    """
    Search through PDF content using semantic search
    """
    try:
        # Get the file
        file = db.query(DBFile).filter(DBFile.id == file_id, DBFile.user_id == user_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        if not file.is_pdf:
            raise HTTPException(status_code=400, detail="File is not a PDF")
        
        if not file.vector_store_ids:
            raise HTTPException(status_code=400, detail="No vector store data available for this file")
        
        # Verify vector store state
        try:
            vector_store_state = vector_store.verify_state()
            logger.info(f"Vector store state before search: {json.dumps(vector_store_state, indent=2)}")
        except Exception as vs_error:
            logger.error(f"Error verifying vector store state: {str(vs_error)}")
            raise HTTPException(status_code=500, detail="Vector store is not properly initialized")
        
        try:
            # Generate embedding for the query using the correct function
            query_embedding = get_query_embedding(query)
            logger.info(f"Generated query embedding with shape: {query_embedding.shape}")
        except Exception as embedding_error:
            logger.error(f"Error generating query embedding: {str(embedding_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {str(embedding_error)}")
        
        try:
            # Search in vector store
            results = vector_store.search_similar(
                query_embedding=query_embedding,
                n_results=n_results
            )
            logger.info(f"Search completed successfully. Found {len(results.get('documents', []))} results")
        except Exception as search_error:
            logger.error(f"Error searching vector store: {str(search_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to search vector store: {str(search_error)}")
        
        # Format results
        formatted_results = []
        if results and "documents" in results:
            for i in range(len(results["documents"])):
                try:
                    # Parse JSON strings back to dictionaries
                    metadata = results.get("metadatas", [{}])[i] or {}
                    parsed_metadata = {}
                    for key, value in metadata.items():
                        try:
                            parsed_metadata[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            parsed_metadata[key] = value
                    
                    # Convert distance to similarity score (0-100 scale)
                    distance = float(results.get("distances", [0.0])[i][0] if results.get("distances") else 0.0)
                    similarity_score = 100.0 * (1.0 / (1.0 + distance))
                    
                    formatted_results.append({
                        "text": results["documents"][i],
                        "metadata": parsed_metadata,
                        "similarity_score": round(similarity_score, 2)
                    })
                except Exception as result_error:
                    logger.error(f"Error formatting result {i}: {str(result_error)}")
                    continue
        
        # Sort results by similarity score
        formatted_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Log search results
        logger.info(f"Search completed successfully. Found {len(formatted_results)} results")
        
        return {
            "file_id": file.id,
            "filename": file.filename,
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search_pdf_content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}") 