from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging
import gc
from ..database import get_db
from ..models import File
from ..db.vector_store import vector_store
from ..utils.embeddings import prepare_documents_for_vector_store
from ..s3 import get_file_from_s3
import PyPDF2
import os
import tempfile
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

def get_user_id(x_user_id: str = Header(...)) -> str:
    return x_user_id

async def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF content."""
    try:
        logger.info("Extracting text from PDF content")
        pdf_file = io.BytesIO(file_content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        total_pages = len(reader.pages)
        logger.info(f"PDF has {total_pages} pages")
        
        for i, page in enumerate(reader.pages):
            logger.debug(f"Processing page {i+1}/{total_pages}")
            page_text = page.extract_text()
            text += page_text + "\n"
            
            # Force garbage collection after each page
            if i % 5 == 0:  # Every 5 pages
                gc.collect()
        
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")

@router.post("/process-pdf/{file_id}")
async def process_pdf(
    file_id: int,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    """
    Process a PDF file from S3, extract its text, and store it in the vector store.
    
    Args:
        file_id: The ID of the file to process
        user_id: The user ID from the header
        
    Returns:
        Dict containing processing status and document IDs
    """
    try:
        # Get file from database
        file = db.query(File).filter(
            File.id == file_id,
            File.user_id == user_id,
            File.is_pdf == True
        ).first()
        
        if not file:
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        logger.info(f"Processing PDF file: {file.filename}")
        
        # Get file content from S3
        try:
            file_content = get_file_from_s3(file.s3_key)
        except Exception as s3_error:
            logger.error(f"Error getting file from S3: {str(s3_error)}")
            raise HTTPException(status_code=500, detail=f"Error getting file from S3: {str(s3_error)}")
        
        # Extract text from PDF
        text = await extract_text_from_pdf(file_content)
        
        # Prepare metadata
        metadata = {
            "source": file.filename,
            "type": "pdf",
            "file_id": file.id,
            "user_id": user_id
        }
        
        # Prepare documents for vector store
        logger.info("Preparing documents for vector store")
        chunks, embeddings, chunk_metadata = prepare_documents_for_vector_store(
            text=text,
            metadata=metadata,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Add to vector store
        logger.info("Adding documents to vector store")
        doc_ids = vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
            metadata=chunk_metadata
        )
        
        # Update file with vector store IDs
        file.vector_store_ids = doc_ids
        db.commit()
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"Successfully processed PDF: {file.filename}")
        return {
            "status": "success",
            "message": f"Successfully processed PDF: {file.filename}",
            "document_ids": doc_ids,
            "chunks_processed": len(chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_documents(query: str, n_results: int = 5):
    """
    Search for similar documents using a text query.
    
    Args:
        query: The text query to search for
        n_results: Number of results to return (default: 5)
        
    Returns:
        List of similar documents with their metadata
    """
    try:
        logger.info(f"Processing search query: {query}")
        from app.utils.embeddings import generate_embedding
        
        # Generate embedding for the query
        logger.info("Generating query embedding")
        query_embedding = generate_embedding(query)
        
        # Search the vector store
        logger.info("Searching vector store")
        results = vector_store.search_similar(query_embedding, n_results=n_results)
        
        # Format results
        formatted_results = []
        for doc, meta, dist_list in zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        ):
            # Get the first (and only) distance from the list
            dist = float(dist_list[0])
            # Convert distance to similarity score (0-100 scale)
            similarity_score = 100.0 * (1.0 / (1.0 + dist))
            
            # Clean and format the content
            content = doc.strip()
            if len(content) > 200:  # Truncate long content
                content = content[:197] + "..."
            
            # Clean up metadata
            cleaned_metadata = {
                "source": meta.get("source", "Unknown"),
                "page": meta.get("chunk_index", 0) + 1,  # Convert to 1-based page numbers
                "total_pages": meta.get("total_chunks", 0)
            }
            
            formatted_results.append({
                "content": content,
                "metadata": cleaned_metadata,
                "similarity_score": round(similarity_score, 2)
            })
        
        # Sort results by similarity score in descending order
        formatted_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        logger.info(f"Found {len(formatted_results)} results")
        return {
            "query": query.strip(),
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats():
    """
    Get statistics about the vector store.
    
    Returns:
        Dict containing vector store statistics
    """
    try:
        logger.info("Getting vector store statistics")
        stats = vector_store.get_collection_stats()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 