from fastapi import APIRouter, UploadFile, File, HTTPException
from app.db.vector_store import vector_store
from app.utils.embeddings import prepare_documents_for_vector_store
import PyPDF2
import os
from typing import List, Dict, Any
import tempfile
import logging
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

async def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        logger.info(f"Extracting text from PDF: {file_path}")
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
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

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract its text, and store it in the vector store.
    
    Args:
        file: The PDF file to upload
        
    Returns:
        Dict containing upload status and document IDs
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        logger.info(f"Processing PDF upload: {file.filename}")
        
        # Clear existing documents from vector store
        logger.info("Clearing existing documents from vector store")
        vector_store.clear()
        
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            logger.info("Reading uploaded file content")
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"Saved temporary file at: {temp_file_path}")
        
        # Extract text from PDF
        text = await extract_text_from_pdf(temp_file_path)
        
        # Prepare metadata
        metadata = {
            "source": file.filename,
            "type": "pdf",
            "file_path": temp_file_path
        }
        
        # Prepare documents for vector store
        logger.info("Preparing documents for vector store")
        chunks, embeddings, chunk_metadata = prepare_documents_for_vector_store(
            text=text,
            metadata=metadata,
            chunk_size=300,  # Reduced chunk size
            chunk_overlap=30,  # Reduced overlap
            batch_size=16  # Reduced batch size
        )
        
        # Add to vector store
        logger.info("Adding documents to vector store")
        doc_ids = vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
            metadata=chunk_metadata
        )
        
        # Clean up temporary file
        logger.info("Cleaning up temporary file")
        os.unlink(temp_file_path)
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"Successfully processed PDF: {file.filename}")
        return {
            "status": "success",
            "message": f"Successfully processed PDF: {file.filename}",
            "document_ids": doc_ids,
            "chunks_processed": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        # Clean up temporary file in case of error
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
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