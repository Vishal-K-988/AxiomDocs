from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from app.db.vector_store import vector_store
from app.utils.langchain_utils import langchain_manager
import numpy as np
from pydantic import BaseModel, Field
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

class DocumentMetadata(BaseModel):
    source: str
    type: str
    additional_info: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    text: str
    metadata: DocumentMetadata

class DocumentBatch(BaseModel):
    documents: List[Document]

class SearchQuery(BaseModel):
    query: str
    n_results: int = Field(default=5, ge=1, le=100)

class SearchResult(BaseModel):
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[List[float]]

class SearchResponse(BaseModel):
    results: List[SearchResult]

class VectorStoreStats(BaseModel):
    total_documents: int
    total_vectors: int
    dimension: int
    index_type: str
    is_trained: bool
    using_gpu: bool

class AddDocumentsRequest(BaseModel):
    documents: List[str]
    embeddings: List[List[float]]
    metadata: List[Dict[str, Any]]

class AddDocumentsResponse(BaseModel):
    message: str
    document_count: int

def flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Flatten metadata dictionary for storage."""
    flattened = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            flattened[key] = json.dumps(value)
        else:
            flattened[key] = str(value)
    return flattened

router = APIRouter()

@router.post("/test")
async def test_vector_store():
    """Test the vector store with sample documents."""
    try:
        # Create test documents
        test_documents = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of AI that focuses on data."
        ]
        
        # Generate proper embeddings using Langchain
        test_embeddings = langchain_manager.get_embeddings(test_documents)
        
        # Create test metadata
        test_metadata = [
            {
                "source": "test",
                "type": "test_document",
                "additional_info": {"test_id": 1}
            },
            {
                "source": "test",
                "type": "test_document",
                "additional_info": {"test_id": 2}
            }
        ]
        
        # Add documents to vector store
        vector_store_ids = vector_store.add_documents(test_documents, test_embeddings, test_metadata)
        
        # Test search
        query = "What is artificial intelligence?"
        query_embedding = langchain_manager.get_query_embedding(query)
        search_results = vector_store.search_similar(query_embedding, n_results=2)
        
        # Clean up test data
        vector_store.clear()
        
        return {
            "status": "success",
            "message": "Vector store test completed successfully",
            "test_results": {
                "documents_added": len(vector_store_ids),
                "search_results": search_results
            }
        }
    except Exception as e:
        logger.error(f"Error testing vector store: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=VectorStoreStats)
async def get_vector_store_stats():
    """Get vector store statistics."""
    try:
        stats = vector_store.get_collection_stats()
        return VectorStoreStats(
            total_documents=stats["count"],
            total_vectors=stats["total_vectors"],
            dimension=stats["dimension"],
            index_type=stats["index_type"],
            is_trained=stats["is_trained"],
            using_gpu=stats["using_gpu"]
        )
    except Exception as e:
        logger.error(f"Error getting vector store stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=SearchResponse)
async def search_documents(query: SearchQuery):
    """Search documents using semantic search."""
    try:
        # Generate query embedding using Langchain
        query_embedding = langchain_manager.get_query_embedding(query.query)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Search in vector store
        results = vector_store.search_similar(
            query_embedding=query_embedding,
            n_results=query.n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"])):
            # Convert distance to similarity score
            distance = float(results["distances"][i][0])
            similarity_score = 100.0 * (1.0 / (1.0 + distance))
            
            formatted_results.append(
                SearchResult(
                    documents=[results["documents"][i]],
                    metadatas=[results["metadatas"][i]],
                    distances=[[distance]]
                )
            )
        
        return SearchResponse(results=formatted_results)
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-documents", response_model=AddDocumentsResponse)
async def add_documents(request: AddDocumentsRequest):
    """
    Add documents to the vector store
    """
    try:
        if len(request.documents) != len(request.embeddings) or len(request.documents) != len(request.metadata):
            raise HTTPException(
                status_code=400,
                detail="Number of documents, embeddings, and metadata must match"
            )
        
        # Flatten metadata
        flattened_metadata = [flatten_metadata(meta) for meta in request.metadata]
        
        vector_store.add_documents(
            documents=request.documents,
            embeddings=request.embeddings,
            metadata=flattened_metadata
        )
        
        return AddDocumentsResponse(
            message="Documents added successfully",
            document_count=len(request.documents)
        )
    except Exception as e:
        logger.error(f"Failed to add documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")

@router.post("/clear")
async def clear_vector_store():
    """Clear all documents from the vector store."""
    try:
        vector_store.clear()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 