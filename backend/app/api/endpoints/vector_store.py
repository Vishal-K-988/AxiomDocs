from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.db.vector_store import vector_store
import numpy as np
from pydantic import BaseModel, Field
import json

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
    query_embedding: List[float]
    n_results: int = Field(default=5, ge=1, le=100)

class SearchResult(BaseModel):
    document: str
    metadata: Dict[str, Any]
    distance: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

class CollectionStats(BaseModel):
    count: int
    name: str
    metadata: Dict[str, Any]

class AddDocumentsRequest(BaseModel):
    documents: List[str]
    embeddings: List[List[float]]
    metadata: List[Dict[str, Any]]

class AddDocumentsResponse(BaseModel):
    message: str
    document_count: int

def flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Flatten metadata dictionary and convert all values to strings
    """
    flattened = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            flattened[key] = json.dumps(value)
        else:
            flattened[key] = str(value)
    return flattened

router = APIRouter()

@router.post("/test-vector-store", response_model=SearchResponse)
async def test_vector_store():
    """
    Test endpoint to verify ChromaDB functionality
    """
    try:
        # Create some test data
        test_documents = ["This is a test document", "Another test document"]
        test_embeddings = [
            [0.1, 0.2, 0.3],  # Simple test embedding
            [0.2, 0.3, 0.4]   # Simple test embedding
        ]
        test_metadata = [
            {"source": "test1", "type": "test"},
            {"source": "test2", "type": "test"}
        ]
        
        # Flatten metadata
        flattened_metadata = [flatten_metadata(meta) for meta in test_metadata]
        
        # Add documents to vector store
        vector_store.add_documents(test_documents, test_embeddings, flattened_metadata)
        
        # Search for similar documents
        query_embedding = [0.15, 0.25, 0.35]  # Similar to first document
        results = vector_store.search_similar(query_embedding, n_results=1)
        
        # Format results according to our response model
        formatted_results = []
        if results and "documents" in results:
            for i in range(len(results["documents"])):
                # Parse JSON strings back to dictionaries
                metadata = results.get("metadatas", [{}])[i] or {}
                parsed_metadata = {}
                for key, value in metadata.items():
                    try:
                        parsed_metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        parsed_metadata[key] = value
                
                formatted_results.append(
                    SearchResult(
                        document=str(results["documents"][i]),
                        metadata=parsed_metadata,
                        distance=float(results.get("distances", [0.0])[i][0] if results.get("distances") else 0.0)
                    )
                )
        
        return SearchResponse(results=formatted_results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store test failed: {str(e)}")

@router.get("/vector-store-stats", response_model=CollectionStats)
async def get_vector_store_stats():
    """
    Get statistics about the vector store collection
    """
    try:
        stats = vector_store.get_collection_stats()
        return CollectionStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get vector store stats: {str(e)}")

@router.post("/search", response_model=SearchResponse)
async def search_documents(query: SearchQuery):
    """
    Search for similar documents using a query embedding
    """
    try:
        results = vector_store.search_similar(
            query_embedding=query.query_embedding,
            n_results=query.n_results
        )
        
        formatted_results = []
        if results and "documents" in results:
            for i in range(len(results["documents"])):
                # Parse JSON strings back to dictionaries
                metadata = results.get("metadatas", [{}])[i] or {}
                parsed_metadata = {}
                for key, value in metadata.items():
                    try:
                        parsed_metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        parsed_metadata[key] = value
                
                formatted_results.append(
                    SearchResult(
                        document=str(results["documents"][i]),
                        metadata=parsed_metadata,
                        distance=float(results.get("distances", [0.0])[i][0] if results.get("distances") else 0.0)
                    )
                )
        
        return SearchResponse(results=formatted_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")

@router.delete("/clear-collection")
async def clear_collection():
    """
    Clear all documents from the collection
    """
    try:
        # Get all document IDs
        stats = vector_store.get_collection_stats()
        if stats["count"] > 0:
            # Delete all documents
            vector_store.delete_documents([f"doc_{i}" for i in range(stats["count"])])
        
        return {"message": "Collection cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}") 