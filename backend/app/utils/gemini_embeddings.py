import google.generativeai as genai
import numpy as np
from typing import List, Union
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Gemini
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    logger.error(f"Error initializing Gemini: {str(e)}")
    raise

def get_embedding_dimension() -> int:
    """Get the dimension of Gemini embeddings."""
    return 384  # Gemini's actual embedding dimension

def get_query_embedding(text: str) -> np.ndarray:
    """
    Get embedding for a query text using Gemini.
    
    Args:
        text: The text to embed
        
    Returns:
        numpy.ndarray: The embedding vector
    """
    try:
        # Get embedding from Gemini
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        embedding = np.array(response["embedding"], dtype=np.float32)
        
        # Log the embedding shape for debugging
        logger.info(f"Generated embedding with shape: {embedding.shape}")
        
        # Ensure correct dimension (384 for Gemini)
        if embedding.shape[0] != 384:
            logger.warning(f"Unexpected embedding dimension: {embedding.shape[0]}, expected 384")
            if embedding.shape[0] == 768:
                # Take first 384 dimensions if we got 768
                embedding = embedding[:384]
            else:
                raise ValueError(f"Invalid embedding dimension: {embedding.shape[0]}")
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    except Exception as e:
        logger.error(f"Error generating query embedding: {str(e)}")
        raise

def get_batch_embeddings(texts: List[str], batch_size: int = 16) -> List[np.ndarray]:
    """
    Get embeddings for a batch of texts using Gemini.
    
    Args:
        texts: List of texts to embed
        batch_size: Size of batches to process
        
    Returns:
        List[numpy.ndarray]: List of embedding vectors
    """
    try:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = genai.embed_content(
                model="models/embedding-001",
                content=batch,
                task_type="retrieval_document"
            )
            
            for result in batch_results["embedding"]:
                embedding = np.array(result, dtype=np.float32)
                # Log the embedding shape for debugging
                logger.info(f"Generated batch embedding with shape: {embedding.shape}")
                
                # Normalize the embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embeddings.append(embedding)
                
        return embeddings
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {str(e)}")
        raise

# Task type constants for better code readability
class TaskType:
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY" 