import numpy as np
from typing import List, Dict, Any, Tuple
import json
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import logging
import gc
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import hashlib
import os
from pathlib import Path
import pickle
import nltk
from nltk.tokenize import sent_tokenize
import re
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from .langchain_utils import langchain_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download required NLTK data."""
    try:
        logger.info("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        logger.info("Successfully downloaded NLTK data")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        raise

# Download NLTK data at module import
download_nltk_data()

# Initialize the model with optimizations
logger.info("Initializing sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Enable model optimizations
model.max_seq_length = 128  # Reduce max sequence length for faster processing
if torch.cuda.is_available():
    model = model.to('cuda')
    # Enable mixed precision for faster inference
    model.half()  # Convert to FP16
    logger.info("Using CUDA with mixed precision for model inference")
else:
    # Use CPU optimizations
    torch.set_num_threads(4)  # Limit CPU threads
    logger.info("Using CPU for model inference")

# Create cache directory
CACHE_DIR = Path("embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_path(text: str) -> Path:
    """Generate cache file path for a text."""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return CACHE_DIR / f"{text_hash}.pkl"

def load_from_cache(text: str) -> np.ndarray:
    """Load embedding from cache if available."""
    cache_path = get_cache_path(text)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading from cache: {str(e)}")
    return None

def save_to_cache(text: str, embedding: np.ndarray):
    """Save embedding to cache."""
    try:
        cache_path = get_cache_path(text)
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
    except Exception as e:
        logger.warning(f"Error saving to cache: {str(e)}")

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?;:()\'"-]', '', text)
    # Normalize whitespace around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text.strip()

def split_text_into_chunks(text: str, chunk_size: int = 300, chunk_overlap: int = 30) -> List[str]:
    """
    Split text into smaller chunks using sentence boundaries and semantic coherence.
    """
    if not text or not isinstance(text, str):
        logger.warning("Empty or invalid text provided for chunking")
        return []
    
    logger.info(f"Starting text chunking. Text length: {len(text)} characters")
    
    # Clean the text
    text = clean_text(text)
    if not text:
        logger.warning("Text is empty after cleaning")
        return []
    
    try:
        # Split into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            logger.warning("No sentences found in text")
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error in text chunking: {str(e)}")
        # Fallback to simple character-based chunking
        logger.info("Falling back to character-based chunking")
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

def generate_embedding_with_timeout(text: str, timeout: int = 30) -> np.ndarray:
    """
    Generate embedding for a single text with timeout and caching.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Invalid text input")
    
    # Check cache first
    cached_embedding = load_from_cache(text)
    if cached_embedding is not None:
        logger.debug("Using cached embedding")
        return cached_embedding

    def _generate():
        with torch.no_grad():
            # Convert to tensor and move to device
            if torch.cuda.is_available():
                # Use mixed precision for faster inference
                with torch.cuda.amp.autocast():
                    return model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            else:
                return model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            embedding = future.result(timeout=timeout)
            
            # Normalize embedding
            embedding = F.normalize(torch.from_numpy(embedding), p=2, dim=0).numpy()
            
            # Save to cache
            save_to_cache(text, embedding)
            return embedding
    except TimeoutError:
        logger.error(f"Timeout while generating embedding for text of length {len(text)}")
        raise Exception("Embedding generation timed out")
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

# Alias for backward compatibility
generate_embedding = generate_embedding_with_timeout

def generate_embeddings_batch(texts: List[str], batch_size: int = 16) -> List[np.ndarray]:
    """
    Generate embeddings for a batch of texts with progress tracking and caching.
    """
    if not texts:
        logger.warning("Empty text list provided")
        return []
    
    logger.info(f"Starting batch embedding generation for {len(texts)} texts")
    embeddings = []
    texts_to_process = []
    text_indices = []
    
    # Check cache for each text
    for i, text in enumerate(texts):
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text at index {i}, skipping")
            continue
            
        cached_embedding = load_from_cache(text)
        if cached_embedding is not None:
            embeddings.append(cached_embedding)
        else:
            texts_to_process.append(text)
            text_indices.append(i)
    
    if not texts_to_process:
        logger.info("All embeddings found in cache")
        return embeddings
    
    try:
        # Process remaining texts in batches
        for i in tqdm(range(0, len(texts_to_process), batch_size), desc="Generating embeddings"):
            batch = texts_to_process[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}, size: {len(batch)}")
            
            # Process batch at once for better performance
            with torch.no_grad():
                if torch.cuda.is_available():
                    # Use mixed precision for faster inference
                    with torch.cuda.amp.autocast():
                        batch_embeddings = model.encode(
                            batch,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            batch_size=batch_size
                        )
                else:
                    batch_embeddings = model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=batch_size
                    )
            
            # Normalize embeddings
            batch_embeddings = F.normalize(torch.from_numpy(batch_embeddings), p=2, dim=1).numpy()
            
            # Save to cache and store results
            for text, embedding in zip(batch, batch_embeddings):
                save_to_cache(text, embedding)
            
            # Insert embeddings at correct positions
            for j, embedding in enumerate(batch_embeddings):
                idx = text_indices[i + j]
                embeddings.insert(idx, embedding)
            
            logger.info(f"Completed batch {i//batch_size + 1}")
            
            # Force garbage collection after each batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Error in batch embedding generation: {str(e)}")
        raise

def prepare_documents_for_vector_store(
    text: str,
    metadata: Dict[str, Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[str], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Prepare documents for vector store using Langchain.
    
    Args:
        text: The text to process
        metadata: Base metadata for all chunks
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Tuple of (chunks, embeddings, chunk_metadata)
    """
    try:
        # Split text into chunks using Langchain
        chunks = langchain_manager.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Generate embeddings using Langchain
        embeddings = langchain_manager.get_embeddings(chunks)
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        
        # Create chunk metadata
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                "embedding_norm": float(np.linalg.norm(embeddings[i])),
                "embedding_model": "gemini",
                "task_type": "retrieval_document"
            })
            chunk_metadata.append(chunk_meta)
        
        logger.info("Document preparation completed successfully")
        return chunks, embeddings, chunk_metadata
    except Exception as e:
        logger.error(f"Error in document preparation: {str(e)}")
        raise

def get_query_embedding(query: str) -> np.ndarray:
    """
    Get embedding for a search query using Gemini.
    Uses RETRIEVAL_QUERY task type for query embeddings.
    """
    try:
        from app.utils.gemini_embeddings import get_gemini_embedding, TaskType
        return get_gemini_embedding(query, task_type=TaskType.RETRIEVAL_QUERY)
    except Exception as e:
        logger.error(f"Error getting query embedding: {str(e)}")
        raise 