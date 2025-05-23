import faiss
import numpy as np
from typing import List, Dict, Any, Optional
import os
import json
from pathlib import Path
import time
import logging
import gc
from functools import lru_cache
import re
from app.utils.gemini_embeddings import get_embedding_dimension
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, dimension: Optional[int] = 384, nlist: int = 100):
        """
        Initialize the vector store.
        
        Args:
            dimension: The dimension of the embeddings (default: 384 for Gemini)
            nlist: Number of clusters for IVF index (default: 100)
        """
        self.dimension = dimension
        self.nlist = nlist
        self.documents = []
        self.metadata = []
        self.storage_dir = Path("vector_store")
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize GPU resources if available
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                logger.info("Successfully initialized GPU resources")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU resources: {str(e)}")
                self.use_gpu = False
        
        # Initialize the quantizer
        self.quantizer = faiss.IndexFlatL2(self.dimension)
        
        # Initialize the index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Load existing index if available
        self._load_index()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better search results."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as'}
        words = text.split()
        words = [word for word in words if word not in stop_words]
        
        return ' '.join(words)
    
    def verify_state(self) -> Dict[str, Any]:
        """
        Verify the current state of the vector store and return detailed information.
        """
        try:
            state = {
                "total_documents": len(self.documents),
                "total_vectors": self.index.ntotal if hasattr(self, 'index') and self.index is not None else 0,
                "index_type": "IVFPQ" if hasattr(self.index, 'nlist') else "Flat",
                "is_trained": self.index.is_trained if hasattr(self, 'index') and self.index is not None else False,
                "nlist": self.nlist,
                "dimension": self.dimension,
                "using_gpu": self.use_gpu,
                "storage_path": str(self.storage_dir),
                "index_exists": hasattr(self, 'index') and self.index is not None,
                "metadata_count": len(self.metadata),
                "last_document_id": len(self.documents) - 1 if self.documents else None
            }
            
            # Add sample metadata if available
            if self.metadata:
                state["sample_metadata"] = {
                    "first": self.metadata[0],
                    "last": self.metadata[-1]
                }
            
            logger.info(f"Vector store state: {json.dumps(state, indent=2)}")
            return state
        except Exception as e:
            logger.error(f"Error verifying vector store state: {str(e)}")
            raise
    
    def _create_index(self, n_vectors: int):
        """Create a new index with appropriate parameters based on data size."""
        try:
            # Calculate number of clusters based on data size
            self.nlist = min(
                int(np.sqrt(n_vectors)),  # Square root of data size
                self.nlist,              # Maximum allowed clusters
                max(1, n_vectors // 4)    # At least 4 vectors per cluster
            )
            
            # Ensure at least 1 cluster and at most n_vectors/2 clusters
            self.nlist = max(1, min(self.nlist, n_vectors // 2))
            
            logger.info(f"Creating index with {self.nlist} clusters for {n_vectors} vectors")
            
            # Create the IVF index with product quantization
            self.index = faiss.IndexIVFPQ(
                self.quantizer, 
                self.dimension,    # dimension
                self.nlist,       # number of clusters
                8,               # number of sub-quantizers
                8               # bits per code
            )
            logger.info(f"Successfully created IVF index with {self.nlist} clusters")
        except Exception as e:
            logger.warning(f"Failed to create IVF index: {str(e)}")
            logger.info("Falling back to flat index")
            # Fallback to flat index if IVF creation fails
            self.index = faiss.IndexFlatL2(self.dimension)
            self.nlist = 1  # Flat index has no clusters
            logger.info("Successfully created flat index")
    
    def _load_index(self):
        """Load existing index and metadata if available."""
        index_path = self.storage_dir / "faiss_index.bin"
        metadata_path = self.storage_dir / "metadata.json"
        backup_path = self.storage_dir / "faiss_index.bin.bak"
        
        if index_path.exists() and metadata_path.exists():
            try:
                logger.info("Loading existing index and metadata")
                # Load the index with memory mapping for large indices
                if os.path.getsize(index_path) > 1_000_000_000:  # 1GB
                    self.index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
                else:
                    self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.metadata = data['metadata']
                    self.documents = data['documents']
                
                # Set nlist from loaded index
                if hasattr(self.index, 'nlist'):
                    self.nlist = self.index.nlist
                else:
                    self.nlist = 1  # Flat index
                
                logger.info(f"Successfully loaded index with {len(self.documents)} documents")
                # Verify the loaded state
                self.verify_state()
            except Exception as e:
                logger.error(f"Error loading existing index: {str(e)}")
                # Try to load from backup
                if backup_path.exists():
                    logger.info("Attempting to load from backup")
                    self.index = faiss.read_index(str(backup_path))
                else:
                    # Reset to empty state
                    self._reset_index()
    
    def _save_index(self):
        """Save the current index and metadata with backup."""
        try:
            logger.info("Saving index and metadata")
            
            # Create backup of existing index
            index_path = self.storage_dir / "faiss_index.bin"
            if index_path.exists():
                backup_path = self.storage_dir / "faiss_index.bin.bak"
                os.replace(index_path, backup_path)
            
            # Save the index with memory mapping for large indices
            if self.index.ntotal > 1_000_000:  # 1M vectors
                faiss.write_index(self.index, str(index_path), faiss.IO_FLAG_MMAP)
            else:
                faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            with open(self.storage_dir / "metadata.json", 'w') as f:
                json.dump({
                    'metadata': self.metadata,
                    'documents': self.documents
                }, f)
            logger.info("Successfully saved index and metadata")
            # Verify the saved state
            self.verify_state()
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            # Restore from backup if save failed
            if backup_path.exists():
                os.replace(backup_path, index_path)
    
    def _reset_index(self):
        """Reset the index to empty state."""
        logger.info("Resetting index to empty state")
        self.quantizer = faiss.IndexFlatL2(self.dimension)
        self.nlist = None
        self.metadata = []
        self.documents = []
        logger.info("Successfully reset index")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add documents to the vector store with optimized batch processing.
        """
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            start_time = time.time()
            
            # Ensure all embeddings are numpy arrays of float32
            embeddings_array = np.array([np.array(emb, dtype=np.float32) for emb in embeddings])
            
            # Verify dimensions
            if not all(emb.shape[0] == self.dimension for emb in embeddings_array):
                raise ValueError(f"Embedding dimensions do not match. Expected {self.dimension}, got shapes: {[emb.shape for emb in embeddings_array]}")
            
            # Create or update index if needed
            if not hasattr(self, 'index') or self.index is None:
                self._create_index(len(embeddings))
            
            # Train the index if it's not trained yet and it's an IVF index
            if (not self.index.is_trained and 
                len(embeddings) > 0 and 
                hasattr(self.index, 'train')):
                try:
                    logger.info(f"Training index with {len(embeddings)} vectors")
                    self.index.train(embeddings_array)
                    logger.info("Successfully trained index")
                except Exception as e:
                    logger.warning(f"Failed to train IVF index: {str(e)}")
                    logger.info("Falling back to flat index")
                    # Fallback to flat index if training fails
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.nlist = 1
            
            # Process in smaller batches
            batch_size = 1000
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings_array[i:i + batch_size]
                
                if self.use_gpu:
                    # Move batch to GPU
                    gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                    gpu_index.add(batch_embeddings)
                    self.index = faiss.index_gpu_to_cpu(gpu_index)
                else:
                    self.index.add(batch_embeddings)
                
                # Force garbage collection after each batch
                gc.collect()
                if self.use_gpu:
                    faiss.cudaFree(self.gpu_resources)
            
            # Store documents and metadata
            start_idx = len(self.documents)
            self.documents.extend(documents)
            self.metadata.extend(metadata)
            
            # Save the updated index
            self._save_index()
            
            end_time = time.time()
            logger.info(f"Successfully added {len(documents)} documents in {end_time - start_time:.2f} seconds")
            
            # Verify the final state
            state = self.verify_state()
            logger.info(f"Final vector store state: {json.dumps(state, indent=2)}")
            
            return list(range(start_idx, start_idx + len(documents)))
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}", exc_info=True)
            raise
    
    def _search_impl(self, query_embedding: np.ndarray, n_results: int) -> Dict[str, List]:
        """Internal search implementation."""
        try:
            if not hasattr(self, 'index') or self.index is None:
                raise ValueError("Vector store index is not initialized")
            
            if self.index.ntotal == 0:
                raise ValueError("Vector store is empty - no documents have been added")
            
            # Ensure query embedding has correct shape and type
            query_embedding = np.array(query_embedding, dtype=np.float32)
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Verify dimension matches
            if query_embedding.shape[1] != self.dimension:
                raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {self.dimension}")
            
            # Set search parameters for IVF index
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = min(20, self.nlist)  # Search in 20 nearest clusters
            
            if self.use_gpu:
                gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                distances, indices = gpu_index.search(query_embedding, n_results * 2)  # Get more results for filtering
            else:
                distances, indices = self.index.search(query_embedding, n_results * 2)  # Get more results for filtering
            
            # Filter and format results
            filtered_results = {
                "documents": [],
                "metadatas": [],
                "distances": []
            }
            
            seen_chunks = set()  # Track seen chunks to avoid duplicates
            min_similarity = 0.3  # Minimum similarity threshold
            
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx == -1:  # Skip invalid indices
                    continue
                
                # Convert distance to similarity score (ensure dist is a float)
                dist_float = float(dist)
                similarity = 1.0 / (1.0 + dist_float)
                
                # Skip if similarity is too low
                if similarity < min_similarity:
                    continue
                
                # Get chunk identifier
                chunk_id = f"{self.metadata[idx].get('source', '')}_{idx}"
                
                # Skip if we've seen this chunk
                if chunk_id in seen_chunks:
                    continue
                
                seen_chunks.add(chunk_id)
                
                # Add to results
                filtered_results["documents"].append(self.documents[idx])
                filtered_results["metadatas"].append(self.metadata[idx])
                filtered_results["distances"].append([dist_float])  # Store as list of single float
                
                # Break if we have enough results
                if len(filtered_results["documents"]) >= n_results:
                    break
            
            return filtered_results
        except Exception as e:
            logger.error(f"Error in search implementation: {str(e)}", exc_info=True)
            raise
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5
    ) -> Dict[str, List]:
        """
        Search for similar documents with performance monitoring.
        """
        try:
            logger.info(f"Searching for {n_results} similar documents")
            start_time = time.time()
            
            # Verify vector store state before search
            state = self.verify_state()
            logger.info(f"Vector store state before search: {json.dumps(state, indent=2)}")
            
            if not state["index_exists"]:
                raise ValueError("Vector store index is not initialized")
            
            if state["total_vectors"] == 0:
                raise ValueError("Vector store is empty - no documents have been added")
            
            results = self._search_impl(query_embedding, n_results)
            
            end_time = time.time()
            logger.info(f"Search completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Found {len(results['documents'])} results")
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}", exc_info=True)
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        """
        try:
            # Ensure index exists
            if not hasattr(self, 'index') or self.index is None:
                self.index = faiss.IndexFlatL2(self.dimension)
            
            stats = {
                "count": len(self.documents),
                "total_vectors": self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
                "dimension": self.dimension,
                "nlist": self.nlist if self.nlist is not None else 1,
                "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else False,
                "using_gpu": self.use_gpu,
                "nprobe": self.index.nprobe if hasattr(self.index, 'nprobe') else None,
                "index_type": "IVFPQ" if hasattr(self.index, 'nlist') else "Flat",
                "sub_quantizers": 8 if hasattr(self.index, 'nlist') else None,
                "bits_per_code": 8 if hasattr(self.index, 'nlist') else None
            }
            logger.info(f"Collection stats: {json.dumps(stats, indent=2)}")
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}", exc_info=True)
            raise
    
    def clear(self):
        """Clear all documents from the vector store."""
        try:
            logger.info("Clearing vector store")
            # Reset all storage
            self.documents = []
            self.metadata = []
            self.embeddings = None
            
            # Reinitialize the index
            if self.use_gpu:
                self.index = faiss.IndexFlatL2(self.dimension)
                self.gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
            
            logger.info("Vector store cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise

# Create a singleton instance with proper initialization
def get_vector_store():
    """Get or create the singleton vector store instance."""
    try:
        logger.info("Initializing new vector store instance")
        # Force reinitialization with new dimension
        if hasattr(get_vector_store, 'instance'):
            delattr(get_vector_store, 'instance')
        
        get_vector_store.instance = VectorStore()
        
        # Ensure the storage directory exists
        storage_dir = Path("vector_store")
        storage_dir.mkdir(exist_ok=True)
        
        # Load any existing index
        try:
            get_vector_store.instance._load_index()
            logger.info("Successfully loaded existing vector store index")
        except Exception as e:
            logger.error(f"Error loading initial vector store: {str(e)}", exc_info=True)
            logger.info("Resetting vector store to empty state")
            get_vector_store.instance._reset_index()
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
        raise
    return get_vector_store.instance

# Initialize the vector store
try:
    vector_store = get_vector_store()
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {str(e)}", exc_info=True)
    raise 