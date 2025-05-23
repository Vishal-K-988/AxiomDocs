from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
import os
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class LangchainManager:
    def __init__(self):
        """Initialize Langchain with Gemini embeddings."""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                task_type="retrieval_document",
                model_kwargs={"dimensions": 384}  # Explicitly set dimension to 384
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            logger.info("Successfully initialized Langchain with Gemini embeddings")
        except Exception as e:
            logger.error(f"Error initializing Langchain: {str(e)}")
            raise

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            raise

    def create_documents(self, texts: List[str], metadata: Dict[str, Any]) -> List[Document]:
        """Create Langchain documents with metadata."""
        try:
            documents = [
                Document(
                    page_content=text,
                    metadata=metadata
                ) for text in texts
            ]
            logger.info(f"Created {len(documents)} documents with metadata")
            return documents
        except Exception as e:
            logger.error(f"Error creating documents: {str(e)}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a list of texts."""
        try:
            # Get embeddings from Gemini
            embeddings = self.embeddings.embed_documents(texts)
            
            # Convert to numpy arrays and ensure correct dimension
            numpy_embeddings = []
            for emb in embeddings:
                # Convert to numpy array and ensure float32
                emb_array = np.array(emb, dtype=np.float32)
                
                # Verify dimension
                if emb_array.shape[0] != 384:
                    logger.warning(f"Unexpected embedding dimension: {emb_array.shape[0]}, expected 384")
                    # Reshape if possible, otherwise raise error
                    if emb_array.shape[0] == 768:
                        # Take first 384 dimensions if we got 768
                        emb_array = emb_array[:384]
                    else:
                        raise ValueError(f"Invalid embedding dimension: {emb_array.shape[0]}")
                
                # Normalize the embedding
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm
                
                numpy_embeddings.append(emb_array)
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return numpy_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def get_query_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single query text."""
        try:
            # Get embedding from Gemini
            embedding = self.embeddings.embed_query(text)
            
            # Convert to numpy array and ensure float32
            emb_array = np.array(embedding, dtype=np.float32)
            
            # Verify dimension
            if emb_array.shape[0] != 384:
                logger.warning(f"Unexpected embedding dimension: {emb_array.shape[0]}, expected 384")
                # Reshape if possible, otherwise raise error
                if emb_array.shape[0] == 768:
                    # Take first 384 dimensions if we got 768
                    emb_array = emb_array[:384]
                else:
                    raise ValueError(f"Invalid embedding dimension: {emb_array.shape[0]}")
            
            # Normalize the embedding
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                emb_array = emb_array / norm
            
            logger.info("Generated query embedding")
            return emb_array
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

# Create a singleton instance
langchain_manager = LangchainManager() 