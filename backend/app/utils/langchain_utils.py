from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from typing import List, Dict, Any
import os
import logging
import numpy as np
from ..config import (
    LANGCHAIN_API_KEY,
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_ENDPOINT,
    LANGCHAIN_PROJECT,
    VECTOR_STORE_DIMENSION,
    VECTOR_STORE_CHUNK_SIZE,
    VECTOR_STORE_CHUNK_OVERLAP
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VECTOR_STORE_DIMENSION = 384  # Gemini's embedding dimension
VECTOR_STORE_CHUNK_SIZE = 1000
VECTOR_STORE_CHUNK_OVERLAP = 200

# Set Langchain environment variables
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = str(LANGCHAIN_TRACING_V2)
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

class LangchainManager:
    def __init__(self):
        """Initialize Langchain with Gemini embeddings and chat model."""
        try:
            # Initialize embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",  # Using the stable embedding model
                google_api_key=GOOGLE_API_KEY,
                task_type="retrieval_document"
            )
            
            # Initialize chat model with Flash model for better rate limits
            self.chat_model = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-flash",  # Using Flash model for better rate limits
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7,
                convert_system_message_to_human=True
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=VECTOR_STORE_CHUNK_SIZE,
                chunk_overlap=VECTOR_STORE_CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info("Successfully initialized Langchain with Gemini embeddings and chat model")
        except Exception as e:
            logger.error(f"Error initializing Langchain: {str(e)}")
            raise

    def generate_response(self, query: str, context: str = None) -> str:
        """Generate a response using the chat model."""
        try:
            # Prepare the prompt
            if context:
                prompt = f"""Context: {context}

Question: {query}

Please provide a helpful response based on the context provided. If the context doesn't contain relevant information, say so."""
            else:
                prompt = query

            # Generate response
            response = self.chat_model.predict(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
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
                if emb_array.shape[0] != VECTOR_STORE_DIMENSION:
                    logger.warning(f"Unexpected embedding dimension: {emb_array.shape[0]}, expected {VECTOR_STORE_DIMENSION}")
                    # Reshape if possible, otherwise raise error
                    if emb_array.shape[0] == 768:
                        # Take first 384 dimensions if we got 768
                        emb_array = emb_array[:VECTOR_STORE_DIMENSION]
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
            if emb_array.shape[0] != VECTOR_STORE_DIMENSION:
                logger.warning(f"Unexpected embedding dimension: {emb_array.shape[0]}, expected {VECTOR_STORE_DIMENSION}")
                # Reshape if possible, otherwise raise error
                if emb_array.shape[0] == 768:
                    # Take first 384 dimensions if we got 768
                    emb_array = emb_array[:VECTOR_STORE_DIMENSION]
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