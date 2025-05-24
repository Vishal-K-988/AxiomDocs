import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Langchain Configuration
LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = "axiom-docs"  # Your project name

# Vector Store Configuration
VECTOR_STORE_DIMENSION = 384  # Gemini's embedding dimension
VECTOR_STORE_CHUNK_SIZE = 1000
VECTOR_STORE_CHUNK_OVERLAP = 200

# Ensure required API keys are set
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

if not LANGCHAIN_API_KEY:
    raise ValueError("LANGCHAIN_API_KEY environment variable is not set") 