from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import files, chat
from .api.endpoints import vector_store
from .database import engine
from .models import Base
from app.api import pdf_routes

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="File Management System",
    description="A sophisticated file management system with AI capabilities",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include routers
app.include_router(files.router)
app.include_router(chat.router)
app.include_router(vector_store.router, prefix="/api/vector-store", tags=["vector-store"])
app.include_router(pdf_routes.router, prefix="/api/pdf", tags=["pdf"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to the PDF Vector Store API"} 