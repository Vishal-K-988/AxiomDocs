from fastapi import APIRouter
from app.api.endpoints import auth, files, vector_store

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(vector_store.router, prefix="/vector-store", tags=["vector-store"]) 