from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import List, Optional
from ..database import get_db
from ..models import Conversation, Message, MessageType
from ..schemas import (
    ConversationCreate,
    Conversation as ConversationSchema,
    MessageCreate,
    Message as MessageSchema,
    ConversationList,
    ChatResponse
)
from ..utils.langchain_utils import langchain_manager
from ..utils.gemini_embeddings import get_query_embedding
from ..db.vector_store import vector_store
import logging
import re

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

def get_user_id(x_user_id: str = Header(...)) -> str:
    return x_user_id

@router.post("/conversations", response_model=ConversationSchema)
async def create_conversation(
    conversation: ConversationCreate,
    db: Session = Depends(get_db)
):
    """Create a new conversation."""
    try:
        db_conversation = Conversation(
            user_id=conversation.user_id,
            title=conversation.title,
            file_id=conversation.file_id
        )
        db.add(db_conversation)
        db.commit()
        db.refresh(db_conversation)
        return db_conversation
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations", response_model=ConversationList)
async def list_conversations(
    user_id: str = Depends(get_user_id),
    file_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """List all conversations for a user, optionally filtered by file_id."""
    try:
        query = db.query(Conversation).filter(Conversation.user_id == user_id)
        if file_id is not None:
            query = query.filter(Conversation.file_id == file_id)
        conversations = query.order_by(Conversation.updated_at.desc()).all()
        return ConversationList(conversations=conversations)
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/by-file/{file_id}", response_model=ConversationSchema)
async def get_conversation_by_file(
    file_id: int,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    """Get a conversation for a user and file."""
    try:
        conversation = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.file_id == file_id
        ).order_by(Conversation.created_at.desc()).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except Exception as e:
        logger.error(f"Error getting conversation by file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/{conversation_id}", response_model=ConversationSchema)
async def get_conversation(
    conversation_id: int,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    """Get a specific conversation with its messages."""
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/conversations/{conversation_id}/messages", response_model=ChatResponse)
async def create_message(
    conversation_id: int,
    message: MessageCreate,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    """Create a new message in a conversation and get AI response."""
    try:
        # Verify conversation exists and belongs to user
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Store user message
        user_message = Message(
            conversation_id=conversation_id,
            content=message.content,
            message_type=MessageType.USER,
            referenced_documents=message.referenced_documents
        )
        db.add(user_message)
        db.flush()

        # Generate AI response using RAG
        try:
            # Get relevant documents from vector store
            query_embedding = get_query_embedding(message.content)
            search_results = vector_store.search_similar(
                query_embedding=query_embedding,
                n_results=10  # Get more results to allow filtering
            )

            # Filter search results to only include chunks from the referenced file(s)
            referenced_file_ids = set(message.referenced_documents or [])
            filtered_results = {
                "documents": [],
                "metadatas": [],
                "distances": []
            }
            for doc, meta, dist in zip(search_results["documents"], search_results["metadatas"], search_results["distances"]):
                if str(meta.get("file_id")) in {str(fid) for fid in referenced_file_ids}:
                    filtered_results["documents"].append(doc)
                    filtered_results["metadatas"].append(meta)
                    filtered_results["distances"].append(dist)

            # Use filtered results for context
            context = "\n".join(filtered_results["documents"])
            
            # Generate AI response using Langchain
            ai_response = langchain_manager.generate_response(
                message.content,
                context=context
            )
            ai_response = format_ai_response(ai_response)

            # Store AI message with referenced document IDs from metadata
            ai_message = Message(
                conversation_id=conversation_id,
                content=ai_response,
                message_type=MessageType.AI,
                referenced_documents=[meta.get("id", 0) for meta in filtered_results["metadatas"]]
            )
            db.add(ai_message)
            db.commit()
            db.refresh(conversation)

            return ChatResponse(
                message=ai_message,
                conversation=conversation
            )

        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            # If AI response generation fails, still save the user message
            db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Error generating AI response: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating message: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db)
):
    """Delete a conversation and all its messages."""
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        db.delete(conversation)
        db.commit()
        return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def format_ai_response(text: str) -> str:
    # Replace markdown bold with nothing or with plain text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Replace triple stars (***Section***) with section headers
    text = re.sub(r'\n?\*{3}(.*?)\*{3}', r'\n\n\1:', text)
    # Replace single or double stars at the start of a line with a bullet
    text = re.sub(r'(\n|^)\*{1,2}', r'\n• ', text)
    # Remove extra spaces at the start of lines
    text = re.sub(r'\n +', '\n', text)
    # Ensure each bullet is on its own line
    text = re.sub(r'• ', r'\n• ', text)
    # Remove duplicate newlines
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip() 