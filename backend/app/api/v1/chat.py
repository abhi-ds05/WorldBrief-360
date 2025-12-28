"""
AI Chat endpoints for interactive Q&A about topics.
"""
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user, get_current_user_ws
from app.db.models import User, Topic, ChatMessage
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessageResponse,
    ChatHistoryResponse,
)
from app.services.chat.qa_engine import get_chat_response
from app.services.chat.chat_history import save_chat_message, get_chat_history

router = APIRouter()


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Ask a question about a topic and get AI response.
    """
    # Validate topic if provided
    topic = None
    if chat_request.topic_id:
        topic = db.query(Topic).filter(Topic.id == chat_request.topic_id).first()
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
    
    # Get AI response
    response = await get_chat_response(
        question=chat_request.question,
        topic=topic,
        db=db,
        user_id=current_user.id
    )
    
    # Save question and response to chat history
    question_message = ChatMessage(
        user_id=current_user.id,
        topic_id=chat_request.topic_id,
        message=chat_request.question,
        is_user=True,
        sources=[],
    )
    db.add(question_message)
    db.commit()
    db.refresh(question_message)
    
    response_message = ChatMessage(
        user_id=current_user.id,
        topic_id=chat_request.topic_id,
        message=response.answer,
        is_user=False,
        sources=response.sources,
    )
    db.add(response_message)
    db.commit()
    
    return ChatResponse(
        question=chat_request.question,
        answer=response.answer,
        sources=response.sources,
        confidence=response.confidence,
        suggested_followups=response.suggested_followups,
    )


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history_endpoint(
    topic_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get chat history for the current user.
    """
    messages = get_chat_history(
        user_id=current_user.id,
        topic_id=topic_id,
        limit=limit,
        offset=offset,
        db=db
    )
    
    total = db.query(ChatMessage).filter(
        ChatMessage.user_id == current_user.id,
        ChatMessage.topic_id == topic_id if topic_id else True
    ).count()
    
    return ChatHistoryResponse(
        messages=[
            ChatMessageResponse.from_orm(msg) for msg in messages
        ],
        total=total,
        has_more=(offset + len(messages)) < total,
    )


@router.delete("/history")
async def clear_chat_history(
    topic_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Clear chat history for the current user.
    """
    query = db.query(ChatMessage).filter(ChatMessage.user_id == current_user.id)
    
    if topic_id:
        query = query.filter(ChatMessage.topic_id == topic_id)
    
    deleted_count = query.delete()
    db.commit()
    
    return {"message": f"Deleted {deleted_count} chat messages"}


@router.websocket("/ws")
async def websocket_chat(
    websocket: WebSocket,
    topic_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """
    WebSocket endpoint for real-time chat.
    """
    await websocket.accept()
    
    try:
        # Authenticate user
        user = await get_current_user_ws(websocket, db)
        if not user:
            await websocket.close(code=1008)
            return
        
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to chat",
            "user_id": user.id,
            "username": user.username,
        })
        
        # Main chat loop
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                question = data.get("question", "").strip()
                
                if not question:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Question cannot be empty"
                    })
                    continue
                
                # Validate topic if provided
                topic = None
                if topic_id:
                    topic = db.query(Topic).filter(Topic.id == topic_id).first()
                    if not topic:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Topic not found"
                        })
                        continue
                
                # Save user question
                question_message = ChatMessage(
                    user_id=user.id,
                    topic_id=topic_id,
                    message=question,
                    is_user=True,
                    sources=[],
                )
                db.add(question_message)
                db.commit()
                db.refresh(question_message)
                
                # Send typing indicator
                await websocket.send_json({
                    "type": "typing",
                    "is_typing": True
                })
                
                # Get AI response
                response = await get_chat_response(
                    question=question,
                    topic=topic,
                    db=db,
                    user_id=user.id
                )
                
                # Save AI response
                response_message = ChatMessage(
                    user_id=user.id,
                    topic_id=topic_id,
                    message=response.answer,
                    is_user=False,
                    sources=response.sources,
                )
                db.add(response_message)
                db.commit()
                
                # Send response
                await websocket.send_json({
                    "type": "typing",
                    "is_typing": False
                })
                
                await websocket.send_json({
                    "type": "response",
                    "answer": response.answer,
                    "sources": response.sources,
                    "confidence": response.confidence,
                    "suggested_followups": response.suggested_followups,
                    "message_id": response_message.id,
                })
            
            elif data.get("type") == "get_history":
                # Get chat history
                messages = get_chat_history(
                    user_id=user.id,
                    topic_id=topic_id,
                    limit=50,
                    offset=0,
                    db=db
                )
                
                await websocket.send_json({
                    "type": "history",
                    "messages": [
                        {
                            "id": msg.id,
                            "message": msg.message,
                            "is_user": msg.is_user,
                            "created_at": msg.created_at.isoformat(),
                            "sources": msg.sources,
                        }
                        for msg in messages
                    ]
                })
            
            elif data.get("type") == "clear_history":
                # Clear chat history
                query = db.query(ChatMessage).filter(ChatMessage.user_id == user.id)
                if topic_id:
                    query = query.filter(ChatMessage.topic_id == topic_id)
                
                deleted_count = query.delete()
                db.commit()
                
                await websocket.send_json({
                    "type": "history_cleared",
                    "message": f"Cleared {deleted_count} messages",
                })
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected")
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011)


@router.get("/topics/suggested", response_model=List[str])
async def get_suggested_topics(
    query: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Get suggested topics for chat based on user interests or search query.
    """
    from sqlalchemy import or_
    
    if query:
        # Search topics
        topics = db.query(Topic).filter(
            or_(
                Topic.title.ilike(f"%{query}%"),
                Topic.description.ilike(f"%{query}%"),
                Topic.category.ilike(f"%{query}%"),
            )
        ).limit(limit).all()
    else:
        # Get trending topics or user's recent topics
        if current_user:
            # Get topics from user's recent briefings
            from app.db.models import Briefing
            topics = db.query(Topic).join(
                Briefing, Briefing.topic_id == Topic.id
            ).filter(
                Briefing.user_id == current_user.id
            ).order_by(
                Briefing.created_at.desc()
            ).limit(limit).all()
        else:
            # Get trending topics
            topics = db.query(Topic).filter(
                Topic.is_active == True
            ).order_by(
                Topic.updated_at.desc()
            ).limit(limit).all()
    
    return [topic.title for topic in topics]


@router.post("/feedback")
async def submit_chat_feedback(
    message_id: int,
    is_helpful: bool,
    feedback_text: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Submit feedback for a chat response.
    """
    message = db.query(ChatMessage).filter(
        ChatMessage.id == message_id,
        ChatMessage.user_id == current_user.id,
        ChatMessage.is_user == False,  # Only rate AI responses
    ).first()
    
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Update feedback
    message.feedback_is_helpful = is_helpful
    message.feedback_text = feedback_text
    message.feedback_at = datetime.utcnow()
    db.commit()
    
    # Update user reputation for providing feedback
    if current_user.reputation_score < 1000:  # Cap reputation
        current_user.reputation_score += 1
        db.commit()
    
    return {"message": "Feedback submitted successfully"}