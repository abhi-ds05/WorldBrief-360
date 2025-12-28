"""
Briefing generation and retrieval endpoints.
"""
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.db.models import User, Topic, Briefing
from app.schemas import (
    BriefingRequest,
    BriefingResponse,
    BriefingLevel,
    PaginatedResponse,
)
from app.schemas.common import PaginationParams
from app.services.generators.briefing_generator import generate_briefing
from app.services.generators.tts_generator import generate_audio_briefing
from app.services.generators.image_generator import generate_briefing_images

router = APIRouter()


@router.post("/generate", response_model=BriefingResponse)
async def generate_new_briefing(
    briefing_request: BriefingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate a new briefing for a topic.
    """
    # Get topic
    topic = db.query(Topic).filter(Topic.id == briefing_request.topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    # Check if briefing already exists for this user/topic/level
    existing_briefing = db.query(Briefing).filter(
        Briefing.topic_id == briefing_request.topic_id,
        Briefing.user_id == current_user.id,
        Briefing.level == briefing_request.level,
    ).first()
    
    if existing_briefing and not briefing_request.force_regenerate:
        return BriefingResponse.from_orm(existing_briefing)
    
    # Generate briefing
    briefing_content = await generate_briefing(
        topic=topic,
        level=briefing_request.level,
        db=db
    )
    
    # Create briefing record
    briefing = Briefing(
        topic_id=topic.id,
        user_id=current_user.id,
        level=briefing_request.level,
        title=briefing_content.get("title", topic.title),
        summary=briefing_content.get("summary", ""),
        background=briefing_content.get("background", ""),
        current_status=briefing_content.get("current_status", ""),
        key_stakeholders=briefing_content.get("key_stakeholders", ""),
        risks_predictions=briefing_content.get("risks_predictions", ""),
        related_events=briefing_content.get("related_events", ""),
        generation_type="manual",
    )
    
    db.add(briefing)
    db.commit()
    db.refresh(briefing)
    
    # Generate audio and images in background
    if briefing_request.generate_audio:
        background_tasks.add_task(
            generate_audio_briefing,
            briefing_id=briefing.id,
            briefing_text=briefing.get_full_text()
        )
    
    if briefing_request.generate_images:
        background_tasks.add_task(
            generate_briefing_images,
            briefing_id=briefing.id,
            briefing_text=briefing.get_full_text(),
            topic_title=topic.title
        )
    
    return BriefingResponse.from_orm(briefing)


@router.get("/{briefing_id}", response_model=BriefingResponse)
async def get_briefing(
    briefing_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Get a specific briefing by ID.
    """
    briefing = db.query(Briefing).filter(Briefing.id == briefing_id).first()
    if not briefing:
        raise HTTPException(status_code=404, detail="Briefing not found")
    
    # Check permissions (user can see their own briefings or public ones)
    if briefing.user_id != current_user.id and not briefing.is_public:
        raise HTTPException(status_code=403, detail="Not authorized to view this briefing")
    
    return BriefingResponse.from_orm(briefing)


@router.get("/topic/{topic_id}", response_model=List[BriefingResponse])
async def get_topic_briefings(
    topic_id: int,
    level: Optional[BriefingLevel] = Query(None, description="Filter by briefing level"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get all briefings for a specific topic.
    """
    query = db.query(Briefing).filter(
        Briefing.topic_id == topic_id,
        Briefing.user_id == current_user.id,
    )
    
    if level:
        query = query.filter(Briefing.level == level)
    
    briefings = query.order_by(Briefing.created_at.desc()).all()
    
    return [BriefingResponse.from_orm(b) for b in briefings]


@router.get("/user/my-briefings", response_model=PaginatedResponse[BriefingResponse])
async def get_my_briefings(
    pagination: PaginationParams = Depends(),
    level: Optional[BriefingLevel] = Query(None, description="Filter by briefing level"),
    topic_id: Optional[int] = Query(None, description="Filter by topic"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get all briefings for the current user.
    """
    query = db.query(Briefing).filter(Briefing.user_id == current_user.id)
    
    if level:
        query = query.filter(Briefing.level == level)
    
    if topic_id:
        query = query.filter(Briefing.topic_id == topic_id)
    
    total = query.count()
    briefings = query.order_by(Briefing.created_at.desc())\
                     .offset(pagination.offset)\
                     .limit(pagination.limit)\
                     .all()
    
    return PaginatedResponse(
        items=[BriefingResponse.from_orm(b) for b in briefings],
        total=total,
        page=pagination.page,
        per_page=pagination.limit,
    )


@router.post("/{briefing_id}/audio/generate")
async def generate_briefing_audio(
    briefing_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate audio for a briefing.
    """
    briefing = db.query(Briefing).filter(
        Briefing.id == briefing_id,
        Briefing.user_id == current_user.id,
    ).first()
    
    if not briefing:
        raise HTTPException(status_code=404, detail="Briefing not found")
    
    # Check if audio already exists
    if briefing.audio_path:
        raise HTTPException(
            status_code=400, 
            detail="Audio already generated for this briefing"
        )
    
    # Generate audio in background
    background_tasks.add_task(
        generate_audio_briefing,
        briefing_id=briefing.id,
        briefing_text=briefing.get_full_text()
    )
    
    return {"message": "Audio generation started", "briefing_id": briefing_id}


@router.post("/{briefing_id}/images/generate")
async def generate_briefing_images_endpoint(
    briefing_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate images for a briefing.
    """
    briefing = db.query(Briefing).filter(
        Briefing.id == briefing_id,
        Briefing.user_id == current_user.id,
    ).first()
    
    if not briefing:
        raise HTTPException(status_code=404, detail="Briefing not found")
    
    topic = db.query(Topic).filter(Topic.id == briefing.topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    # Generate images in background
    background_tasks.add_task(
        generate_briefing_images,
        briefing_id=briefing.id,
        briefing_text=briefing.get_full_text(),
        topic_title=topic.title
    )
    
    return {"message": "Image generation started", "briefing_id": briefing_id}


@router.delete("/{briefing_id}")
async def delete_briefing(
    briefing_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a briefing.
    """
    briefing = db.query(Briefing).filter(
        Briefing.id == briefing_id,
        Briefing.user_id == current_user.id,
    ).first()
    
    if not briefing:
        raise HTTPException(status_code=404, detail="Briefing not found")
    
    # TODO: Delete associated files (audio, images) from storage
    
    db.delete(briefing)
    db.commit()
    
    return {"message": "Briefing deleted successfully"}


@router.put("/{briefing_id}/visibility")
async def update_briefing_visibility(
    briefing_id: int,
    is_public: bool,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update briefing visibility (public/private).
    """
    briefing = db.query(Briefing).filter(
        Briefing.id == briefing_id,
        Briefing.user_id == current_user.id,
    ).first()
    
    if not briefing:
        raise HTTPException(status_code=404, detail="Briefing not found")
    
    briefing.is_public = is_public
    briefing.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Briefing visibility updated", "is_public": is_public}