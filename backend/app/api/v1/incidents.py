"""
Incident reporting and management endpoints.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import geopy.distance

from app.core.database import get_db
from app.core.security import get_current_user
from app.db.models import User, Incident, IncidentImage, Vote, Comment
from app.schemas import (
    IncidentCreate,
    IncidentResponse,
    IncidentUpdate,
    IncidentListResponse,
    IncidentVerificationRequest,
    VoteCreate,
    CommentCreate,
    CommentResponse,
    PaginatedResponse,
)
from app.schemas.common import PaginationParams
from app.services.community.incident_service import (
    create_incident,
    update_incident,
    verify_incident,
    calculate_confidence_score,
)
from app.services.community.reputation_service import update_user_reputation
from app.services.utils.file_manager import save_uploaded_file
from app.services.utils.geo import validate_coordinates, calculate_distance

router = APIRouter()


@router.post("/", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED)
async def report_incident(
    title: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    severity: str = Form("medium"),
    images: Optional[List[UploadFile]] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> IncidentResponse:
    """
    Report a new incident.
    """
    # Validate coordinates
    if not validate_coordinates(latitude, longitude):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid coordinates"
        )
    
    # Validate category
    valid_categories = ["accident", "fire", "flood", "protest", "crime", "hazard", "other"]
    if category not in valid_categories:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
        )
    
    # Validate severity
    valid_severities = ["low", "medium", "high", "critical"]
    if severity not in valid_severities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid severity. Must be one of: {', '.join(valid_severities)}"
        )
    
    # Create incident
    incident_data = IncidentCreate(
        title=title,
        description=description,
        category=category,
        latitude=latitude,
        longitude=longitude,
        severity=severity,
        reporter_id=current_user.id,
    )
    
    incident = await create_incident(incident_data, db)
    
    # Save uploaded images
    if images:
        for image_file in images:
            # Validate image
            if not image_file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only image files are allowed"
                )
            
            # Save image
            image_path = await save_uploaded_file(
                file=image_file,
                user_id=current_user.id,
                prefix=f"incident_{incident.id}"
            )
            
            # Create image record
            incident_image = IncidentImage(
                incident_id=incident.id,
                image_path=image_path,
                uploaded_by=current_user.id,
                is_primary=False,  # First image is primary
            )
            db.add(incident_image)
        
        db.commit()
        
        # Set first image as primary
        if incident.images:
            incident.images[0].is_primary = True
            db.commit()
    
    # Update user reputation for reporting
    await update_user_reputation(current_user.id, "incident_report", db)
    
    return IncidentResponse.from_orm(incident)


@router.get("/", response_model=PaginatedResponse[IncidentResponse])
async def list_incidents(
    pagination: PaginationParams = Depends(),
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by verification status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    latitude: Optional[float] = Query(None, description="Center latitude for location-based search"),
    longitude: Optional[float] = Query(None, description="Center longitude for location-based search"),
    radius_km: Optional[float] = Query(10.0, description="Search radius in kilometers"),
    timeframe_hours: Optional[int] = Query(None, description="Filter by time (last N hours)"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> PaginatedResponse[IncidentResponse]:
    """
    List incidents with filtering and pagination.
    """
    query = db.query(Incident).filter(Incident.is_deleted == False)
    
    # Apply filters
    if category:
        query = query.filter(Incident.category == category)
    
    if status:
        query = query.filter(Incident.verification_status == status)
    
    if severity:
        query = query.filter(Incident.severity == severity)
    
    if timeframe_hours:
        timeframe = datetime.utcnow() - timedelta(hours=timeframe_hours)
        query = query.filter(Incident.created_at >= timeframe)
    
    # Location-based filtering
    if latitude and longitude and radius_km:
        # Simple bounding box filter (could be optimized with PostGIS)
        # For now, we'll fetch all and filter in Python for small datasets
        # In production, use PostGIS or Redis Geo
        all_incidents = query.all()
        filtered_incidents = []
        
        for incident in all_incidents:
            distance = calculate_distance(
                (latitude, longitude),
                (incident.latitude, incident.longitude)
            )
            if distance <= radius_km:
                filtered_incidents.append(incident)
        
        # Apply pagination manually
        total = len(filtered_incidents)
        incidents = filtered_incidents[pagination.offset:pagination.offset + pagination.limit]
    else:
        # Normal pagination
        total = query.count()
        query = query.order_by(Incident.created_at.desc())
        incidents = query.offset(pagination.offset).limit(pagination.limit).all()
    
    return PaginatedResponse(
        items=[IncidentResponse.from_orm(incident) for incident in incidents],
        total=total,
        page=pagination.page,
        per_page=pagination.limit,
    )


@router.get("/nearby", response_model=List[IncidentResponse])
async def get_nearby_incidents(
    latitude: float = Query(..., description="Current latitude"),
    longitude: float = Query(..., description="Current longitude"),
    radius_km: float = Query(5.0, description="Search radius in kilometers"),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> List[IncidentResponse]:
    """
    Get incidents near a specific location.
    """
    # Validate coordinates
    if not validate_coordinates(latitude, longitude):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid coordinates"
        )
    
    # Get all incidents (this is inefficient for large datasets)
    # In production, use PostGIS or Redis Geo for spatial queries
    all_incidents = db.query(Incident).filter(
        Incident.is_deleted == False,
        Incident.verification_status.in_(["verified", "pending"])
    ).all()
    
    # Filter by distance
    nearby_incidents = []
    for incident in all_incidents:
        distance = calculate_distance(
            (latitude, longitude),
            (incident.latitude, incident.longitude)
        )
        if distance <= radius_km:
            incident.distance_km = distance  # Add distance attribute
            nearby_incidents.append(incident)
    
    # Sort by distance and limit
    nearby_incidents.sort(key=lambda x: x.distance_km)
    nearby_incidents = nearby_incidents[:limit]
    
    return [IncidentResponse.from_orm(incident) for incident in nearby_incidents]


@router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> IncidentResponse:
    """
    Get a specific incident by ID.
    """
    incident = db.query(Incident).filter(
        Incident.id == incident_id,
        Incident.is_deleted == False
    ).first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Increment view count
    incident.views += 1
    db.commit()
    
    return IncidentResponse.from_orm(incident)


@router.put("/{incident_id}", response_model=IncidentResponse)
async def update_incident_details(
    incident_id: int,
    incident_update: IncidentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> IncidentResponse:
    """
    Update an incident (only by reporter or admin).
    """
    incident = db.query(Incident).filter(
        Incident.id == incident_id,
        Incident.is_deleted == False
    ).first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Check permissions
    if incident.reporter_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this incident"
        )
    
    # Update incident
    updated_incident = await update_incident(incident_id, incident_update, db)
    
    return IncidentResponse.from_orm(updated_incident)


@router.delete("/{incident_id}")
async def delete_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Delete an incident (soft delete).
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Check permissions
    if incident.reporter_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this incident"
        )
    
    # Soft delete
    incident.is_deleted = True
    incident.deleted_at = datetime.utcnow()
    incident.deleted_by = current_user.id
    db.commit()
    
    return {"message": "Incident deleted successfully"}


@router.post("/{incident_id}/verify", response_model=IncidentResponse)
async def request_verification(
    incident_id: int,
    verification_request: IncidentVerificationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> IncidentResponse:
    """
    Request verification for an incident.
    """
    incident = db.query(Incident).filter(
        Incident.id == incident_id,
        Incident.is_deleted == False
    ).first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Check if user can verify (reporter or admin)
    if incident.reporter_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to verify this incident"
        )
    
    # Verify incident
    verified_incident = await verify_incident(
        incident_id=incident_id,
        verification_request=verification_request,
        db=db,
        user_id=current_user.id
    )
    
    # Award coins for verification
    if verified_incident.verification_status == "verified":
        from app.services.community.wallet_service import award_incident_coins
        await award_incident_coins(incident.reporter_id, incident_id, db)
    
    return IncidentResponse.from_orm(verified_incident)


@router.post("/{incident_id}/vote", response_model=Dict[str, Any])
async def vote_on_incident(
    incident_id: int,
    vote_data: VoteCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Vote on an incident (upvote/downvote).
    """
    incident = db.query(Incident).filter(
        Incident.id == incident_id,
        Incident.is_deleted == False
    ).first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Check if user has already voted
    existing_vote = db.query(Vote).filter(
        Vote.incident_id == incident_id,
        Vote.user_id == current_user.id
    ).first()
    
    if existing_vote:
        # Update existing vote
        existing_vote.vote_type = vote_data.vote_type
        existing_vote.updated_at = datetime.utcnow()
    else:
        # Create new vote
        vote = Vote(
            incident_id=incident_id,
            user_id=current_user.id,
            vote_type=vote_data.vote_type,
        )
        db.add(vote)
    
    db.commit()
    
    # Recalculate confidence score
    confidence_score = calculate_confidence_score(incident_id, db)
    incident.confidence_score = confidence_score
    db.commit()
    
    # Update user reputation for voting
    await update_user_reputation(current_user.id, "incident_vote", db)
    
    # Get updated vote counts
    upvotes = db.query(Vote).filter(
        Vote.incident_id == incident_id,
        Vote.vote_type == "upvote"
    ).count()
    
    downvotes = db.query(Vote).filter(
        Vote.incident_id == incident_id,
        Vote.vote_type == "downvote"
    ).count()
    
    return {
        "message": "Vote recorded",
        "incident_id": incident_id,
        "upvotes": upvotes,
        "downvotes": downvotes,
        "confidence_score": confidence_score,
    }


@router.get("/{incident_id}/votes")
async def get_incident_votes(
    incident_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get vote statistics for an incident.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    upvotes = db.query(Vote).filter(
        Vote.incident_id == incident_id,
        Vote.vote_type == "upvote"
    ).count()
    
    downvotes = db.query(Vote).filter(
        Vote.incident_id == incident_id,
        Vote.vote_type == "downvote"
    ).count()
    
    user_vote = None
    if current_user:
        user_vote = db.query(Vote).filter(
            Vote.incident_id == incident_id,
            Vote.user_id == current_user.id
        ).first()
    
    return {
        "incident_id": incident_id,
        "upvotes": upvotes,
        "downvotes": downvotes,
        "total_votes": upvotes + downvotes,
        "user_vote": user_vote.vote_type if user_vote else None,
        "confidence_score": incident.confidence_score,
    }


@router.post("/{incident_id}/comments", response_model=CommentResponse)
async def add_comment(
    incident_id: int,
    comment_data: CommentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> CommentResponse:
    """
    Add a comment to an incident.
    """
    incident = db.query(Incident).filter(
        Incident.id == incident_id,
        Incident.is_deleted == False
    ).first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Create comment
    comment = Comment(
        incident_id=incident_id,
        user_id=current_user.id,
        content=comment_data.content,
        is_public=True,
    )
    
    db.add(comment)
    db.commit()
    db.refresh(comment)
    
    # Update user reputation for commenting
    await update_user_reputation(current_user.id, "incident_comment", db)
    
    return CommentResponse.from_orm(comment)


@router.get("/{incident_id}/comments", response_model=List[CommentResponse])
async def get_incident_comments(
    incident_id: int,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> List[CommentResponse]:
    """
    Get comments for an incident.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    comments = db.query(Comment).filter(
        Comment.incident_id == incident_id,
        Comment.is_deleted == False
    ).order_by(Comment.created_at.desc())\
     .offset(offset)\
     .limit(limit)\
     .all()
    
    return [CommentResponse.from_orm(comment) for comment in comments]


@router.get("/{incident_id}/similar")
async def get_similar_incidents(
    incident_id: int,
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> List[IncidentResponse]:
    """
    Get incidents similar to the specified one.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Find similar incidents by category and location
    similar = db.query(Incident).filter(
        Incident.id != incident_id,
        Incident.category == incident.category,
        Incident.is_deleted == False,
        Incident.verification_status.in_(["verified", "pending"])
    ).order_by(
        func.abs(Incident.latitude - incident.latitude) +
        func.abs(Incident.longitude - incident.longitude)
    ).limit(limit).all()
    
    return [IncidentResponse.from_orm(inc) for inc in similar]


@router.get("/categories/stats")
async def get_category_statistics(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get statistics for incident categories.
    """
    from datetime import datetime, timedelta
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get category counts
    category_counts = db.query(
        Incident.category,
        func.count(Incident.id).label("count")
    ).filter(
        Incident.created_at >= start_date,
        Incident.is_deleted == False
    ).group_by(Incident.category).all()
    
    # Get verification rates by category
    verification_stats = db.query(
        Incident.category,
        Incident.verification_status,
        func.count(Incident.id).label("count")
    ).filter(
        Incident.created_at >= start_date,
        Incident.is_deleted == False
    ).group_by(Incident.category, Incident.verification_status).all()
    
    # Organize verification stats
    verification_by_category = {}
    for category, status, count in verification_stats:
        if category not in verification_by_category:
            verification_by_category[category] = {}
        verification_by_category[category][status] = count
    
    return {
        "period_days": days,
        "categories": {
            cat: count for cat, count in category_counts
        },
        "verification_stats": verification_by_category,
        "total_incidents": sum(count for _, count in category_counts),
        "generated_at": datetime.utcnow().isoformat(),
    }