"""
Admin-only endpoints for system management and moderation.
"""
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user, require_admin_role
from app.db.models import User, Incident, Topic, Article, Transaction
from app.schemas import (
    UserAdminResponse,
    IncidentAdminResponse,
    TopicAdminResponse,
    SystemStatsResponse,
    PaginatedResponse,
    AdminActionRequest,
    ModerationAction,
)
from app.schemas.common import PaginationParams

router = APIRouter()


@router.get("/users", response_model=PaginatedResponse[UserAdminResponse])
async def list_users(
    pagination: PaginationParams = Depends(),
    search: Optional[str] = Query(None, description="Search by email or username"),
    role: Optional[str] = Query(None, description="Filter by role"),
    active: Optional[bool] = Query(None, description="Filter by active status"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
):
    """
    List all users with admin view.
    """
    from sqlalchemy import or_

    query = db.query(User)

    if search:
        query = query.filter(
            or_(
                User.email.ilike(f"%{search}%"),
                User.username.ilike(f"%{search}%"),
            )
        )

    if role:
        query = query.filter(User.role == role)
    
    if active is not None:
        query = query.filter(User.is_active == active)

    total = query.count()
    users = query.order_by(User.created_at.desc())\
                .offset(pagination.offset)\
                .limit(pagination.limit)\
                .all()

    return PaginatedResponse(
        items=users,
        total=total,
        page=pagination.page,
        per_page=pagination.limit,
    )


@router.get("/incidents", response_model=PaginatedResponse[IncidentAdminResponse])
async def list_incidents(
    pagination: PaginationParams = Depends(),
    status: Optional[str] = Query(None, description="Filter by verification status"),
    category: Optional[str] = Query(None, description="Filter by incident category"),
    reporter_id: Optional[int] = Query(None, description="Filter by reporter"),
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
):
    """
    List all incidents with admin view.
    """
    query = db.query(Incident)

    if status:
        query = query.filter(Incident.verification_status == status)
    
    if category:
        query = query.filter(Incident.category == category)
    
    if reporter_id:
        query = query.filter(Incident.reporter_id == reporter_id)
    
    if start_date:
        query = query.filter(Incident.created_at >= start_date)
    
    if end_date:
        query = query.filter(Incident.created_at <= end_date)

    total = query.count()
    incidents = query.order_by(Incident.created_at.desc())\
                     .offset(pagination.offset)\
                     .limit(pagination.limit)\
                     .all()

    return PaginatedResponse(
        items=incidents,
        total=total,
        page=pagination.page,
        per_page=pagination.limit,
    )


@router.post("/incidents/{incident_id}/moderate")
async def moderate_incident(
    incident_id: int,
    action: AdminActionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
):
    """
    Take moderation action on an incident.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    if action.action == ModerationAction.VERIFY:
        incident.verification_status = "verified"
        incident.verified_at = datetime.utcnow()
        incident.verified_by = current_user.id
        
        # Award coins to reporter if not already awarded
        if not incident.coins_awarded:
            from app.services.community.wallet_service import award_incident_coins
            award_incident_coins(db, incident.reporter_id, incident.id)
            incident.coins_awarded = True
    
    elif action.action == ModerationAction.REJECT:
        incident.verification_status = "rejected"
        incident.rejected_at = datetime.utcnow()
        incident.rejected_by = current_user.id
        incident.rejection_reason = action.reason
    
    elif action.action == ModerationAction.DELETE:
        incident.is_deleted = True
        incident.deleted_at = datetime.utcnow()
        incident.deleted_by = current_user.id
        incident.deletion_reason = action.reason

    db.commit()
    
    return {"message": f"Incident {action.action} successfully", "incident_id": incident_id}


@router.get("/topics", response_model=PaginatedResponse[TopicAdminResponse])
async def list_topics(
    pagination: PaginationParams = Depends(),
    search: Optional[str] = Query(None, description="Search by title or description"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
):
    """
    List all topics with admin view.
    """
    from sqlalchemy import or_

    query = db.query(Topic)

    if search:
        query = query.filter(
            or_(
                Topic.title.ilike(f"%{search}%"),
                Topic.description.ilike(f"%{search}%"),
            )
        )

    total = query.count()
    topics = query.order_by(Topic.created_at.desc())\
                  .offset(pagination.offset)\
                  .limit(pagination.limit)\
                  .all()

    return PaginatedResponse(
        items=topics,
        total=total,
        page=pagination.page,
        per_page=pagination.limit,
    )


@router.get("/stats/system", response_model=SystemStatsResponse)
async def get_system_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
):
    """
    Get comprehensive system statistics.
    """
    from sqlalchemy import func, and_
    from datetime import datetime, timedelta

    # User stats
    total_users = db.query(func.count(User.id)).scalar()
    active_users_24h = db.query(func.count(User.id)).filter(
        User.last_active >= datetime.utcnow() - timedelta(hours=24)
    ).scalar()
    new_users_today = db.query(func.count(User.id)).filter(
        User.created_at >= datetime.utcnow().date()
    ).scalar()

    # Incident stats
    total_incidents = db.query(func.count(Incident.id)).scalar()
    verified_incidents = db.query(func.count(Incident.id)).filter(
        Incident.verification_status == "verified"
    ).scalar()
    pending_incidents = db.query(func.count(Incident.id)).filter(
        Incident.verification_status == "pending"
    ).scalar()

    # Topic stats
    total_topics = db.query(func.count(Topic.id)).scalar()
    active_topics = db.query(func.count(Topic.id)).filter(
        Topic.is_active == True
    ).scalar()

    # Transaction stats
    total_coins_awarded = db.query(func.sum(Transaction.amount)).filter(
        Transaction.transaction_type == "reward"
    ).scalar() or 0

    total_coins_spent = db.query(func.sum(Transaction.amount)).filter(
        Transaction.transaction_type == "purchase"
    ).scalar() or 0

    # Recent activity
    recent_incidents_24h = db.query(func.count(Incident.id)).filter(
        Incident.created_at >= datetime.utcnow() - timedelta(hours=24)
    ).scalar()

    return SystemStatsResponse(
        users={
            "total": total_users,
            "active_24h": active_users_24h,
            "new_today": new_users_today,
        },
        incidents={
            "total": total_incidents,
            "verified": verified_incidents,
            "pending": pending_incidents,
            "recent_24h": recent_incidents_24h,
        },
        topics={
            "total": total_topics,
            "active": active_topics,
        },
        economy={
            "total_coins_awarded": total_coins_awarded,
            "total_coins_spent": total_coins_spent,
            "coins_in_circulation": total_coins_awarded - total_coins_spent,
        },
        generated_at=datetime.utcnow(),
    )


@router.post("/users/{user_id}/actions")
async def manage_user(
    user_id: int,
    action: AdminActionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
):
    """
    Take administrative action on a user.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot perform action on yourself")

    if action.action == ModerationAction.SUSPEND:
        user.is_active = False
        user.suspended_at = datetime.utcnow()
        user.suspended_by = current_user.id
        user.suspension_reason = action.reason
    
    elif action.action == ModerationAction.REACTIVATE:
        user.is_active = True
        user.reactivated_at = datetime.utcnow()
        user.reactivated_by = current_user.id
    
    elif action.action == ModerationAction.DELETE:
        # Soft delete
        user.is_deleted = True
        user.deleted_at = datetime.utcnow()
        user.deleted_by = current_user.id
        user.deletion_reason = action.reason

    db.commit()

    return {"message": f"User {action.action} successfully", "user_id": user_id}