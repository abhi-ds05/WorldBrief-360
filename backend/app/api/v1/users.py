"""
User management endpoints.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func

from app.core.database import get_db
from app.core.security import get_current_user, require_admin_role, get_password_hash, verify_password
from app.db.models import User, Wallet, Incident, Briefing, Subscription, UserNotificationSettings, WalletTransaction
from app.schemas import (
    UserResponse,
    UserUpdate,
    UserProfileResponse,
    UserStatsResponse,
    UserSearchResponse,
    PasswordChange,
    PaginatedResponse,
    IncidentResponse,
)
from app.schemas.common import PaginationParams
from app.services.utils.email_client import send_account_update_email

router = APIRouter()


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserProfileResponse:
    """
    Get current user's full profile.
    """
    # Get user wallet
    wallet = db.query(Wallet).filter(Wallet.user_id == current_user.id).first()
    
    # Get user stats
    incident_count = db.query(Incident).filter(Incident.reporter_id == current_user.id).count()
    briefing_count = db.query(Briefing).filter(Briefing.user_id == current_user.id).count()
    subscription_count = db.query(Subscription).filter(
        Subscription.user_id == current_user.id,
        Subscription.is_active == True,
    ).count()
    
    # Get notification settings
    notification_settings = db.query(UserNotificationSettings).filter(
        UserNotificationSettings.user_id == current_user.id
    ).first()
    
    return UserProfileResponse(
        **UserResponse.from_orm(current_user).dict(),
        wallet_balance=wallet.balance if wallet else 0,
        incident_count=incident_count,
        briefing_count=briefing_count,
        subscription_count=subscription_count,
        notification_settings=notification_settings.to_dict() if notification_settings else None,
        last_active=current_user.last_active,
    )


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """
    Update current user's information.
    """
    # Check if email is being changed and if it's already taken
    if user_update.email and user_update.email != current_user.email:
        existing_user = db.query(User).filter(
            User.email == user_update.email,
            User.id != current_user.id,
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
    
    # Check if username is being changed and if it's already taken
    if user_update.username and user_update.username != current_user.username:
        existing_user = db.query(User).filter(
            User.username == user_update.username,
            User.id != current_user.id,
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Username already taken"
            )
    
    # Update user fields
    for field, value in user_update.dict(exclude_unset=True).items():
        if value is not None:  # Only update if value is provided
            setattr(current_user, field, value)
    
    current_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(current_user)
    
    # Send notification email if email was changed
    if user_update.email and user_update.email != current_user.email:
        background_tasks.add_task(
            send_account_update_email,
            current_user.email,
            current_user.username,
            "email_changed"
        )
    
    return UserResponse.from_orm(current_user)


@router.post("/me/password", response_model=Dict[str, str])
async def change_password(
    password_data: PasswordChange,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Change current user's password.
    """
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="Current password is incorrect"
        )
    
    # Update password
    current_user.hashed_password = get_password_hash(password_data.new_password)
    current_user.updated_at = datetime.utcnow()
    db.commit()
    
    # Send notification email
    background_tasks.add_task(
        send_account_update_email,
        current_user.email,
        current_user.username,
        "password_changed"
    )
    
    return {"message": "Password changed successfully"}


@router.delete("/me")
async def delete_current_user(
    password: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Delete current user's account (soft delete).
    """
    # Verify password
    if not verify_password(password, current_user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="Password is incorrect"
        )
    
    # Soft delete user
    current_user.is_active = False
    current_user.is_deleted = True
    current_user.deleted_at = datetime.utcnow()
    current_user.deletion_reason = "user_requested"
    current_user.updated_at = datetime.utcnow()
    
    # Deactivate all active sessions/tokens (in production)
    
    db.commit()
    
    # Send goodbye email
    background_tasks.add_task(
        send_account_update_email,
        current_user.email,
        current_user.username,
        "account_deleted"
    )
    
    return {"message": "Account deleted successfully"}


@router.get("/search", response_model=List[UserSearchResponse])
async def search_users(
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[UserSearchResponse]:
    """
    Search for users by username or email.
    """
    users = db.query(User).filter(
        and_(
            User.is_active == True,
            User.is_deleted == False,
            or_(
                User.username.ilike(f"%{query}%"),
                User.email.ilike(f"%{query}%"),
                User.full_name.ilike(f"%{query}%"),
            )
        )
    ).limit(limit).all()
    
    return [
        UserSearchResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            reputation_score=user.reputation_score,
            is_verified=user.email_verified,
            created_at=user.created_at,
        )
        for user in users
    ]


@router.get("/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserProfileResponse:
    """
    Get another user's public profile.
    """
    user = db.query(User).filter(
        User.id == user_id,
        User.is_active == True,
        User.is_deleted == False,
    ).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get public stats
    incident_count = db.query(Incident).filter(
        Incident.reporter_id == user_id,
        Incident.is_deleted == False,
        Incident.verification_status == "verified",  # Only show verified incidents
    ).count()
    
    # Get user rank based on reputation
    user_rank = db.query(func.count(User.id)).filter(
        User.reputation_score > user.reputation_score,
        User.is_active == True,
    ).scalar() + 1
    
    # Check if current user is following this user (if following feature exists)
    is_following = False
    # This would require a followers/following table
    
    return UserProfileResponse(
        **UserResponse.from_orm(user).dict(),
        wallet_balance=None,  # Don't show other users' wallet balance
        incident_count=incident_count,
        briefing_count=None,  # Don't show other users' briefing count
        subscription_count=None,  # Don't show other users' subscription count
        notification_settings=None,  # Don't show other users' notification settings
        user_rank=user_rank,
        is_following=is_following,
        last_active=None if user.id != current_user.id else user.last_active,
    )


@router.get("/{user_id}/stats", response_model=UserStatsResponse)
async def get_user_stats(
    user_id: int,
    timeframe_days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> UserStatsResponse:
    """
    Get statistics for a user.
    """
    user = db.query(User).filter(
        User.id == user_id,
        User.is_active == True,
    ).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check permissions (users can only see their own detailed stats)
    if user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Not authorized to view detailed stats for this user"
        )
    
    start_date = datetime.utcnow() - timedelta(days=timeframe_days)
    
    # Incident statistics
    total_incidents = db.query(Incident).filter(
        Incident.reporter_id == user_id,
        Incident.is_deleted == False,
    ).count()
    
    verified_incidents = db.query(Incident).filter(
        Incident.reporter_id == user_id,
        Incident.verification_status == "verified",
        Incident.is_deleted == False,
    ).count()
    
    recent_incidents = db.query(Incident).filter(
        Incident.reporter_id == user_id,
        Incident.created_at >= start_date,
        Incident.is_deleted == False,
    ).count()
    
    # Briefing statistics
    total_briefings = db.query(Briefing).filter(
        Briefing.user_id == user_id,
    ).count()
    
    recent_briefings = db.query(Briefing).filter(
        Briefing.user_id == user_id,
        Briefing.created_at >= start_date,
    ).count()
    
    # Wallet statistics
    wallet = db.query(Wallet).filter(Wallet.user_id == user_id).first()
    wallet_balance = wallet.balance if wallet else 0
    
    total_earned = db.query(func.sum(WalletTransaction.amount)).filter(
        WalletTransaction.user_id == user_id,
        WalletTransaction.transaction_type == "reward",
    ).scalar() or 0
    
    total_spent = db.query(func.sum(WalletTransaction.amount)).filter(
        WalletTransaction.user_id == user_id,
        WalletTransaction.transaction_type == "purchase",
    ).scalar() or 0
    
    # Activity timeline (last 30 days)
    activity_by_day = []
    for i in range(min(30, timeframe_days)):
        date = datetime.utcnow() - timedelta(days=i)
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_incidents = db.query(func.count(Incident.id)).filter(
            Incident.reporter_id == user_id,
            Incident.created_at >= day_start,
            Incident.created_at < day_end,
        ).scalar() or 0
        
        day_briefings = db.query(func.count(Briefing.id)).filter(
            Briefing.user_id == user_id,
            Briefing.created_at >= day_start,
            Briefing.created_at < day_end,
        ).scalar() or 0
        
        activity_by_day.append({
            "date": day_start.date().isoformat(),
            "incidents": day_incidents,
            "briefings": day_briefings,
            "total_activity": day_incidents + day_briefings,
        })
    
    return UserStatsResponse(
        user_id=user_id,
        username=user.username,
        timeframe_days=timeframe_days,
        incident_stats={
            "total": total_incidents,
            "verified": verified_incidents,
            "verification_rate": round(verified_incidents / total_incidents * 100, 2) if total_incidents > 0 else 0,
            "recent": recent_incidents,
        },
        briefing_stats={
            "total": total_briefings,
            "recent": recent_briefings,
        },
        wallet_stats={
            "balance": wallet_balance,
            "total_earned": total_earned,
            "total_spent": total_spent,
            "net_earnings": total_earned - total_spent,
        },
        reputation_stats={
            "current_score": user.reputation_score,
            "rank": db.query(func.count(User.id)).filter(
                User.reputation_score > user.reputation_score,
                User.is_active == True,
            ).scalar() + 1,
            "total_users": db.query(User).filter(User.is_active == True).count(),
        },
        activity_timeline=activity_by_day,
        generated_at=datetime.utcnow(),
    )


@router.get("/{user_id}/incidents", response_model=PaginatedResponse[IncidentResponse])
async def get_user_incidents(
    user_id: int,
    pagination: PaginationParams = Depends(),
    status: Optional[str] = Query(None, description="Filter by verification status"),
    timeframe_days: Optional[int] = Query(None, description="Filter by time (last N days)"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> PaginatedResponse[IncidentResponse]:
    """
    Get incidents reported by a user.
    """
    user = db.query(User).filter(
        User.id == user_id,
        User.is_active == True,
    ).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check permissions (users can only see their own incidents or public ones)
    if user_id != current_user.id and current_user.role != "admin":
        # Only show verified incidents for other users
        query = db.query(Incident).filter(
            Incident.reporter_id == user_id,
            Incident.is_deleted == False,
            Incident.verification_status == "verified",
        )
    else:
        query = db.query(Incident).filter(
            Incident.reporter_id == user_id,
            Incident.is_deleted == False,
        )
    
    # Apply filters
    if status:
        query = query.filter(Incident.verification_status == status)
    
    if timeframe_days:
        timeframe = datetime.utcnow() - timedelta(days=timeframe_days)
        query = query.filter(Incident.created_at >= timeframe)
    
    total = query.count()
    incidents = query.order_by(Incident.created_at.desc())\
                     .offset(pagination.offset)\
                     .limit(pagination.limit)\
                     .all()
    
    return PaginatedResponse(
        items=[IncidentResponse.from_orm(incident) for incident in incidents],
        total=total,
        page=pagination.page,
        per_page=pagination.limit,
    )


@router.get("/leaderboard/reputation", response_model=List[Dict[str, Any]])
async def get_reputation_leaderboard(
    timeframe: str = Query("alltime", description="Timeframe: daily, weekly, monthly, alltime"),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """
    Get reputation leaderboard.
    """
    # Determine timeframe for reputation changes
    # For simplicity, we'll just use current reputation scores
    # In production, you might track reputation changes over time
    
    users = db.query(User).filter(
        User.is_active == True,
        User.is_deleted == False,
    ).order_by(User.reputation_score.desc())\
     .limit(limit)\
     .all()
    
    leaderboard = []
    for rank, user in enumerate(users, 1):
        # Get user stats
        incident_count = db.query(Incident).filter(
            Incident.reporter_id == user.id,
            Incident.is_deleted == False,
        ).count()
        
        verified_count = db.query(Incident).filter(
            Incident.reporter_id == user.id,
            Incident.verification_status == "verified",
            Incident.is_deleted == False,
        ).count()
        
        leaderboard.append({
            "rank": rank,
            "user_id": user.id,
            "username": user.username,
            "reputation_score": user.reputation_score,
            "incident_count": incident_count,
            "verified_count": verified_count,
            "verification_rate": round(verified_count / incident_count * 100, 2) if incident_count > 0 else 0,
            "is_current_user": current_user and user.id == current_user.id,
        })
    
    return leaderboard


@router.post("/{user_id}/follow")
async def follow_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Follow another user.
    """
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot follow yourself")
    
    user_to_follow = db.query(User).filter(
        User.id == user_id,
        User.is_active == True,
    ).first()
    
    if not user_to_follow:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if already following
    # This requires a followers table
    # For now, return placeholder response
    
    return {"message": f"Started following {user_to_follow.username}"}


@router.post("/{user_id}/unfollow")
async def unfollow_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Unfollow a user.
    """
    user_to_unfollow = db.query(User).filter(User.id == user_id).first()
    if not user_to_unfollow:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if actually following
    # This requires a followers table
    
    return {"message": f"Stopped following {user_to_unfollow.username}"}


@router.get("/{user_id}/followers")
async def get_user_followers(
    user_id: int,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get a user's followers.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # This requires a followers table
    # For now, return placeholder
    
    return {
        "user_id": user_id,
        "username": user.username,
        "followers": [],
        "total_followers": 0,
        "is_following": False,  # Whether current user is following this user
    }


@router.get("/{user_id}/following")
async def get_user_following(
    user_id: int,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get users that a user is following.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # This requires a followers table
    
    return {
        "user_id": user_id,
        "username": user.username,
        "following": [],
        "total_following": 0,
    }


@router.put("/{user_id}/role")
async def update_user_role(
    user_id: int,
    role: str = Query(..., description="New role for user"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
) -> Dict[str, str]:
    """
    Update a user's role (admin only).
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    valid_roles = ["user", "moderator", "admin"]
    if role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}"
        )
    
    user.role = role
    user.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": f"User role updated to {role}"}


@router.post("/{user_id}/suspend")
async def suspend_user(
    user_id: int,
    reason: str = Query(..., description="Reason for suspension"),
    duration_days: Optional[int] = Query(None, description="Suspension duration in days"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
) -> Dict[str, Any]:
    """
    Suspend a user (admin only).
    """
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot suspend yourself")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_active = False
    user.suspended_at = datetime.utcnow()
    user.suspended_by = current_user.id
    user.suspension_reason = reason
    
    if duration_days:
        user.suspension_until = datetime.utcnow() + timedelta(days=duration_days)
    
    user.updated_at = datetime.utcnow()
    db.commit()
    
    return {
        "message": f"User {user.username} suspended",
        "reason": reason,
        "suspended_until": user.suspension_until,
        "suspended_by": current_user.username,
    }


@router.post("/{user_id}/reactivate")
async def reactivate_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
) -> Dict[str, str]:
    """
    Reactivate a suspended user (admin only).
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_active = True
    user.suspended_at = None
    user.suspended_by = None
    user.suspension_reason = None
    user.suspension_until = None
    user.reactivated_at = datetime.utcnow()
    user.reactivated_by = current_user.id
    user.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": f"User {user.username} reactivated"}


@router.get("/analytics/daily")
async def get_daily_user_analytics(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin_role),
) -> Dict[str, Any]:
    """
    Get daily user analytics (admin only).
    """
    daily_stats = []
    
    for i in range(days):
        date = datetime.utcnow() - timedelta(days=i)
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        # New users
        new_users = db.query(func.count(User.id)).filter(
            User.created_at >= day_start,
            User.created_at < day_end,
        ).scalar() or 0
        
        # Active users (users with any activity)
        active_users = db.query(func.count(User.id)).filter(
            User.last_active >= day_start,
            User.last_active < day_end,
        ).scalar() or 0
        
        # New incidents
        new_incidents = db.query(func.count(Incident.id)).filter(
            Incident.created_at >= day_start,
            Incident.created_at < day_end,
        ).scalar() or 0
        
        daily_stats.append({
            "date": day_start.date().isoformat(),
            "new_users": new_users,
            "active_users": active_users,
            "new_incidents": new_incidents,
        })
    
    # Total statistics
    total_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
    total_incidents = db.query(func.count(Incident.id)).filter(Incident.is_deleted == False).scalar()
    
    # User growth rate (last 7 days vs previous 7 days)
    if days >= 14:
        recent_start = datetime.utcnow() - timedelta(days=7)
        previous_start = datetime.utcnow() - timedelta(days=14)
        
        recent_users = db.query(func.count(User.id)).filter(
            User.created_at >= recent_start,
        ).scalar() or 0
        
        previous_users = db.query(func.count(User.id)).filter(
            User.created_at >= previous_start,
            User.created_at < recent_start,
        ).scalar() or 0
        
        growth_rate = round((recent_users - previous_users) / previous_users * 100, 2) if previous_users > 0 else 0
    else:
        growth_rate = None
    
    return {
        "period_days": days,
        "daily_stats": daily_stats,
        "summary": {
            "total_users": total_users,
            "total_incidents": total_incidents,
            "growth_rate": growth_rate,
            "active_user_rate": round(sum(stat["active_users"] for stat in daily_stats) / total_users * 100, 2) if total_users > 0 else 0,
        },
        "generated_at": datetime.utcnow().isoformat(),
    }