"""
Analytics and usage statistics endpoints.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, case

from app.core.database import get_db
from app.core.security import get_current_user
from app.db.models import User, Topic, Incident, Briefing, Transaction
from app.schemas import (
    UsageStatsResponse,
    TopicTrendResponse,
    IncidentAnalyticsResponse,
    UserAnalyticsResponse,
)

router = APIRouter()


@router.get("/usage", response_model=UsageStatsResponse)
async def get_usage_stats(
    period: str = Query("24h", description="Time period: 24h, 7d, 30d"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Get usage statistics for the platform.
    """
    now = datetime.utcnow()
    
    if period == "24h":
        start_time = now - timedelta(hours=24)
    elif period == "7d":
        start_time = now - timedelta(days=7)
    elif period == "30d":
        start_time = now - timedelta(days=30)
    else:
        raise HTTPException(status_code=400, detail="Invalid period. Use: 24h, 7d, 30d")

    # Active users in period
    active_users = db.query(func.count(User.id)).filter(
        User.last_active >= start_time
    ).scalar()

    # New users in period
    new_users = db.query(func.count(User.id)).filter(
        User.created_at >= start_time
    ).scalar()

    # New incidents in period
    new_incidents = db.query(func.count(Incident.id)).filter(
        Incident.created_at >= start_time
    ).scalar()

    # Verified incidents in period
    verified_incidents = db.query(func.count(Incident.id)).filter(
        and_(
            Incident.created_at >= start_time,
            Incident.verification_status == "verified"
        )
    ).scalar()

    # Briefings generated in period
    briefings_generated = db.query(func.count(Briefing.id)).filter(
        Briefing.created_at >= start_time
    ).scalar()

    # Chat interactions in period
    chat_interactions = db.query(func.count(Briefing.id)).filter(
        and_(
            Briefing.created_at >= start_time,
            Briefing.generation_type == "chat"
        )
    ).scalar()

    # Coins awarded in period
    coins_awarded = db.query(func.sum(Transaction.amount)).filter(
        and_(
            Transaction.created_at >= start_time,
            Transaction.transaction_type == "reward"
        )
    ).scalar() or 0

    # Coins spent in period
    coins_spent = db.query(func.sum(Transaction.amount)).filter(
        and_(
            Transaction.created_at >= start_time,
            Transaction.transaction_type == "purchase"
        )
    ).scalar() or 0

    return UsageStatsResponse(
        period=period,
        active_users=active_users,
        new_users=new_users,
        new_incidents=new_incidents,
        verified_incidents=verified_incidents,
        briefings_generated=briefings_generated,
        chat_interactions=chat_interactions,
        coins_awarded=coins_awarded,
        coins_spent=coins_spent,
        generated_at=now,
    )


@router.get("/topics/trending", response_model=List[TopicTrendResponse])
async def get_trending_topics(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Get trending topics based on recent activity.
    """
    from datetime import datetime, timedelta

    # Get topics with most briefings in last 24 hours
    yesterday = datetime.utcnow() - timedelta(hours=24)
    
    trending = db.query(
        Topic.id,
        Topic.title,
        Topic.category,
        func.count(Briefing.id).label("briefing_count"),
        func.max(Briefing.created_at).label("last_activity")
    ).join(Briefing, Briefing.topic_id == Topic.id)\
     .filter(Briefing.created_at >= yesterday)\
     .group_by(Topic.id)\
     .order_by(func.count(Briefing.id).desc())\
     .limit(limit)\
     .all()

    return [
        TopicTrendResponse(
            id=topic.id,
            title=topic.title,
            category=topic.category,
            briefing_count=topic.briefing_count,
            last_activity=topic.last_activity,
            trend_score=calculate_trend_score(topic.briefing_count, topic.last_activity)
        )
        for topic in trending
    ]


def calculate_trend_score(count: int, last_activity: datetime) -> float:
    """
    Calculate trending score based on recency and volume.
    """
    from datetime import datetime
    
    # Time decay factor (more recent = higher score)
    hours_since = (datetime.utcnow() - last_activity).total_seconds() / 3600
    recency_factor = max(0, 1 - (hours_since / 24))
    
    # Volume factor (logarithmic to prevent domination by single topics)
    volume_factor = min(1, count / 10)  # Cap at 10 briefings
    
    return round(recency_factor * volume_factor * 100, 2)


@router.get("/incidents/analytics", response_model=IncidentAnalyticsResponse)
async def get_incident_analytics(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Get analytics for incidents.
    """
    from datetime import datetime, timedelta
    import json

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Category distribution
    category_distribution = db.query(
        Incident.category,
        func.count(Incident.id).label("count")
    ).filter(
        Incident.created_at >= start_date,
        Incident.created_at <= end_date
    ).group_by(Incident.category)\
     .all()

    # Status distribution
    status_distribution = db.query(
        Incident.verification_status,
        func.count(Incident.id).label("count")
    ).filter(
        Incident.created_at >= start_date,
        Incident.created_at <= end_date
    ).group_by(Incident.verification_status)\
     .all()

    # Daily incident count
    daily_counts = []
    for i in range(days):
        date = start_date + timedelta(days=i)
        next_date = date + timedelta(days=1)
        
        count = db.query(func.count(Incident.id)).filter(
            and_(
                Incident.created_at >= date,
                Incident.created_at < next_date
            )
        ).scalar()
        
        daily_counts.append({
            "date": date.date().isoformat(),
            "count": count
        })

    # Top reporters
    top_reporters = db.query(
        User.id,
        User.username,
        func.count(Incident.id).label("incident_count"),
        func.sum(case([(Incident.verification_status == "verified", 1)], else_=0)).label("verified_count")
    ).join(Incident, Incident.reporter_id == User.id)\
     .filter(Incident.created_at >= start_date)\
     .group_by(User.id)\
     .order_by(func.count(Incident.id).desc())\
     .limit(10)\
     .all()

    return IncidentAnalyticsResponse(
        period_days=days,
        category_distribution={
            cat: count for cat, count in category_distribution
        },
        status_distribution={
            status: count for status, count in status_distribution
        },
        daily_counts=daily_counts,
        top_reporters=[
            {
                "user_id": reporter.id,
                "username": reporter.username,
                "incident_count": reporter.incident_count,
                "verified_count": reporter.verified_count or 0,
                "verification_rate": round((reporter.verified_count or 0) / reporter.incident_count * 100, 2) if reporter.incident_count > 0 else 0
            }
            for reporter in top_reporters
        ],
        generated_at=datetime.utcnow(),
    )


@router.get("/user/{user_id}/analytics", response_model=UserAnalyticsResponse)
async def get_user_analytics(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get analytics for a specific user.
    """
    from datetime import datetime, timedelta

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check permission (user can only see their own analytics or admin)
    if user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")

    # User activity in last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    # Incidents reported
    incidents_reported = db.query(func.count(Incident.id)).filter(
        and_(
            Incident.reporter_id == user_id,
            Incident.created_at >= thirty_days_ago
        )
    ).scalar()

    # Incidents verified
    incidents_verified = db.query(func.count(Incident.id)).filter(
        and_(
            Incident.reporter_id == user_id,
            Incident.verification_status == "verified",
            Incident.created_at >= thirty_days_ago
        )
    ).scalar()

    # Briefings generated
    briefings_generated = db.query(func.count(Briefing.id)).filter(
        and_(
            Briefing.user_id == user_id,
            Briefing.created_at >= thirty_days_ago
        )
    ).scalar()

    # Chat interactions
    chat_interactions = db.query(func.count(Briefing.id)).filter(
        and_(
            Briefing.user_id == user_id,
            Briefing.generation_type == "chat",
            Briefing.created_at >= thirty_days_ago
        )
    ).scalar()

    # Coins earned
    coins_earned = db.query(func.sum(Transaction.amount)).filter(
        and_(
            Transaction.user_id == user_id,
            Transaction.transaction_type == "reward",
            Transaction.created_at >= thirty_days_ago
        )
    ).scalar() or 0

    # Coins spent
    coins_spent = db.query(func.sum(Transaction.amount)).filter(
        and_(
            Transaction.user_id == user_id,
            Transaction.transaction_type == "purchase",
            Transaction.created_at >= thirty_days_ago
        )
    ).scalar() or 0

    # Daily activity
    daily_activity = []
    for i in range(30):
        date = datetime.utcnow() - timedelta(days=i)
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        day_incidents = db.query(func.count(Incident.id)).filter(
            and_(
                Incident.reporter_id == user_id,
                Incident.created_at >= start_of_day,
                Incident.created_at < end_of_day
            )
        ).scalar()

        day_briefings = db.query(func.count(Briefing.id)).filter(
            and_(
                Briefing.user_id == user_id,
                Briefing.created_at >= start_of_day,
                Briefing.created_at < end_of_day
            )
        ).scalar()

        daily_activity.append({
            "date": start_of_day.date().isoformat(),
            "incidents": day_incidents,
            "briefings": day_briefings,
        })

    return UserAnalyticsResponse(
        user_id=user_id,
        username=user.username,
        period_days=30,
        incidents_reported=incidents_reported,
        incidents_verified=incidents_verified,
        verification_rate=round(incidents_verified / incidents_reported * 100, 2) if incidents_reported > 0 else 0,
        briefings_generated=briefings_generated,
        chat_interactions=chat_interactions,
        coins_earned=coins_earned,
        coins_spent=coins_spent,
        coins_balance=coins_earned - coins_spent,
        daily_activity=daily_activity,
        generated_at=datetime.utcnow(),
    )