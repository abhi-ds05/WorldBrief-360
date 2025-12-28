"""
Topic exploration and management endpoints.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func, desc

from app.core.database import get_db
from app.core.security import get_current_user
from app.db.models import User, Topic, Article, Briefing, Subscription
from app.schemas import (
    TopicResponse,
    TopicCreate,
    TopicUpdate,
    TopicDetailResponse,
    ArticleResponse,
    SubscriptionResponse,
    PaginatedResponse,
    TrendingTopicResponse,
)
from app.schemas.common import PaginationParams
from app.services.ingestion.news_ingestor import fetch_news_for_topic
from app.services.processing.rag_indexer import index_topic_content
from app.services.generators.briefing_generator import generate_topic_summary

router = APIRouter()


@router.get("/", response_model=PaginatedResponse[TopicResponse])
async def list_topics(
    pagination: PaginationParams = Depends(),
    search: Optional[str] = Query(None, description="Search topics by title or description"),
    category: Optional[str] = Query(None, description="Filter by category"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    sort_by: str = Query("created_at", description="Sort field: created_at, updated_at, popularity"),
    sort_order: str = Query("desc", description="Sort order: asc, desc"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> PaginatedResponse[TopicResponse]:
    """
    List all topics with filtering and pagination.
    """
    query = db.query(Topic)
    
    # Apply filters
    if search:
        query = query.filter(
            or_(
                Topic.title.ilike(f"%{search}%"),
                Topic.description.ilike(f"%{search}%"),
                Topic.keywords.ilike(f"%{search}%"),
            )
        )
    
    if category:
        query = query.filter(Topic.category == category)
    
    if is_active is not None:
        query = query.filter(Topic.is_active == is_active)
    
    # Apply sorting
    sort_field = getattr(Topic, sort_by, Topic.created_at)
    if sort_order.lower() == "desc":
        query = query.order_by(desc(sort_field))
    else:
        query = query.order_by(sort_field)
    
    total = query.count()
    topics = query.offset(pagination.offset).limit(pagination.limit).all()
    
    return PaginatedResponse(
        items=[TopicResponse.from_orm(topic) for topic in topics],
        total=total,
        page=pagination.page,
        per_page=pagination.limit,
    )


@router.get("/trending", response_model=List[TrendingTopicResponse])
async def get_trending_topics(
    limit: int = Query(10, ge=1, le=50),
    timeframe_hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> List[TrendingTopicResponse]:
    """
    Get trending topics based on recent activity.
    """
    from datetime import datetime, timedelta
    
    timeframe = datetime.utcnow() - timedelta(hours=timeframe_hours)
    
    # Get topics with most articles in timeframe
    trending = db.query(
        Topic.id,
        Topic.title,
        Topic.category,
        Topic.description,
        func.count(Article.id).label("article_count"),
        func.max(Article.published_at).label("latest_article"),
        func.count(Briefing.id).label("briefing_count"),
    ).outerjoin(Article, Article.topic_id == Topic.id)\
     .outerjoin(Briefing, Briefing.topic_id == Topic.id)\
     .filter(
        Article.published_at >= timeframe,
        Topic.is_active == True,
     ).group_by(Topic.id)\
     .order_by(desc("article_count"), desc("briefing_count"))\
     .limit(limit)\
     .all()
    
    # Calculate trend score
    trending_topics = []
    for topic in trending:
        # Time decay factor
        hours_since = (datetime.utcnow() - (topic.latest_article or datetime.utcnow())).total_seconds() / 3600
        recency_factor = max(0, 1 - (hours_since / timeframe_hours))
        
        # Activity factor
        activity_factor = min(1, (topic.article_count * 0.5 + topic.briefing_count * 0.5) / 10)
        
        trend_score = round(recency_factor * activity_factor * 100, 2)
        
        trending_topics.append(TrendingTopicResponse(
            id=topic.id,
            title=topic.title,
            category=topic.category,
            description=topic.description,
            article_count=topic.article_count,
            briefing_count=topic.briefing_count,
            latest_activity=topic.latest_article,
            trend_score=trend_score,
        ))
    
    return trending_topics


@router.get("/categories")
async def get_topic_categories(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get all topic categories with statistics.
    """
    categories = db.query(
        Topic.category,
        func.count(Topic.id).label("topic_count"),
        func.sum(Topic.popularity_score).label("total_popularity"),
    ).filter(
        Topic.is_active == True,
        Topic.category.isnot(None),
    ).group_by(Topic.category).all()
    
    return {
        "categories": [
            {
                "name": category,
                "topic_count": count,
                "average_popularity": round(popularity / count, 2) if count > 0 else 0,
            }
            for category, count, popularity in categories
        ],
        "total_categories": len(categories),
        "total_topics": sum(count for _, count, _ in categories),
    }


@router.post("/", response_model=TopicResponse)
async def create_topic(
    topic_data: TopicCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TopicResponse:
    """
    Create a new topic (admin only).
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Check if topic already exists
    existing_topic = db.query(Topic).filter(
        or_(
            Topic.title == topic_data.title,
            Topic.slug == topic_data.slug,
        )
    ).first()
    
    if existing_topic:
        raise HTTPException(
            status_code=400,
            detail="Topic with this title or slug already exists"
        )
    
    # Create topic
    topic = Topic(
        title=topic_data.title,
        slug=topic_data.slug,
        description=topic_data.description,
        category=topic_data.category,
        keywords=topic_data.keywords,
        region=topic_data.region,
        is_active=True,
        created_by=current_user.id,
    )
    
    db.add(topic)
    db.commit()
    db.refresh(topic)
    
    # Fetch initial news for the topic
    try:
        await fetch_news_for_topic(topic.id, db)
    except Exception as e:
        print(f"Failed to fetch initial news for topic {topic.id}: {e}")
    
    # Index topic content for RAG
    try:
        await index_topic_content(topic.id, db)
    except Exception as e:
        print(f"Failed to index topic {topic.id}: {e}")
    
    return TopicResponse.from_orm(topic)


@router.get("/{topic_id}", response_model=TopicDetailResponse)
async def get_topic(
    topic_id: int,
    include_articles: bool = Query(True, description="Include related articles"),
    include_briefings: bool = Query(False, description="Include user's briefings for this topic"),
    limit_articles: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> TopicDetailResponse:
    """
    Get detailed information about a topic.
    """
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    # Increment view count
    topic.views += 1
    topic.popularity_score = calculate_popularity_score(topic)
    db.commit()
    
    # Get related articles
    articles = []
    if include_articles:
        articles = db.query(Article).filter(
            Article.topic_id == topic_id,
            Article.is_active == True,
        ).order_by(desc(Article.published_at))\
         .limit(limit_articles)\
         .all()
    
    # Get user's briefings for this topic
    briefings = []
    if include_briefings and current_user:
        briefings = db.query(Briefing).filter(
            Briefing.topic_id == topic_id,
            Briefing.user_id == current_user.id,
        ).order_by(desc(Briefing.created_at)).all()
    
    # Get subscription status
    is_subscribed = False
    if current_user:
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id,
            Subscription.topic_id == topic_id,
            Subscription.is_active == True,
        ).first()
        is_subscribed = subscription is not None
    
    # Generate topic summary if not exists
    if not topic.summary:
        try:
            summary = await generate_topic_summary(topic, db)
            topic.summary = summary
            db.commit()
        except Exception as e:
            print(f"Failed to generate summary for topic {topic_id}: {e}")
    
    return TopicDetailResponse(
        **TopicResponse.from_orm(topic).dict(),
        articles=[ArticleResponse.from_orm(article) for article in articles],
        briefings=briefings,  # This would need proper serialization
        is_subscribed=is_subscribed,
        total_articles=db.query(Article).filter(
            Article.topic_id == topic_id,
            Article.is_active == True,
        ).count(),
        total_briefings=db.query(Briefing).filter(
            Briefing.topic_id == topic_id,
        ).count(),
        last_updated=topic.updated_at,
    )


@router.put("/{topic_id}", response_model=TopicResponse)
async def update_topic(
    topic_id: int,
    topic_update: TopicUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TopicResponse:
    """
    Update a topic (admin only).
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    # Update fields
    for field, value in topic_update.dict(exclude_unset=True).items():
        setattr(topic, field, value)
    
    topic.updated_at = datetime.utcnow()
    topic.updated_by = current_user.id
    
    db.commit()
    db.refresh(topic)
    
    # Re-index if content changed
    if any(field in topic_update.dict() for field in ['title', 'description', 'keywords']):
        try:
            await index_topic_content(topic_id, db)
        except Exception as e:
            print(f"Failed to re-index topic {topic_id}: {e}")
    
    return TopicResponse.from_orm(topic)


@router.delete("/{topic_id}")
async def delete_topic(
    topic_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Delete a topic (soft delete, admin only).
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    # Soft delete
    topic.is_active = False
    topic.is_deleted = True
    topic.deleted_at = datetime.utcnow()
    topic.deleted_by = current_user.id
    
    db.commit()
    
    return {"message": "Topic deleted successfully"}


@router.post("/{topic_id}/refresh")
async def refresh_topic_content(
    topic_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Refresh topic content by fetching latest news and re-indexing.
    """
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    # Fetch latest news
    new_articles = await fetch_news_for_topic(topic_id, db)
    
    # Re-index topic content
    await index_topic_content(topic_id, db)
    
    # Update topic summary
    summary = await generate_topic_summary(topic, db)
    topic.summary = summary
    topic.updated_at = datetime.utcnow()
    db.commit()
    
    return {
        "message": "Topic content refreshed successfully",
        "topic_id": topic_id,
        "new_articles_count": len(new_articles),
        "summary_updated": bool(summary),
        "last_refreshed": datetime.utcnow().isoformat(),
    }


@router.post("/{topic_id}/subscribe", response_model=SubscriptionResponse)
async def subscribe_to_topic(
    topic_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SubscriptionResponse:
    """
    Subscribe to a topic for updates.
    """
    topic = db.query(Topic).filter(
        Topic.id == topic_id,
        Topic.is_active == True,
    ).first()
    
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    # Check if already subscribed
    existing_subscription = db.query(Subscription).filter(
        Subscription.user_id == current_user.id,
        Subscription.topic_id == topic_id,
    ).first()
    
    if existing_subscription:
        # Reactivate if inactive
        if not existing_subscription.is_active:
            existing_subscription.is_active = True
            existing_subscription.updated_at = datetime.utcnow()
        else:
            raise HTTPException(
                status_code=400,
                detail="Already subscribed to this topic"
            )
    else:
        # Create new subscription
        subscription = Subscription(
            user_id=current_user.id,
            topic_id=topic_id,
            is_active=True,
        )
        db.add(subscription)
    
    db.commit()
    
    if existing_subscription:
        db.refresh(existing_subscription)
        return SubscriptionResponse.from_orm(existing_subscription)
    else:
        db.refresh(subscription)
        return SubscriptionResponse.from_orm(subscription)


@router.post("/{topic_id}/unsubscribe")
async def unsubscribe_from_topic(
    topic_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Unsubscribe from a topic.
    """
    subscription = db.query(Subscription).filter(
        Subscription.user_id == current_user.id,
        Subscription.topic_id == topic_id,
        Subscription.is_active == True,
    ).first()
    
    if not subscription:
        raise HTTPException(
            status_code=404,
            detail="Not subscribed to this topic"
        )
    
    subscription.is_active = False
    subscription.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": "Unsubscribed from topic successfully"}


@router.get("/user/subscriptions", response_model=List[SubscriptionResponse])
async def get_user_subscriptions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[SubscriptionResponse]:
    """
    Get all topics the user is subscribed to.
    """
    subscriptions = db.query(Subscription).filter(
        Subscription.user_id == current_user.id,
        Subscription.is_active == True,
    ).all()
    
    return [SubscriptionResponse.from_orm(sub) for sub in subscriptions]


@router.get("/{topic_id}/articles", response_model=PaginatedResponse[ArticleResponse])
async def get_topic_articles(
    topic_id: int,
    pagination: PaginationParams = Depends(),
    timeframe_days: Optional[int] = Query(None, description="Filter by time (last N days)"),
    source: Optional[str] = Query(None, description="Filter by news source"),
    sort_by: str = Query("published_at", description="Sort field: published_at, relevance, popularity"),
    sort_order: str = Query("desc", description="Sort order: asc, desc"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> PaginatedResponse[ArticleResponse]:
    """
    Get articles for a specific topic.
    """
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    query = db.query(Article).filter(
        Article.topic_id == topic_id,
        Article.is_active == True,
    )
    
    # Apply filters
    if timeframe_days:
        timeframe = datetime.utcnow() - timedelta(days=timeframe_days)
        query = query.filter(Article.published_at >= timeframe)
    
    if source:
        query = query.filter(Article.source == source)
    
    # Apply sorting
    if sort_by == "relevance":
        # Simple relevance based on title/description match with topic keywords
        # In production, use full-text search or ML ranking
        query = query.order_by(desc(Article.relevance_score))
    elif sort_by == "popularity":
        query = query.order_by(desc(Article.popularity_score))
    else:  # published_at
        sort_field = getattr(Article, sort_by, Article.published_at)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(sort_field))
        else:
            query = query.order_by(sort_field)
    
    total = query.count()
    articles = query.offset(pagination.offset).limit(pagination.limit).all()
    
    return PaginatedResponse(
        items=[ArticleResponse.from_orm(article) for article in articles],
        total=total,
        page=pagination.page,
        per_page=pagination.limit,
    )


@router.get("/{topic_id}/stats")
async def get_topic_statistics(
    topic_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get statistics for a topic.
    """
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    # Article statistics
    article_stats = db.query(
        func.count(Article.id).label("total_articles"),
        func.count(Article.id).filter(Article.published_at >= datetime.utcnow() - timedelta(days=1)).label("articles_last_24h"),
        func.count(Article.id).filter(Article.published_at >= datetime.utcnow() - timedelta(days=7)).label("articles_last_7d"),
        func.avg(Article.sentiment_score).label("avg_sentiment"),
    ).filter(
        Article.topic_id == topic_id,
        Article.is_active == True,
    ).first()
    
    # Source distribution
    source_dist = db.query(
        Article.source,
        func.count(Article.id).label("count")
    ).filter(
        Article.topic_id == topic_id,
        Article.is_active == True,
    ).group_by(Article.source).all()
    
    # Briefing statistics
    briefing_stats = db.query(
        func.count(Briefing.id).label("total_briefings"),
        func.count(Briefing.id).filter(Briefing.created_at >= datetime.utcnow() - timedelta(days=1)).label("briefings_last_24h"),
    ).filter(Briefing.topic_id == topic_id).first()
    
    # Subscription count
    subscription_count = db.query(Subscription).filter(
        Subscription.topic_id == topic_id,
        Subscription.is_active == True,
    ).count()
    
    return {
        "topic_id": topic_id,
        "title": topic.title,
        "article_statistics": {
            "total_articles": article_stats.total_articles or 0,
            "articles_last_24h": article_stats.articles_last_24h or 0,
            "articles_last_7d": article_stats.articles_last_7d or 0,
            "average_sentiment": round(article_stats.avg_sentiment or 0, 2),
        },
        "source_distribution": {
            source: count for source, count in source_dist
        },
        "briefing_statistics": {
            "total_briefings": briefing_stats.total_briefings or 0,
            "briefings_last_24h": briefing_stats.briefings_last_24h or 0,
        },
        "engagement": {
            "views": topic.views,
            "subscriptions": subscription_count,
            "popularity_score": topic.popularity_score,
        },
        "generated_at": datetime.utcnow().isoformat(),
    }


def calculate_popularity_score(topic: Topic) -> float:
    """
    Calculate popularity score for a topic.
    """
    # Base score from views
    view_score = min(topic.views / 100, 1.0)  # Cap at 100 views
    
    # Time decay factor (more recent activity = higher score)
    days_since_update = (datetime.utcnow() - topic.updated_at).days
    recency_factor = max(0, 1 - (days_since_update / 30))  # 30-day window
    
    # Combine factors
    popularity = (view_score * 0.6 + recency_factor * 0.4) * 100
    
    return round(popularity, 2)


@router.get("/search/suggestions")
async def get_search_suggestions(
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get search suggestions for topics.
    """
    # Search in titles
    title_matches = db.query(Topic).filter(
        Topic.title.ilike(f"%{query}%"),
        Topic.is_active == True,
    ).limit(limit).all()
    
    # Search in descriptions
    desc_matches = db.query(Topic).filter(
        Topic.description.ilike(f"%{query}%"),
        Topic.is_active == True,
        ~Topic.id.in_([t.id for t in title_matches])  # Exclude duplicates
    ).limit(limit).all()
    
    # Search in keywords
    keyword_matches = db.query(Topic).filter(
        Topic.keywords.ilike(f"%{query}%"),
        Topic.is_active == True,
        ~Topic.id.in_([t.id for t in title_matches + desc_matches])  # Exclude duplicates
    ).limit(limit).all()
    
    all_matches = title_matches + desc_matches + keyword_matches
    
    return {
        "query": query,
        "suggestions": [
            {
                "id": topic.id,
                "title": topic.title,
                "category": topic.category,
                "description": topic.description[:100] + "..." if len(topic.description) > 100 else topic.description,
                "match_type": "title" if topic in title_matches else "description" if topic in desc_matches else "keywords",
            }
            for topic in all_matches[:limit]
        ],
        "total_matches": len(all_matches),
    }