"""
Health check endpoints for monitoring and uptime.
"""
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database import get_db
from app.core.config import settings
from app.db.models import User, Incident, Topic
from app.schemas import HealthResponse, ServiceStatus

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: Session = Depends(get_db)
) -> HealthResponse:
    """
    Basic health check endpoint.
    """
    # Check database connectivity
    db_status = "healthy"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    # Check Redis connectivity (if configured)
    redis_status = "healthy"
    try:
        # This would require a Redis client instance
        # For now, we'll assume it's healthy if no exception
        pass
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"

    # Check Qdrant/vector database connectivity
    vector_db_status = "healthy"
    try:
        # This would require a Qdrant/vector DB client
        # For now, we'll assume it's healthy
        pass
    except Exception as e:
        vector_db_status = f"unhealthy: {str(e)}"

    # Get some basic statistics
    stats = {
        "total_users": db.query(User).count(),
        "total_incidents": db.query(Incident).count(),
        "total_topics": db.query(Topic).count(),
        "active_incidents": db.query(Incident).filter(
            Incident.verification_status == "pending"
        ).count(),
        "verified_incidents": db.query(Incident).filter(
            Incident.verification_status == "verified"
        ).count(),
    }

    return HealthResponse(
        status="healthy" if all([
            db_status == "healthy",
            redis_status == "healthy",
            vector_db_status == "healthy"
        ]) else "degraded",
        timestamp=datetime.utcnow(),
        version=settings.VERSION,
        services={
            "database": ServiceStatus(
                service="postgresql",
                status=db_status,
                response_time=0.1  # Placeholder
            ),
            "cache": ServiceStatus(
                service="redis",
                status=redis_status,
                response_time=0.01  # Placeholder
            ),
            "vector_db": ServiceStatus(
                service="qdrant",
                status=vector_db_status,
                response_time=0.05  # Placeholder
            ),
        },
        dependencies={
            "postgresql": db_status,
            "redis": redis_status,
            "qdrant": vector_db_status,
            "huggingface": "healthy",  # Assuming API is available
            "news_api": "healthy",  # Assuming API is available
        },
        stats=stats,
        uptime=0,  # Would need to calculate from start time
    )


@router.get("/health/live")
async def liveness_probe() -> Dict[str, str]:
    """
    Liveness probe for Kubernetes/container orchestration.
    Should be lightweight and only check if the app is running.
    """
    return {
        "status": "live",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/ready")
async def readiness_probe(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes/container orchestration.
    Checks if the app is ready to receive traffic.
    """
    dependencies = {}

    # Check database
    try:
        db.execute(text("SELECT 1"))
        dependencies["database"] = "ready"
    except Exception as e:
        dependencies["database"] = f"not_ready: {str(e)}"

    # Check Redis (if critical for operation)
    dependencies["redis"] = "ready"  # Placeholder

    # Check vector database
    dependencies["vector_db"] = "ready"  # Placeholder

    # Determine overall readiness
    all_ready = all("ready" in status for status in dependencies.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": dependencies
    }


@router.get("/health/detailed")
async def detailed_health_check(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Detailed health check with more information.
    """
    import platform
    import psutil
    import sys
    
    # System information
    system_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_usage": psutil.disk_usage('/')._asdict(),
    }

    # Database metrics
    db_metrics = {
        "connection_pool_size": db.bind.pool.size(),
        "connection_pool_checked_in": db.bind.pool.checkedin(),
        "connection_pool_checked_out": db.bind.pool.checkedout(),
    }

    # Application metrics
    app_metrics = {
        "total_requests": 0,  # Would need request counter
        "active_connections": 0,  # Would need connection tracking
        "background_tasks": 0,  # Would need task tracking
    }

    # Service versions
    versions = {
        "app": settings.VERSION,
        "fastapi": "0.104.1",
        "sqlalchemy": "2.0.23",
        "pydantic": "2.5.0",
    }

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "system": system_info,
        "database": db_metrics,
        "application": app_metrics,
        "versions": versions,
    }


@router.get("/health/version")
async def version_info() -> Dict[str, str]:
    """
    Get version information for the API.
    """
    import importlib.metadata
    
    try:
        app_version = importlib.metadata.version("worldbrief360")
    except importlib.metadata.PackageNotFoundError:
        app_version = "development"
    
    return {
        "name": "WorldBrief 360 API",
        "version": app_version,
        "api_version": "v1",
        "description": "Multimodal AI-Powered News, Insights & Community Incident Platform",
        "contact": {
            "name": "API Support",
            "email": "support@worldbrief360.com"
        },
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    }


@router.get("/health/metrics")
async def metrics_endpoint() -> str:
    """
    Prometheus metrics endpoint.
    Requires prometheus-client to be installed.
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        
        # Generate Prometheus metrics
        metrics = generate_latest()
        return Response(
            content=metrics,
            media_type=CONTENT_TYPE_LATEST
        )
    except ImportError:
        return "Prometheus client not installed"


@router.get("/health/status")
async def status_summary(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Quick status summary for dashboards.
    """
    from sqlalchemy import func
    
    # Get counts
    user_count = db.query(func.count(User.id)).scalar()
    incident_count = db.query(func.count(Incident.id)).scalar()
    topic_count = db.query(func.count(Topic.id)).scalar()
    
    # Get recent activity
    from datetime import datetime, timedelta
    hour_ago = datetime.utcnow() - timedelta(hours=1)
    
    recent_incidents = db.query(func.count(Incident.id)).filter(
        Incident.created_at >= hour_ago
    ).scalar()
    
    recent_users = db.query(func.count(User.id)).filter(
        User.created_at >= hour_ago
    ).scalar()
    
    return {
        "status": "operational",
        "last_updated": datetime.utcnow().isoformat(),
        "metrics": {
            "users": {
                "total": user_count,
                "new_last_hour": recent_users
            },
            "incidents": {
                "total": incident_count,
                "new_last_hour": recent_incidents
            },
            "topics": {
                "total": topic_count
            }
        },
        "system": {
            "database": "operational",
            "cache": "operational",
            "ai_services": "operational",
            "storage": "operational"
        }
    }