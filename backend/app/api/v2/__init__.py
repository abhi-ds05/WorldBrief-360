"""
API Version 2 Router
Version 2 introduces breaking changes and new features.
"""
from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(prefix="/v2", tags=["v2"])


@router.get("/")
async def v2_root():
    """
    Version 2 API root endpoint.
    """
    return {
        "message": "Welcome to WorldBrief 360 API Version 2",
        "version": "2.0.0",
        "status": "active",
        "documentation": f"{settings.API_DOCS_URL}/v2",
        "changelog": [
            "BREAKING: Changed response format for all endpoints",
            "BREAKING: New authentication system with refresh tokens",
            "NEW: GraphQL endpoint for flexible queries",
            "NEW: Real-time event streaming with Server-Sent Events",
            "NEW: Bulk operations for incidents and topics",
            "NEW: Advanced filtering with query language",
            "NEW: WebSocket connections for live updates",
            "IMPROVED: Enhanced RAG system with multi-vector search",
            "IMPROVED: Better error messages with error codes",
            "IMPROVED: Rate limiting with sliding window algorithm",
            "DEPRECATED: Some v1 endpoints (see migration guide)",
        ],
        "migration_guide": f"{settings.API_DOCS_URL}/v2/migration",
    }


@router.get("/health")
async def v2_health_check():
    """
    Version 2 health check endpoint.
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": "2024-01-01T00:00:00Z",
        "services": {
            "api": "healthy",
            "database": "healthy",
            "cache": "healthy",
            "ai_services": "healthy",
            "storage": "healthy",
        },
    }


# Import and include version 2 routers here
# These will be implemented as we develop v2 features

# from .v2 import auth, users, topics, incidents, etc.
# router.include_router(auth.router)
# router.include_router(users.router)
# router.include_router(topics.router)
# router.include_router(incidents.router)

# Note: Actual v2 endpoints will be added in future iterations
# This is a placeholder structure for version 2 API