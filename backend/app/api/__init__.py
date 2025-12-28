"""
API Router - Main entry point for all API versions.
This file assembles all API version routers.
"""
from fastapi import APIRouter

from app.api import v1, v2
from app.core.config import settings

# Create main API router
api_router = APIRouter()

# Include version 1 API
api_router.include_router(v1.router, prefix="/api")

# Include version 2 API (when ready)
# api_router.include_router(v2.router, prefix="/api")

# Health check endpoint (available at root)
@api_router.get("/health")
async def api_health_check():
    """
    Root health check endpoint for the entire API.
    """
    return {
        "status": "healthy",
        "service": "WorldBrief 360 API",
        "version": settings.VERSION,
        "api_versions": ["v1"],  # Add "v2" when ready
        "documentation": "/docs",
        "openapi_spec": "/openapi.json",
        "timestamp": "2024-01-01T00:00:00Z",  # In production, use datetime.utcnow().isoformat()
    }


# API info endpoint
@api_router.get("/")
async def api_root():
    """
    Root endpoint showing API information and available versions.
    """
    return {
        "message": "Welcome to WorldBrief 360 API",
        "description": "Multimodal AI-Powered News, Insights & Community Incident Platform",
        "version": settings.VERSION,
        "api_versions": {
            "v1": {
                "status": "stable",
                "prefix": "/api/v1",
                "docs": "/docs",
                "openapi": "/openapi.json",
                "endpoints": [
                    "/auth",
                    "/users",
                    "/topics",
                    "/briefings",
                    "/chat",
                    "/incidents",
                    "/rewards",
                    "/uploads",
                    "/notifications",
                    "/analytics",
                    "/admin",
                    "/health",
                    "/webhooks",
                ]
            },
            "v2": {
                "status": "planned",
                "prefix": "/api/v2",
                "docs": "/docs/v2",
                "availability": "Q2 2024",
                "features": [
                    "GraphQL endpoint",
                    "Real-time streaming",
                    "Bulk operations",
                    "Advanced filtering",
                    "WebSocket support",
                ]
            }
        },
        "contact": {
            "name": "API Support",
            "email": "api-support@worldbrief360.com",
            "docs": "https://docs.worldbrief360.com"
        },
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        },
        "rate_limits": {
            "authenticated": "100 requests/minute",
            "unauthenticated": "10 requests/minute",
            "burst_limit": "50 requests/10 seconds"
        }
    }


# Version-specific endpoints
@api_router.get("/versions")
async def list_api_versions():
    """
    List all available API versions.
    """
    return {
        "versions": [
            {
                "version": "v1",
                "status": "stable",
                "base_url": "/api/v1",
                "release_date": "2024-01-01",
                "deprecation_date": None,
                "end_of_life": None,
                "changelog": "https://docs.worldbrief360.com/v1/changelog"
            },
            {
                "version": "v2",
                "status": "planned",
                "base_url": "/api/v2",
                "release_date": "2024-06-01",
                "deprecation_date": None,
                "end_of_life": None,
                "features": [
                    "GraphQL support",
                    "Real-time event streaming",
                    "Advanced filtering language",
                    "Bulk operations",
                    "WebSocket API"
                ]
            }
        ],
        "current_version": "v1",
        "default_version": "v1"
    }


# API status endpoint
@api_router.get("/status")
async def api_status():
    """
    Get detailed API status and metrics.
    """
    # In production, this would fetch real metrics
    return {
        "status": "operational",
        "uptime": "99.9%",
        "response_time": "150ms",
        "requests": {
            "total_today": 12543,
            "successful": 12498,
            "failed": 45,
            "success_rate": "99.64%"
        },
        "services": {
            "api_gateway": "operational",
            "authentication": "operational",
            "database": "operational",
            "cache": "operational",
            "ai_services": "operational",
            "storage": "operational",
            "email": "operational"
        },
        "last_incident": None,
        "timestamp": "2024-01-01T12:00:00Z"
    }


# API documentation redirect
@api_router.get("/docs")
async def redirect_to_docs():
    """
    Redirect to API documentation.
    In production, this would redirect to the actual docs URL.
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# OpenAPI schema endpoint
@api_router.get("/openapi.json")
async def get_openapi_schema():
    """
    Get OpenAPI schema for the API.
    This endpoint would be handled by FastAPI automatically,
    but we're including it for documentation purposes.
    """
    # FastAPI automatically generates this at /openapi.json
    return {"message": "OpenAPI schema available at /openapi.json"}


# Include routers from different API versions
# Note: v1 router is already included above with /api prefix
# v2 will be included when ready:
# api_router.include_router(v2.router, prefix="/api")

# You can also add middleware-specific endpoints here if needed
# For example, request ID endpoint, rate limit info, etc.

@api_router.get("/request-id")
async def get_request_id_info():
    """
    Information about request IDs for tracing.
    """
    return {
        "message": "All API requests include a unique request ID",
        "header_name": "X-Request-ID",
        "purpose": "Request tracing and debugging",
        "format": "UUID v4",
        "example": "123e4567-e89b-12d3-a456-426614174000"
    }


@api_router.get("/rate-limits")
async def get_rate_limit_info():
    """
    Information about API rate limits.
    """
    return {
        "rate_limits": {
            "default": {
                "authenticated": {
                    "limit": 100,
                    "period": "minute",
                    "description": "100 requests per minute for authenticated users"
                },
                "unauthenticated": {
                    "limit": 10,
                    "period": "minute",
                    "description": "10 requests per minute for unauthenticated users"
                }
            },
            "endpoint_specific": {
                "/api/v1/chat": {
                    "limit": 30,
                    "period": "minute",
                    "description": "30 chat requests per minute"
                },
                "/api/v1/incidents": {
                    "limit": 60,
                    "period": "minute",
                    "description": "60 incident reports per minute"
                },
                "/api/v1/uploads": {
                    "limit": 20,
                    "period": "minute",
                    "description": "20 file uploads per minute"
                }
            }
        },
        "headers": {
            "X-RateLimit-Limit": "Request limit per period",
            "X-RateLimit-Remaining": "Remaining requests in period",
            "X-RateLimit-Reset": "Timestamp when limit resets",
            "Retry-After": "Seconds to wait before retrying (when limited)"
        }
    }


# Export the main router
__all__ = ["api_router"]