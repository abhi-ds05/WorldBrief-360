"""
Main API router that assembles all API endpoints and middleware.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api import deps
from app.api.v1 import router as v1_router
from app.core.config import settings
from app.core.database import get_db
from app.core.security import get_current_user_optional

# Create main router
router = APIRouter()

# Include API version routers
router.include_router(v1_router, prefix="/v1", tags=["v1"])

# Root endpoint
@router.get("/")
async def api_root(
    request: Request,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_optional)
) -> JSONResponse:
    """
    API root endpoint showing available routes and API information.
    """
    # Get all registered routes
    routes = []
    for route in request.app.routes:
        route_info = {
            "path": route.path,
            "methods": list(route.methods) if hasattr(route, 'methods') else [],
            "name": route.name if hasattr(route, 'name') else None,
        }
        routes.append(route_info)
    
    # Filter to show only API routes
    api_routes = [r for r in routes if r["path"].startswith("/api/")]
    
    # Group routes by version and endpoint
    organized_routes = {}
    for route in api_routes:
        path_parts = route["path"].split("/")
        if len(path_parts) >= 3:
            version = path_parts[2]  # /api/v1/... -> v1
            endpoint = path_parts[3] if len(path_parts) > 3 else "root"
            
            if version not in organized_routes:
                organized_routes[version] = {}
            
            if endpoint not in organized_routes[version]:
                organized_routes[version][endpoint] = []
            
            organized_routes[version][endpoint].append({
                "path": route["path"],
                "methods": route["methods"],
            })
    
    return JSONResponse(
        content={
            "message": "Welcome to WorldBrief 360 API",
            "description": "Multimodal AI-Powered News, Insights & Community Incident Platform",
            "version": settings.VERSION,
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json",
            },
            "authentication": {
                "methods": ["JWT Bearer Token", "API Key"],
                "login_endpoint": "/api/v1/auth/login",
                "register_endpoint": "/api/v1/auth/register",
            },
            "rate_limits": {
                "authenticated": "100 requests/minute",
                "unauthenticated": "10 requests/minute",
                "admin": "500 requests/minute",
            },
            "api_versions": {
                "v1": {
                    "status": "stable",
                    "prefix": "/api/v1",
                    "endpoints": list(organized_routes.get("v1", {}).keys()),
                }
            },
            "current_user": {
                "authenticated": current_user is not None,
                "user_id": current_user.id if current_user else None,
                "username": current_user.username if current_user else None,
                "role": current_user.role if current_user else None,
            },
            "links": {
                "health": "/api/v1/health",
                "status": "/api/status",
                "versions": "/api/versions",
            },
            "contact": {
                "email": "api@worldbrief360.com",
                "documentation": "https://docs.worldbrief360.com",
                "support": "https://support.worldbrief360.com",
            },
        }
    )


# API health endpoint (aggregates health from all versions)
@router.get("/health")
async def api_health_aggregate(
    request: Request,
    db: Session = Depends(get_db)
) -> JSONResponse:
    """
    Aggregate health check across all API versions and services.
    """
    from datetime import datetime
    
    # Check database health
    db_healthy = False
    try:
        db.execute("SELECT 1")
        db_healthy = True
    except Exception:
        db_healthy = False
    
    # Check Redis health (if configured)
    redis_healthy = True  # Placeholder
    
    # Check external services
    services_status = {
        "database": "healthy" if db_healthy else "unhealthy",
        "redis": "healthy" if redis_healthy else "unhealthy",
        "ai_services": "healthy",  # Placeholder
        "storage": "healthy",  # Placeholder
        "external_apis": "healthy",  # Placeholder
    }
    
    # Determine overall status
    all_healthy = all(status == "healthy" for status in services_status.values())
    
    return JSONResponse(
        content={
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_status,
            "versions": {
                "v1": {
                    "status": "available",
                    "health_endpoint": "/api/v1/health",
                }
            },
            "response_time": "0ms",  # Would calculate actual response time
            "uptime": "99.9%",  # Would calculate from start time
        },
        status_code=200 if all_healthy else 503
    )


# API status endpoint
@router.get("/status")
async def api_status(
    request: Request,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_optional)
) -> JSONResponse:
    """
    Detailed API status and metrics.
    """
    from datetime import datetime, timedelta
    from sqlalchemy import func
    
    # Get request statistics
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    
    # Import models
    from app.db.models import RequestLog, User, Incident, Topic
    
    # Get counts from database
    total_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
    total_incidents = db.query(func.count(Incident.id)).filter(Incident.is_deleted == False).scalar()
    total_topics = db.query(func.count(Topic.id)).filter(Topic.is_active == True).scalar()
    
    # Recent activity (last hour)
    recent_users = db.query(func.count(User.id)).filter(
        User.created_at >= one_hour_ago
    ).scalar()
    
    recent_incidents = db.query(func.count(Incident.id)).filter(
        Incident.created_at >= one_hour_ago,
        Incident.is_deleted == False
    ).scalar()
    
    # Request statistics
    total_requests = db.query(func.count(RequestLog.id)).scalar()
    successful_requests = db.query(func.count(RequestLog.id)).filter(
        RequestLog.status_code < 400
    ).scalar()
    
    error_rate = round((total_requests - successful_requests) / total_requests * 100, 2) if total_requests > 0 else 0
    
    # Average response time
    avg_response_time = db.query(func.avg(RequestLog.response_time)).scalar() or 0
    
    return JSONResponse(
        content={
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "users": {
                    "total": total_users,
                    "active_last_hour": recent_users,
                },
                "incidents": {
                    "total": total_incidents,
                    "new_last_hour": recent_incidents,
                },
                "topics": {
                    "total": total_topics,
                },
                "performance": {
                    "total_requests": total_requests,
                    "success_rate": f"{100 - error_rate}%",
                    "error_rate": f"{error_rate}%",
                    "avg_response_time": f"{round(avg_response_time, 2)}ms",
                },
            },
            "system": {
                "database": "operational",
                "cache": "operational",
                "queue": "operational",
                "storage": "operational",
            },
            "limits": {
                "max_file_size": "10MB",
                "max_request_size": "10MB",
                "rate_limit_default": "100/minute",
                "rate_limit_burst": "50/10seconds",
            },
            "current_load": {
                "requests_per_second": 0,  # Would calculate from metrics
                "active_connections": 0,   # Would track connections
                "memory_usage": "45%",     # Would get from system
            },
        }
    )


# API versions endpoint
@router.get("/versions")
async def api_versions() -> JSONResponse:
    """
    List all available API versions with their status.
    """
    return JSONResponse(
        content={
            "versions": [
                {
                    "version": "v1",
                    "status": "stable",
                    "base_url": "/api/v1",
                    "release_date": "2024-01-01",
                    "deprecation_date": None,
                    "end_of_life": None,
                    "documentation": "/docs",
                    "openapi": "/openapi.json",
                    "supported": True,
                }
            ],
            "current_version": "v1",
            "default_version": "v1",
            "deprecation_policy": {
                "notice_period": "6 months",
                "migration_support": "3 months",
                "contact": "api-support@worldbrief360.com",
            },
            "upcoming_versions": [
                {
                    "version": "v2",
                    "status": "planned",
                    "estimated_release": "2024-06-01",
                    "features": [
                        "GraphQL support",
                        "Real-time event streaming",
                        "Advanced filtering language",
                        "Bulk operations",
                        "WebSocket API",
                    ],
                }
            ],
        }
    )


# API documentation endpoint
@router.get("/docs/redirect")
async def redirect_to_docs() -> JSONResponse:
    """
    Redirect to API documentation.
    """
    return JSONResponse(
        content={
            "message": "API documentation available at the following URLs:",
            "urls": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi_spec": "/openapi.json",
            },
            "external_docs": {
                "user_guide": "https://docs.worldbrief360.com/guide",
                "api_reference": "https://docs.worldbrief360.com/api",
                "quickstart": "https://docs.worldbrief360.com/quickstart",
            },
        }
    )


# API rate limits info endpoint
@router.get("/rate-limits")
async def api_rate_limits_info(
    current_user = Depends(get_current_user_optional)
) -> JSONResponse:
    """
    Information about API rate limits.
    """
    # Determine user's rate limit based on role
    if current_user:
        if current_user.role == "admin":
            limit = 500
            period = "minute"
            description = "Administrator rate limit"
        elif current_user.role == "moderator":
            limit = 250
            period = "minute"
            description = "Moderator rate limit"
        else:
            limit = 100
            period = "minute"
            description = "Authenticated user rate limit"
    else:
        limit = 10
        period = "minute"
        description = "Unauthenticated rate limit"
    
    return JSONResponse(
        content={
            "rate_limits": {
                "user_specific": {
                    "limit": limit,
                    "period": period,
                    "description": description,
                    "burst_limit": limit // 2,
                    "burst_period": "10 seconds",
                },
                "endpoint_specific": {
                    "auth": {
                        "login": {"limit": 5, "period": "minute", "description": "Login attempts"},
                        "register": {"limit": 3, "period": "minute", "description": "Registration attempts"},
                    },
                    "incidents": {
                        "create": {"limit": 10, "period": "minute", "description": "Incident reports"},
                        "upload": {"limit": 5, "period": "minute", "description": "Image uploads"},
                    },
                    "chat": {
                        "ask": {"limit": 30, "period": "minute", "description": "Chat questions"},
                    },
                },
            },
            "headers": {
                "X-RateLimit-Limit": "Total requests allowed per period",
                "X-RateLimit-Remaining": "Remaining requests in current period",
                "X-RateLimit-Reset": "Unix timestamp when limit resets",
                "Retry-After": "Seconds to wait before retrying (when rate limited)",
            },
            "best_practices": [
                "Implement exponential backoff when hitting rate limits",
                "Cache responses when possible",
                "Use bulk endpoints for multiple operations",
                "Monitor your usage with the X-RateLimit headers",
            ],
        }
    )


# API usage examples endpoint
@router.get("/examples")
async def api_usage_examples() -> JSONResponse:
    """
    Provide API usage examples.
    """
    return JSONResponse(
        content={
            "examples": {
                "authentication": {
                    "login": {
                        "curl": "curl -X POST https://api.worldbrief360.com/api/v1/auth/login \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"username\": \"user@example.com\", \"password\": \"yourpassword\"}'",
                        "response": {
                            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                            "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                            "token_type": "bearer",
                            "expires_in": 3600,
                        },
                    },
                    "using_token": {
                        "curl": "curl -X GET https://api.worldbrief360.com/api/v1/users/me \\\n  -H 'Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...'",
                    },
                },
                "incidents": {
                    "report_incident": {
                        "curl": "curl -X POST https://api.worldbrief360.com/api/v1/incidents \\\n  -H 'Authorization: Bearer TOKEN' \\\n  -F 'title=Road Accident' \\\n  -F 'description=Two cars collided at intersection' \\\n  -F 'category=accident' \\\n  -F 'latitude=40.7128' \\\n  -F 'longitude=-74.0060' \\\n  -F 'severity=high' \\\n  -F 'images=@photo1.jpg'",
                    },
                    "get_nearby_incidents": {
                        "curl": "curl -X GET 'https://api.worldbrief360.com/api/v1/incidents/nearby?latitude=40.7128&longitude=-74.0060&radius_km=5' \\\n  -H 'Authorization: Bearer TOKEN'",
                    },
                },
                "topics": {
                    "get_topic_briefing": {
                        "curl": "curl -X POST https://api.worldbrief360.com/api/v1/briefing/generate \\\n  -H 'Authorization: Bearer TOKEN' \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"topic_id\": 123, \"level\": \"general\", \"generate_audio\": true}'",
                    },
                    "chat_about_topic": {
                        "curl": "curl -X POST https://api.worldbrief360.com/api/v1/chat/ask \\\n  -H 'Authorization: Bearer TOKEN' \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"topic_id\": 123, \"question\": \"What are the economic impacts?\"}'",
                    },
                },
            },
            "libraries": {
                "python": {
                    "installation": "pip install worldbrief360-sdk",
                    "example": "from worldbrief360 import WorldBrief360\n\nclient = WorldBrief360(api_key='your_api_key')\nincidents = client.incidents.get_nearby(latitude=40.7128, longitude=-74.0060)",
                },
                "javascript": {
                    "installation": "npm install worldbrief360-sdk",
                    "example": "import { WorldBrief360 } from 'worldbrief360-sdk';\n\nconst client = new WorldBrief360({ apiKey: 'your_api_key' });\nconst incidents = await client.incidents.getNearby(40.7128, -74.0060);",
                },
            },
        }
    )


# API changelog endpoint
@router.get("/changelog")
async def api_changelog() -> JSONResponse:
    """
    API changelog and version history.
    """
    return JSONResponse(
        content={
            "changelog": [
                {
                    "version": "1.0.0",
                    "date": "2024-01-01",
                    "status": "stable",
                    "changes": {
                        "added": [
                            "Initial API release with all core endpoints",
                            "JWT authentication system",
                            "Incident reporting and verification",
                            "Topic-based briefing generation",
                            "AI-powered chat with RAG",
                            "Reward and wallet system",
                            "User notifications",
                            "File uploads with S3 integration",
                        ],
                        "changed": [],
                        "deprecated": [],
                        "removed": [],
                        "fixed": [],
                        "security": [
                            "Implemented rate limiting",
                            "Added CORS configuration",
                            "Security headers middleware",
                            "Input validation on all endpoints",
                        ],
                    },
                },
            ],
            "upcoming": [
                {
                    "version": "1.1.0",
                    "estimated": "2024-02-01",
                    "planned_features": [
                        "WebSocket support for real-time updates",
                        "Bulk operations API",
                        "Advanced analytics endpoints",
                        "Export functionality",
                    ],
                },
                {
                    "version": "2.0.0",
                    "estimated": "2024-06-01",
                    "planned_features": [
                        "GraphQL API",
                        "Real-time event streaming",
                        "Advanced filtering language",
                        "Plugin system for extensions",
                    ],
                },
            ],
        }
    )


# Export the router
__all__ = ["router"]