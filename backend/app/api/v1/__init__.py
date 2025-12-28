"""
API Version 1 Router
"""
from fastapi import APIRouter

from . import (
    admin,
    analytics,
    auth,
    briefing,
    chat,
    health,
    incidents,
    notifications,
    rewards,
    topics,
    uploads,
    users,
    webhooks,
)

router = APIRouter(prefix="/v1", tags=["v1"])

# Include all routers
router.include_router(auth.router, tags=["authentication"])
router.include_router(users.router, tags=["users"])
router.include_router(topics.router, tags=["topics"])
router.include_router(briefing.router, tags=["briefings"])
router.include_router(chat.router, tags=["chat"])
router.include_router(incidents.router, tags=["incidents"])
router.include_router(rewards.router, tags=["rewards"])
router.include_router(uploads.router, tags=["uploads"])
router.include_router(analytics.router, tags=["analytics"])
router.include_router(notifications.router, tags=["notifications"])
router.include_router(webhooks.router, tags=["webhooks"])
router.include_router(health.router, tags=["health"])
router.include_router(admin.router, tags=["admin"], prefix="/admin")