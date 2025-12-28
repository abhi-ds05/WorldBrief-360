"""
User notification endpoints.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from app.core.database import get_db
from app.core.security import get_current_user
from app.db.models import User, Notification, UserNotificationSettings
from app.schemas import (
    NotificationResponse,
    NotificationSettings,
    NotificationCreate,
    NotificationUpdate,
    PaginatedResponse,
)
from app.schemas.common import PaginationParams
from app.services.utils.notification_client import (
    send_push_notification,
    send_email_notification,
    send_sms_notification,
)

router = APIRouter()


@router.get("/", response_model=PaginatedResponse[NotificationResponse])
async def get_notifications(
    pagination: PaginationParams = Depends(),
    unread_only: bool = Query(False, description="Only show unread notifications"),
    notification_type: Optional[str] = Query(None, description="Filter by notification type"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PaginatedResponse[NotificationResponse]:
    """
    Get notifications for the current user.
    """
    query = db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_deleted == False,
    )
    
    if unread_only:
        query = query.filter(Notification.is_read == False)
    
    if notification_type:
        query = query.filter(Notification.notification_type == notification_type)
    
    total = query.count()
    notifications = query.order_by(Notification.created_at.desc())\
                         .offset(pagination.offset)\
                         .limit(pagination.limit)\
                         .all()
    
    return PaginatedResponse(
        items=[NotificationResponse.from_orm(notif) for notif in notifications],
        total=total,
        page=pagination.page,
        per_page=pagination.limit,
    )


@router.get("/unread/count")
async def get_unread_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, int]:
    """
    Get count of unread notifications.
    """
    count = db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_read == False,
        Notification.is_deleted == False,
    ).count()
    
    return {"unread_count": count}


@router.get("/{notification_id}", response_model=NotificationResponse)
async def get_notification(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> NotificationResponse:
    """
    Get a specific notification.
    """
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id,
        Notification.is_deleted == False,
    ).first()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    # Mark as read when retrieved
    if not notification.is_read:
        notification.is_read = True
        notification.read_at = datetime.utcnow()
        db.commit()
    
    return NotificationResponse.from_orm(notification)


@router.post("/", response_model=NotificationResponse)
async def create_notification(
    notification_data: NotificationCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> NotificationResponse:
    """
    Create a new notification (admin/internal use).
    """
    # Only admins can create notifications for other users
    if notification_data.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Not authorized to create notifications for other users"
        )
    
    # Check user notification settings
    user_settings = db.query(UserNotificationSettings).filter(
        UserNotificationSettings.user_id == notification_data.user_id
    ).first()
    
    # Create notification
    notification = Notification(
        user_id=notification_data.user_id,
        title=notification_data.title,
        message=notification_data.message,
        notification_type=notification_data.notification_type,
        data=notification_data.data or {},
        priority=notification_data.priority,
    )
    
    db.add(notification)
    db.commit()
    db.refresh(notification)
    
    # Send push/email/sms notifications based on user settings
    if user_settings:
        background_tasks.add_task(
            send_notification_to_channels,
            notification,
            user_settings,
            current_user.id
        )
    
    return NotificationResponse.from_orm(notification)


@router.put("/{notification_id}/read")
async def mark_as_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Mark a notification as read.
    """
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id,
    ).first()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    if not notification.is_read:
        notification.is_read = True
        notification.read_at = datetime.utcnow()
        db.commit()
    
    return {"message": "Notification marked as read"}


@router.put("/read/all")
async def mark_all_as_read(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Mark all notifications as read.
    """
    updated = db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_read == False,
        Notification.is_deleted == False,
    ).update({
        "is_read": True,
        "read_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    })
    
    db.commit()
    
    return {
        "message": f"Marked {updated} notifications as read",
        "updated_count": updated,
    }


@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Delete a notification (soft delete).
    """
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id,
    ).first()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    notification.is_deleted = True
    notification.deleted_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Notification deleted"}


@router.delete("/")
async def delete_all_notifications(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Delete all notifications for the current user.
    """
    deleted = db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_deleted == False,
    ).update({
        "is_deleted": True,
        "deleted_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    })
    
    db.commit()
    
    return {
        "message": f"Deleted {deleted} notifications",
        "deleted_count": deleted,
    }


@router.get("/settings", response_model=NotificationSettings)
async def get_notification_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> NotificationSettings:
    """
    Get notification settings for the current user.
    """
    settings = db.query(UserNotificationSettings).filter(
        UserNotificationSettings.user_id == current_user.id
    ).first()
    
    if not settings:
        # Create default settings
        settings = UserNotificationSettings(
            user_id=current_user.id,
            email_notifications=True,
            push_notifications=True,
            sms_notifications=False,
            incident_alerts=True,
            briefing_updates=True,
            verification_updates=True,
            reward_notifications=True,
            system_announcements=True,
            marketing_emails=False,
            quiet_hours_start=None,
            quiet_hours_end=None,
        )
        db.add(settings)
        db.commit()
        db.refresh(settings)
    
    return NotificationSettings.from_orm(settings)


@router.put("/settings", response_model=NotificationSettings)
async def update_notification_settings(
    settings_data: NotificationSettings,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> NotificationSettings:
    """
    Update notification settings.
    """
    settings = db.query(UserNotificationSettings).filter(
        UserNotificationSettings.user_id == current_user.id
    ).first()
    
    if not settings:
        settings = UserNotificationSettings(user_id=current_user.id)
        db.add(settings)
    
    # Update settings
    settings.email_notifications = settings_data.email_notifications
    settings.push_notifications = settings_data.push_notifications
    settings.sms_notifications = settings_data.sms_notifications
    settings.incident_alerts = settings_data.incident_alerts
    settings.briefing_updates = settings_data.briefing_updates
    settings.verification_updates = settings_data.verification_updates
    settings.reward_notifications = settings_data.reward_notifications
    settings.system_announcements = settings_data.system_announcements
    settings.marketing_emails = settings_data.marketing_emails
    settings.quiet_hours_start = settings_data.quiet_hours_start
    settings.quiet_hours_end = settings_data.quiet_hours_end
    settings.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(settings)
    
    return NotificationSettings.from_orm(settings)


@router.post("/test/push")
async def test_push_notification(
    title: str = "Test Notification",
    message: str = "This is a test push notification",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Send a test push notification.
    """
    # Check if user has push notifications enabled
    settings = db.query(UserNotificationSettings).filter(
        UserNotificationSettings.user_id == current_user.id
    ).first()
    
    if not settings or not settings.push_notifications:
        raise HTTPException(
            status_code=400,
            detail="Push notifications are disabled in your settings"
        )
    
    # Create test notification
    notification = Notification(
        user_id=current_user.id,
        title=title,
        message=message,
        notification_type="test",
        data={"test": True},
        priority="medium",
    )
    
    db.add(notification)
    db.commit()
    
    # Send push notification
    try:
        await send_push_notification(
            user_id=current_user.id,
            title=title,
            message=message,
            data={"notification_id": notification.id}
        )
        
        return {"message": "Test push notification sent successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send push notification: {str(e)}"
        )


@router.post("/test/email")
async def test_email_notification(
    subject: str = "Test Email Notification",
    body: str = "This is a test email notification",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Send a test email notification.
    """
    # Check if user has email notifications enabled
    settings = db.query(UserNotificationSettings).filter(
        UserNotificationSettings.user_id == current_user.id
    ).first()
    
    if not settings or not settings.email_notifications:
        raise HTTPException(
            status_code=400,
            detail="Email notifications are disabled in your settings"
        )
    
    # Send test email
    try:
        await send_email_notification(
            user_id=current_user.id,
            subject=subject,
            body=body,
            template_name="test_notification.html"
        )
        
        return {"message": "Test email notification sent successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send email notification: {str(e)}"
        )


async def send_notification_to_channels(
    notification: Notification,
    user_settings: UserNotificationSettings,
    sender_id: int
) -> None:
    """
    Send notification through appropriate channels based on user settings.
    """
    # Check if within quiet hours
    if is_quiet_hours(user_settings):
        return  # Don't send notifications during quiet hours
    
    # Send push notification
    if user_settings.push_notifications and notification.priority in ["high", "medium"]:
        try:
            await send_push_notification(
                user_id=notification.user_id,
                title=notification.title,
                message=notification.message,
                data={
                    "notification_id": notification.id,
                    "type": notification.notification_type,
                    **notification.data
                }
            )
        except Exception as e:
            print(f"Failed to send push notification: {e}")
    
    # Send email notification
    if user_settings.email_notifications and notification.priority == "high":
        try:
            await send_email_notification(
                user_id=notification.user_id,
                subject=notification.title,
                body=notification.message,
                template_name="notification.html",
                template_data={
                    "notification": notification,
                    "user_settings": user_settings,
                }
            )
        except Exception as e:
            print(f"Failed to send email notification: {e}")
    
    # Send SMS notification (only for critical alerts)
    if (user_settings.sms_notifications and 
        notification.priority == "critical" and
        notification.notification_type in ["incident_alert", "emergency"]):
        try:
            await send_sms_notification(
                user_id=notification.user_id,
                message=f"{notification.title}: {notification.message}"
            )
        except Exception as e:
            print(f"Failed to send SMS notification: {e}")


def is_quiet_hours(settings: UserNotificationSettings) -> bool:
    """
    Check if current time is within quiet hours.
    """
    if not settings.quiet_hours_start or not settings.quiet_hours_end:
        return False
    
    now = datetime.utcnow().time()
    return settings.quiet_hours_start <= now <= settings.quiet_hours_end


@router.get("/types")
async def get_notification_types() -> Dict[str, Any]:
    """
    Get available notification types and their descriptions.
    """
    return {
        "notification_types": [
            {
                "type": "incident_alert",
                "name": "Incident Alert",
                "description": "Alerts about nearby incidents",
                "default_priority": "high"
            },
            {
                "type": "verification_update",
                "name": "Verification Update",
                "description": "Updates on incident verification status",
                "default_priority": "medium"
            },
            {
                "type": "reward_notification",
                "name": "Reward Notification",
                "description": "Notifications about earned coins and rewards",
                "default_priority": "medium"
            },
            {
                "type": "briefing_ready",
                "name": "Briefing Ready",
                "description": "Notification when a briefing is generated",
                "default_priority": "low"
            },
            {
                "type": "system_announcement",
                "name": "System Announcement",
                "description": "Important system updates and announcements",
                "default_priority": "medium"
            },
            {
                "type": "community_update",
                "name": "Community Update",
                "description": "Updates from the community",
                "default_priority": "low"
            },
            {
                "type": "emergency",
                "name": "Emergency Alert",
                "description": "Critical emergency alerts",
                "default_priority": "critical"
            },
        ],
        "priority_levels": [
            {"level": "critical", "description": "Highest priority, immediate attention required"},
            {"level": "high", "description": "High priority, attention needed soon"},
            {"level": "medium", "description": "Normal priority"},
            {"level": "low", "description": "Low priority, informational only"},
        ]
    }


@router.post("/broadcast")
async def broadcast_notification(
    notification_data: NotificationCreate,
    background_tasks: BackgroundTasks,
    user_ids: Optional[List[int]] = None,
    user_roles: Optional[List[str]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Broadcast a notification to multiple users (admin only).
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Determine target users
    if user_ids:
        # Send to specific users
        target_users = db.query(User).filter(
            User.id.in_(user_ids),
            User.is_active == True,
        ).all()
    elif user_roles:
        # Send to users with specific roles
        target_users = db.query(User).filter(
            User.role.in_(user_roles),
            User.is_active == True,
        ).all()
    else:
        # Send to all active users
        target_users = db.query(User).filter(
            User.is_active == True,
        ).all()
    
    # Create notifications for each user
    notifications_created = 0
    for user in target_users:
        notification = Notification(
            user_id=user.id,
            title=notification_data.title,
            message=notification_data.message,
            notification_type=notification_data.notification_type,
            data=notification_data.data or {},
            priority=notification_data.priority,
        )
        db.add(notification)
        notifications_created += 1
    
    db.commit()
    
    # Send notifications in background
    background_tasks.add_task(
        process_broadcast_notifications,
        target_users,
        notification_data
    )
    
    return {
        "message": f"Notification broadcast to {notifications_created} users",
        "users_notified": notifications_created,
    }


async def process_broadcast_notifications(
    users: List[User],
    notification_data: NotificationCreate
) -> None:
    """
    Process broadcast notifications in background.
    """
    # This would handle sending notifications through various channels
    # For now, just log the broadcast
    print(f"Broadcast notification sent to {len(users)} users")
    print(f"Title: {notification_data.title}")
    print(f"Message: {notification_data.message}")