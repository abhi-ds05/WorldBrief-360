"""
Analytics subscriber for processing analytics and tracking events.

This subscriber handles all analytics-related events including:
- User behavior tracking
- Feature usage analytics
- Performance monitoring
- Business metrics
- Data pipeline metrics
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import defaultdict
from sqlalchemy import select

from app.core.logging_config import logger
from app.events.event_decorators import event_handler, retry_on_failure
from app.events.event_types import EventType, Event
from app.events.event_schemas import (
    UserActivityEvent,
    FeatureUsageEvent,
    UserRegisteredEvent,
    UserLoginEvent,
    IncidentReportedEvent,
    BriefingGeneratedEvent,
    ChatMessageSentEvent,
    WalletTransactionEvent,
    SystemAlertEvent,
    ExternalAPICalledEvent
)
from app.db.session import get_db
from app.db.models.user import User
from app.db.models.analytics import ( # type: ignore
    UserActivity,
    FeatureUsage,
    Session,
    PageView,
    Conversion,
    ErrorLog,
    PerformanceMetric,
    RetentionMetric,
    GeographicData,
    DeviceData,
    ReferralSource,
    ABTestResult,
    CustomerFeedback,
    SystemHealth,
    DataPipelineMetric
)


@event_handler(EventType.USER_REGISTERED)
@retry_on_failure(max_retries=3, delay=1.0)
async def track_user_registration(event_data: UserRegisteredEvent, event: Event):
    """
    Track user registration for analytics.
    
    Metrics tracked:
    - New user registrations
    - Registration sources
    - Time to first action
    - Conversion from visitor to registered user
    """
    try:
        async with get_db() as db:
            # Store registration event
            user_activity = UserActivity(
                user_id=event_data.user_id,
                activity_type="user_registration",
                activity_data={
                    "email": event_data.email,
                    "username": event_data.username,
                    "registration_source": event_data.registration_source,
                    "referrer_id": event_data.referrer_id,
                    "metadata": event_data.metadata
                },
                timestamp=event.timestamp,
                session_id=event.metadata.get('context', {}).get('session_id'),
                device_info=event.metadata.get('context', {}).get('device_info', {}),
                location=event.metadata.get('context', {}).get('location', {})
            )
            db.add(user_activity)
            
            # Update referral source tracking if referrer exists
            if event_data.referrer_id:
                referral = ReferralSource(
                    referrer_id=event_data.referrer_id,
                    referred_user_id=event_data.user_id,
                    registration_source=event_data.registration_source,
                    timestamp=event.timestamp
                )
                db.add(referral)
            
            # Create initial session if not exists
            session_id = event.metadata.get('context', {}).get('session_id')
            if session_id:
                session = await db.get(Session, session_id)
                if not session:
                    session = Session(
                        id=session_id,
                        user_id=event_data.user_id,
                        start_time=event.timestamp,
                        device_info=event.metadata.get('context', {}).get('device_info', {}),
                        location=event.metadata.get('context', {}).get('location', {})
                    )
                    db.add(session)
            
            await db.commit()
            
            # Update real-time analytics cache
            await _update_realtime_metrics("user_registrations", 1)
            await _update_daily_registrations(event.timestamp.date())
            
            logger.info(f"Tracked user registration for user {event_data.user_id}")
            
    except Exception as e:
        logger.error(f"Failed to track user registration: {e}")
        raise


@event_handler(EventType.USER_LOGIN)
@retry_on_failure(max_retries=2, delay=0.5)
async def track_user_login(event_data: UserLoginEvent, event: Event):
    """
    Track user login events.
    
    Metrics tracked:
    - Login frequency
    - Login methods
    - Failed login attempts
    - Session duration
    """
    try:
        async with get_db() as db:
            # Store login activity
            user_activity = UserActivity(
                user_id=event_data.user_id,
                activity_type="user_login",
                activity_data={
                    "ip_address": event_data.ip_address,
                    "login_method": event_data.login_method,
                    "device_id": event_data.device_id,
                    "location": event_data.location,
                    "success": event_data.success,
                    "failure_reason": event_data.failure_reason
                },
                timestamp=event.timestamp,
                session_id=event.metadata.get('context', {}).get('session_id'),
                device_info=event.metadata.get('context', {}).get('device_info', {}),
                location=event_data.location or event.metadata.get('context', {}).get('location', {})
            )
            db.add(user_activity)
            
            # Update session if exists
            session_id = event.metadata.get('context', {}).get('session_id')
            if session_id and event_data.success:
                session = await db.get(Session, session_id)
                if session:
                    session.user_id = event_data.user_id
                    session.last_activity = event.timestamp
                else:
                    session = Session(
                        id=session_id,
                        user_id=event_data.user_id,
                        start_time=event.timestamp,
                        device_info=event.metadata.get('context', {}).get('device_info', {}),
                        location=event_data.location or event.metadata.get('context', {}).get('location', {})
                    )
                    db.add(session)
            
            await db.commit()
            
            # Update metrics
            if event_data.success:
                await _update_realtime_metrics("successful_logins", 1)
                await _update_dau_wau_mau(event_data.user_id, event.timestamp)
            else:
                await _update_realtime_metrics("failed_logins", 1)
                await _track_security_event("failed_login_attempt", event_data)
            
            logger.debug(f"Tracked user login for user {event_data.user_id}")
            
    except Exception as e:
        logger.error(f"Failed to track user login: {e}")
        raise


@event_handler(EventType.INCIDENT_REPORTED)
async def track_incident_report(event_data: IncidentReportedEvent, event: Event):
    """
    Track incident reporting metrics.
    
    Metrics tracked:
    - Incident reporting volume
    - Categories and severity distribution
    - Reporting frequency per user
    - Geographic distribution
    """
    try:
        async with get_db() as db:
            # Track incident reporting activity
            user_activity = UserActivity(
                user_id=event_data.reporter_id,
                activity_type="incident_reported",
                activity_data={
                    "incident_id": event_data.incident_id,
                    "category": event_data.category,
                    "severity": event_data.severity,
                    "location": event_data.location,
                    "tags": event_data.tags
                },
                timestamp=event.timestamp,
                session_id=event.metadata.get('context', {}).get('session_id')
            )
            db.add(user_activity)
            
            # Update geographic distribution
            if event_data.coordinates:
                geo_data = GeographicData(
                    user_id=event_data.reporter_id,
                    incident_id=event_data.incident_id,
                    latitude=event_data.coordinates['lat'],
                    longitude=event_data.coordinates['lng'],
                    activity_type="incident_report",
                    timestamp=event.timestamp,
                    metadata={
                        "category": event_data.category,
                        "severity": event_data.severity
                    }
                )
                db.add(geo_data)
            
            await db.commit()
            
            # Update real-time metrics
            await _update_realtime_metrics("incidents_reported", 1)
            await _update_category_metrics(event_data.category, 1)
            await _update_severity_metrics(event_data.severity, 1)
            
            logger.info(f"Tracked incident report {event_data.incident_id}")
            
    except Exception as e:
        logger.error(f"Failed to track incident report: {e}")
        raise


@event_handler(EventType.BRIEFING_GENERATED)
async def track_briefing_generation(event_data: BriefingGeneratedEvent, event: Event):
    """
    Track briefing generation metrics.
    
    Metrics tracked:
    - Briefing generation volume
    - Topic popularity
    - Briefing levels used
    - Generation time
    - User engagement with briefings
    """
    try:
        async with get_db() as db:
            # Track briefing generation
            feature_usage = FeatureUsage(
                user_id=event_data.user_id,
                feature_name="briefing_generation",
                feature_action="generate",
                duration_seconds=event_data.duration_seconds,
                success=True,
                metadata={
                    "briefing_id": event_data.briefing_id,
                    "topic": event_data.topic,
                    "level": event_data.level,
                    "language": event_data.language,
                    "source_count": event_data.source_count,
                    "has_audio": event_data.has_audio,
                    "has_images": event_data.has_images
                },
                timestamp=event.timestamp
            )
            db.add(feature_usage)
            
            # Update topic popularity
            await _update_topic_popularity(event_data.topic, 1)
            
            # Update briefing level usage
            await _update_briefing_level_metrics(event_data.level, 1)
            
            await db.commit()
            
            # Update real-time metrics
            await _update_realtime_metrics("briefings_generated", 1)
            await _update_average_briefing_duration(event_data.duration_seconds)
            
            logger.debug(f"Tracked briefing generation {event_data.briefing_id}")
            
    except Exception as e:
        logger.error(f"Failed to track briefing generation: {e}")
        raise


@event_handler(EventType.CHAT_MESSAGE_SENT)
async def track_chat_interaction(event_data: ChatMessageSentEvent, event: Event):
    """
    Track chat interactions.
    
    Metrics tracked:
    - Chat message volume
    - Conversation length
    - Response times
    - AI vs user message ratio
    - Citation usage
    """
    try:
        async with get_db() as db:
            # Track chat activity
            user_activity = UserActivity(
                user_id=event_data.user_id,
                activity_type="chat_message",
                activity_data={
                    "message_id": event_data.message_id,
                    "conversation_id": event_data.conversation_id,
                    "message_type": event_data.message_type,
                    "is_ai_response": event_data.is_ai_response,
                    "parent_message_id": event_data.parent_message_id,
                    "citation_count": len(event_data.citations)
                },
                timestamp=event.timestamp,
                session_id=event.metadata.get('context', {}).get('session_id')
            )
            db.add(user_activity)
            
            # Update conversation metrics
            await _update_conversation_metrics(
                event_data.conversation_id,
                event_data.is_ai_response,
                len(event_data.citations)
            )
            
            await db.commit()
            
            # Update real-time metrics
            if event_data.is_ai_response:
                await _update_realtime_metrics("ai_responses", 1)
            else:
                await _update_realtime_metrics("user_messages", 1)
            
            await _update_average_message_length(len(event_data.content))
            
            logger.debug(f"Tracked chat message {event_data.message_id}")
            
    except Exception as e:
        logger.error(f"Failed to track chat interaction: {e}")
        raise


@event_handler(EventType.WALLET_TRANSACTION)
async def track_wallet_transaction(event_data: WalletTransactionEvent, event: Event):
    """
    Track wallet transactions.
    
    Metrics tracked:
    - Transaction volume and value
    - Transaction types distribution
    - User spending patterns
    - Reward redemption rates
    """
    try:
        async with get_db() as db:
            # Track wallet activity
            user_activity = UserActivity(
                user_id=event_data.user_id,
                activity_type="wallet_transaction",
                activity_data={
                    "transaction_id": event_data.transaction_id,
                    "transaction_type": event_data.transaction_type,
                    "amount": event_data.amount,
                    "description": event_data.description,
                    "balance_before": event_data.balance_before,
                    "balance_after": event_data.balance_after
                },
                timestamp=event.timestamp,
                session_id=event.metadata.get('context', {}).get('session_id')
            )
            db.add(user_activity)
            
            await db.commit()
            
            # Update transaction metrics
            await _update_transaction_metrics(
                event_data.transaction_type,
                event_data.amount,
                event_data.user_id
            )
            
            # Update user wallet balance trends
            await _update_wallet_balance_trend(
                event_data.user_id,
                event_data.balance_after,
                event.timestamp
            )
            
            logger.debug(f"Tracked wallet transaction {event_data.transaction_id}")
            
    except Exception as e:
        logger.error(f"Failed to track wallet transaction: {e}")
        raise


@event_handler([
    EventType.ANALYTICS_EVENT_TRACKED,
    EventType.ANALYTICS_FEATURE_USED,
    EventType.USER_ACTIVITY
])
async def track_feature_usage(event_data: FeatureUsageEvent, event: Event):
    """
    Track general feature usage.
    
    Metrics tracked:
    - Feature adoption rates
    - Usage frequency
    - Success/failure rates
    - Time spent on features
    """
    try:
        async with get_db() as db:
            # Store feature usage
            feature_usage = FeatureUsage(
                user_id=event_data.user_id,
                feature_name=event_data.feature_name,
                feature_action=event_data.feature_action,
                duration_seconds=event_data.duration_seconds,
                success=event_data.success,
                error_message=event_data.error_message,
                metadata=event_data.metadata,
                timestamp=event.timestamp
            )
            db.add(feature_usage)
            
            await db.commit()
            
            # Update feature adoption metrics
            await _update_feature_adoption_metrics(
                event_data.feature_name,
                event_data.feature_action,
                event_data.success
            )
            
            # Update user engagement score
            if event_data.success and event_data.duration_seconds:
                await _update_user_engagement_score(
                    event_data.user_id,
                    event_data.feature_name,
                    event_data.duration_seconds
                )
            
            logger.debug(f"Tracked feature usage: {event_data.feature_name}.{event_data.feature_action}")
            
    except Exception as e:
        logger.error(f"Failed to track feature usage: {e}")
        raise


@event_handler(EventType.CONTENT_VIEWED)
async def track_content_view(event_data: Dict[str, Any], event: Event):
    """
    Track content viewing metrics.
    
    Metrics tracked:
    - Page views
    - Time on page
    - Bounce rates
    - Content popularity
    """
    try:
        async with get_db() as db:
            # Store page view
            page_view = PageView(
                user_id=event_data.get('user_id'),
                page_path=event_data.get('path'),
                page_title=event_data.get('title'),
                referrer=event_data.get('referrer'),
                duration_seconds=event_data.get('duration_seconds'),
                session_id=event.metadata.get('context', {}).get('session_id'),
                device_info=event.metadata.get('context', {}).get('device_info', {}),
                timestamp=event.timestamp
            )
            db.add(page_view)
            
            await db.commit()
            
            # Update content popularity
            await _update_content_popularity(event_data.get('path'), 1)
            
            # Update bounce rate if duration is short
            if event_data.get('duration_seconds', 0) < 5:
                await _update_realtime_metrics("bounces", 1)
            
            logger.debug(f"Tracked content view: {event_data.get('path')}")
            
    except Exception as e:
        logger.error(f"Failed to track content view: {e}")
        raise


@event_handler(EventType.EXTERNAL_API_CALLED)
async def track_api_performance(event_data: ExternalAPICalledEvent, event: Event):
    """
    Track API performance metrics.
    
    Metrics tracked:
    - API response times
    - Success/failure rates
    - Error patterns
    - API usage by endpoint
    """
    try:
        async with get_db() as db:
            # Store performance metric
            performance = PerformanceMetric(
                api_name=event_data.api_name,
                endpoint=event_data.endpoint,
                method=event_data.method,
                status_code=event_data.status_code,
                duration_ms=event_data.duration_ms,
                success=event_data.success,
                error_message=event_data.error_message,
                timestamp=event.timestamp,
                metadata={
                    "request_id": event_data.request_id,
                    "user_id": event.metadata.get('context', {}).get('user_id')
                }
            )
            db.add(performance)
            
            await db.commit()
            
            # Update API performance aggregates
            await _update_api_performance_metrics(
                event_data.api_name,
                event_data.endpoint,
                event_data.duration_ms,
                event_data.success
            )
            
            # Alert on slow or failing APIs
            if event_data.duration_ms > 5000:  # 5 seconds threshold
                await _alert_slow_api(event_data)
            
            if not event_data.success:
                await _alert_api_failure(event_data)
            
            logger.debug(f"Tracked API call: {event_data.api_name} - {event_data.endpoint}")
            
    except Exception as e:
        logger.error(f"Failed to track API performance: {e}")
        raise


@event_handler(EventType.ANALYTICS_CONVERSION_COMPLETED)
async def track_conversion_funnel(event_data: Dict[str, Any], event: Event):
    """
    Track conversion funnel metrics.
    
    Metrics tracked:
    - Conversion rates
    - Funnel drop-off points
    - Time to conversion
    - Conversion by source
    """
    try:
        async with get_db() as db:
            # Store conversion event
            conversion = Conversion(
                user_id=event_data.get('user_id'),
                conversion_type=event_data.get('conversion_type'),
                conversion_value=event_data.get('conversion_value'),
                funnel_stage=event_data.get('funnel_stage'),
                source=event_data.get('source'),
                campaign=event_data.get('campaign'),
                session_id=event.metadata.get('context', {}).get('session_id'),
                timestamp=event.timestamp,
                metadata=event_data.get('metadata', {})
            )
            db.add(conversion)
            
            await db.commit()
            
            # Update conversion metrics
            await _update_conversion_metrics(
                event_data.get('conversion_type'),
                event_data.get('funnel_stage'),
                event_data.get('conversion_value', 0)
            )
            
            logger.info(f"Tracked conversion: {event_data.get('conversion_type')}")
            
    except Exception as e:
        logger.error(f"Failed to track conversion funnel: {e}")
        raise


@event_handler(EventType.ANALYTICS_SESSION_ENDED)
async def track_session_duration(event_data: Dict[str, Any], event: Event):
    """
    Track session duration metrics.
    
    Metrics tracked:
    - Average session duration
    - Session frequency
    - Time between sessions
    - Session depth (pages per session)
    """
    try:
        async with get_db() as db:
            # Update session end time
            session_id = event_data.get('session_id')
            if session_id:
                session = await db.get(Session, session_id)
                if session:
                    session.end_time = event.timestamp
                    session.duration_seconds = event_data.get('duration_seconds', 0)
                    session.page_count = event_data.get('page_count', 0)
                    session.activity_count = event_data.get('activity_count', 0)
            
            await db.commit()
            
            # Update session metrics
            if session_id and event_data.get('duration_seconds'):
                await _update_session_metrics(event_data['duration_seconds'])
            
            logger.debug(f"Tracked session end: {session_id}")
            
    except Exception as e:
        logger.error(f"Failed to track session duration: {e}")
        raise


@event_handler(EventType.USER_ACTIVITY)
async def track_retention_metrics(event_data: UserActivityEvent, event: Event):
    """
    Track user retention metrics.
    
    Metrics tracked:
    - Daily/Weekly/Monthly Active Users (DAU/WAU/MAU)
    - Retention cohorts
    - Churn rates
    - User lifetime value
    """
    try:
        # Update retention metrics in background
        asyncio.create_task(_update_retention_metrics_background(event_data, event))
        
        logger.debug(f"Tracking retention for user {event_data.user_id}")
        
    except Exception as e:
        logger.error(f"Failed to track retention metrics: {e}")
        raise


@event_handler(EventType.LOCATION_DETECTED)
async def track_geographic_distribution(event_data: Dict[str, Any], event: Event):
    """
    Track geographic distribution of users and activities.
    
    Metrics tracked:
    - User locations
    - Activity hotspots
    - Regional usage patterns
    - Language preferences by region
    """
    try:
        async with get_db() as db:
            if event_data.get('coordinates'):
                geo_data = GeographicData(
                    user_id=event_data.get('user_id'),
                    latitude=event_data['coordinates']['lat'],
                    longitude=event_data['coordinates']['lng'],
                    activity_type=event_data.get('activity_type', 'location_detected'),
                    timestamp=event.timestamp,
                    metadata=event_data.get('metadata', {})
                )
                db.add(geo_data)
                
                await db.commit()
                
                # Update geographic aggregates
                await _update_geographic_aggregates(
                    event_data['coordinates']['lat'],
                    event_data['coordinates']['lng'],
                    event_data.get('activity_type')
                )
                
                logger.debug(f"Tracked geographic location for user {event_data.get('user_id')}")
                
    except Exception as e:
        logger.error(f"Failed to track geographic distribution: {e}")
        raise


@event_handler(EventType.ANALYTICS_EVENT_TRACKED)
async def track_device_usage(event_data: Dict[str, Any], event: Event):
    """
    Track device usage metrics.
    
    Metrics tracked:
    - Device types and models
    - OS and browser versions
    - Screen resolutions
    - App vs web usage
    """
    try:
        device_info = event.metadata.get('context', {}).get('device_info', {})
        if device_info:
            async with get_db() as db:
                device_data = DeviceData(
                    user_id=event_data.get('user_id'),
                    device_type=device_info.get('type'),
                    device_model=device_info.get('model'),
                    os=device_info.get('os'),
                    os_version=device_info.get('os_version'),
                    browser=device_info.get('browser'),
                    browser_version=device_info.get('browser_version'),
                    screen_resolution=device_info.get('screen_resolution'),
                    app_version=device_info.get('app_version'),
                    timestamp=event.timestamp
                )
                db.add(device_data)
                
                await db.commit()
                
                # Update device usage aggregates
                await _update_device_usage_metrics(device_info)
                
                logger.debug(f"Tracked device usage for user {event_data.get('user_id')}")
                
    except Exception as e:
        logger.error(f"Failed to track device usage: {e}")
        raise


@event_handler(EventType.USER_REGISTERED)
async def track_referral_sources(event_data: UserRegisteredEvent, event: Event):
    """
    Track referral sources and attribution.
    
    Metrics tracked:
    - Referral sources effectiveness
    - Multi-touch attribution
    - Campaign performance
    - Channel ROI
    """
    try:
        # This is handled in track_user_registration
        # Additional attribution logic can be added here
        
        # Example: Track UTM parameters
        metadata = event_data.metadata or {}
        utm_source = metadata.get('utm_source')
        utm_medium = metadata.get('utm_medium')
        utm_campaign = metadata.get('utm_campaign')
        
        if any([utm_source, utm_medium, utm_campaign]):
            async with get_db() as db:
                # Store UTM data
                # ... implementation
                pass
        
        logger.debug(f"Tracked referral source for user {event_data.user_id}")
        
    except Exception as e:
        logger.error(f"Failed to track referral sources: {e}")
        raise


@event_handler(EventType.ANALYTICS_EVENT_TRACKED)
async def track_a_b_test_results(event_data: Dict[str, Any], event: Event):
    """
    Track A/B test results.
    
    Metrics tracked:
    - Test variant performance
    - Statistical significance
    - Impact on key metrics
    - Test duration and sample size
    """
    try:
        test_data = event_data.get('ab_test', {})
        if test_data:
            async with get_db() as db:
                ab_test = ABTestResult(
                    test_id=test_data.get('test_id'),
                    user_id=event_data.get('user_id'),
                    variant=test_data.get('variant'),
                    metric=test_data.get('metric'),
                    value=test_data.get('value'),
                    timestamp=event.timestamp,
                    metadata=test_data.get('metadata', {})
                )
                db.add(ab_test)
                
                await db.commit()
                
                # Update test aggregates
                await _update_ab_test_aggregates(
                    test_data.get('test_id'),
                    test_data.get('variant'),
                    test_data.get('metric'),
                    test_data.get('value')
                )
                
                logger.debug(f"Tracked A/B test result for test {test_data.get('test_id')}")
                
    except Exception as e:
        logger.error(f"Failed to track A/B test results: {e}")
        raise


@event_handler(EventType.BRIEFING_FEEDBACK_GIVEN)
async def track_customer_feedback(event_data: Dict[str, Any], event: Event):
    """
    Track customer feedback and satisfaction.
    
    Metrics tracked:
    - Net Promoter Score (NPS)
    - Customer Satisfaction (CSAT)
    - Customer Effort Score (CES)
    - Feedback sentiment analysis
    """
    try:
        async with get_db() as db:
            feedback = CustomerFeedback(
                user_id=event_data.get('user_id'),
                feedback_type=event_data.get('feedback_type'),
                rating=event_data.get('rating'),
                comments=event_data.get('comments'),
                source=event_data.get('source'),
                timestamp=event.timestamp,
                metadata=event_data.get('metadata', {})
            )
            db.add(feedback)
            
            await db.commit()
            
            # Update feedback aggregates
            await _update_feedback_metrics(
                event_data.get('feedback_type'),
                event_data.get('rating'),
                event_data.get('comments')
            )
            
            # Analyze sentiment if comments exist
            if event_data.get('comments'):
                await _analyze_feedback_sentiment(
                    event_data.get('comments'),
                    event_data.get('user_id')
                )
            
            logger.info(f"Tracked customer feedback from user {event_data.get('user_id')}")
            
    except Exception as e:
        logger.error(f"Failed to track customer feedback: {e}")
        raise


@event_handler(EventType.MONITORING_HEALTH_CHECK_PASSED)
async def track_system_health(event_data: Dict[str, Any], event: Event):
    """
    Track system health metrics.
    
    Metrics tracked:
    - Uptime and availability
    - Resource utilization
    - Error rates
    - Performance benchmarks
    """
    try:
        async with get_db() as db:
            health = SystemHealth(
                component=event_data.get('component'),
                check_type=event_data.get('check_type'),
                status=event_data.get('status'),
                response_time=event_data.get('response_time'),
                error_count=event_data.get('error_count'),
                timestamp=event.timestamp,
                metadata=event_data.get('metadata', {})
            )
            db.add(health)
            
            await db.commit()
            
            # Update system health aggregates
            await _update_system_health_metrics(
                event_data.get('component'),
                event_data.get('status'),
                event_data.get('response_time')
            )
            
            # Alert on system health issues
            if event_data.get('status') != 'healthy':
                await _alert_system_health_issue(event_data)
            
            logger.debug(f"Tracked system health for {event_data.get('component')}")
            
    except Exception as e:
        logger.error(f"Failed to track system health: {e}")
        raise


@event_handler(EventType.DATA_INGESTION_COMPLETED)
async def track_data_pipeline_metrics(event_data: Dict[str, Any], event: Event):
    """
    Track data pipeline metrics.
    
    Metrics tracked:
    - Pipeline execution times
    - Data volume processed
    - Success/failure rates
    - Data quality metrics
    """
    try:
        async with get_db() as db:
            pipeline_metric = DataPipelineMetric(
                pipeline_name=event_data.get('pipeline_name'),
                execution_id=event_data.get('execution_id'),
                status=event_data.get('status'),
                duration_seconds=event_data.get('duration_seconds'),
                records_processed=event_data.get('records_processed'),
                success_count=event_data.get('success_count'),
                failure_count=event_data.get('failure_count'),
                timestamp=event.timestamp,
                metadata=event_data.get('metadata', {})
            )
            db.add(pipeline_metric)
            
            await db.commit()
            
            # Update pipeline aggregates
            await _update_pipeline_metrics(
                event_data.get('pipeline_name'),
                event_data.get('status'),
                event_data.get('duration_seconds'),
                event_data.get('records_processed')
            )
            
            logger.info(f"Tracked data pipeline metrics for {event_data.get('pipeline_name')}")
            
    except Exception as e:
        logger.error(f"Failed to track data pipeline metrics: {e}")
        raise


# ============== Helper Functions ==============

async def _update_realtime_metrics(metric_name: str, value: int = 1):
    """Update real-time metrics in cache."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Increment counter
        key = f"analytics:realtime:{metric_name}"
        await redis.incrby(key, value)
        
        # Set expiry to auto-clean old metrics
        await redis.expire(key, 3600)  # 1 hour
        
    except Exception as e:
        logger.error(f"Failed to update realtime metrics: {e}")


async def _update_daily_registrations(date: datetime.date):
    """Update daily registration counts."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        key = f"analytics:daily:registrations:{date.isoformat()}"
        await redis.incrby(key, 1)
        
    except Exception as e:
        logger.error(f"Failed to update daily registrations: {e}")


async def _update_dau_wau_mau(user_id: str, timestamp: datetime):
    """Update DAU/WAU/MAU metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        date_str = timestamp.date().isoformat()
        
        # Daily Active Users
        dau_key = f"analytics:dau:{date_str}"
        await redis.sadd(dau_key, user_id)
        await redis.expire(dau_key, 86400 * 7)  # Keep for 7 days
        
        # Weekly Active Users (last 7 days)
        for i in range(7):
            day = (timestamp - timedelta(days=i)).date().isoformat()
            wau_key = f"analytics:wau:{day}"
            await redis.sadd(wau_key, user_id)
            await redis.expire(wau_key, 86400 * 30)  # Keep for 30 days
        
        # Monthly Active Users (last 30 days)
        for i in range(30):
            day = (timestamp - timedelta(days=i)).date().isoformat()
            mau_key = f"analytics:mau:{day}"
            await redis.sadd(mau_key, user_id)
            await redis.expire(mau_key, 86400 * 60)  # Keep for 60 days
        
    except Exception as e:
        logger.error(f"Failed to update DAU/WAU/MAU: {e}")


async def _track_security_event(event_type: str, event_data: Any):
    """Track security-related events."""
    try:
        # Log security event
        logger.warning(f"Security event: {event_type} - {event_data}")
        
        # Could send to security monitoring system
        # await security_service.track_event(event_type, event_data)
        
    except Exception as e:
        logger.error(f"Failed to track security event: {e}")


async def _update_category_metrics(category: str, count: int = 1):
    """Update incident category metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        key = f"analytics:categories:{category}"
        await redis.incrby(key, count)
        
    except Exception as e:
        logger.error(f"Failed to update category metrics: {e}")


async def _update_severity_metrics(severity: str, count: int = 1):
    """Update incident severity metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        key = f"analytics:severities:{severity}"
        await redis.incrby(key, count)
        
    except Exception as e:
        logger.error(f"Failed to update severity metrics: {e}")


async def _update_topic_popularity(topic: str, count: int = 1):
    """Update topic popularity metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        key = f"analytics:topics:{topic}"
        await redis.incrby(key, count)
        
    except Exception as e:
        logger.error(f"Failed to update topic popularity: {e}")


async def _update_briefing_level_metrics(level: str, count: int = 1):
    """Update briefing level usage metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        key = f"analytics:briefing_levels:{level}"
        await redis.incrby(key, count)
        
    except Exception as e:
        logger.error(f"Failed to update briefing level metrics: {e}")


async def _update_average_briefing_duration(duration: int):
    """Update average briefing duration."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Use Redis sorted sets or hyperloglog for averages
        key = "analytics:briefing_durations"
        await redis.rpush(key, duration)
        
        # Keep only last 1000 durations
        await redis.ltrim(key, -1000, -1)
        
    except Exception as e:
        logger.error(f"Failed to update average briefing duration: {e}")


async def _update_conversation_metrics(conversation_id: str, is_ai: bool, citation_count: int):
    """Update conversation metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Track conversation length
        key = f"analytics:conversation:{conversation_id}:messages"
        await redis.incr(key)
        
        # Track AI vs human ratio
        if is_ai:
            ai_key = f"analytics:conversation:{conversation_id}:ai_messages"
            await redis.incr(ai_key)
        
        # Track citation usage
        if citation_count > 0:
            citation_key = f"analytics:conversation:{conversation_id}:citations"
            await redis.incrby(citation_key, citation_count)
        
    except Exception as e:
        logger.error(f"Failed to update conversation metrics: {e}")


async def _update_average_message_length(length: int):
    """Update average message length."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        key = "analytics:message_lengths"
        await redis.rpush(key, length)
        await redis.ltrim(key, -10000, -1)  # Keep last 10k messages
        
    except Exception as e:
        logger.error(f"Failed to update average message length: {e}")


async def _update_transaction_metrics(transaction_type: str, amount: int, user_id: str):
    """Update transaction metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Update transaction type counts
        type_key = f"analytics:transactions:type:{transaction_type}"
        await redis.incr(type_key)
        
        # Update transaction volume
        volume_key = f"analytics:transactions:volume:{transaction_type}"
        await redis.incrby(volume_key, amount)
        
        # Update user transaction history
        user_key = f"analytics:user:{user_id}:transactions"
        await redis.incr(user_key)
        
    except Exception as e:
        logger.error(f"Failed to update transaction metrics: {e}")


async def _update_wallet_balance_trend(user_id: str, balance: int, timestamp: datetime):
    """Update wallet balance trend."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        key = f"analytics:user:{user_id}:balance_trend"
        data = {
            "timestamp": timestamp.isoformat(),
            "balance": balance
        }
        await redis.rpush(key, json.dumps(data))
        
        # Keep only last 100 balance points
        await redis.ltrim(key, -100, -1)
        
    except Exception as e:
        logger.error(f"Failed to update wallet balance trend: {e}")


async def _update_feature_adoption_metrics(feature_name: str, action: str, success: bool):
    """Update feature adoption metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Track feature usage
        usage_key = f"analytics:feature:{feature_name}:{action}"
        await redis.incr(usage_key)
        
        # Track success rate
        if success:
            success_key = f"analytics:feature:{feature_name}:{action}:success"
            await redis.incr(success_key)
        else:
            failure_key = f"analytics:feature:{feature_name}:{action}:failure"
            await redis.incr(failure_key)
        
    except Exception as e:
        logger.error(f"Failed to update feature adoption metrics: {e}")


async def _update_user_engagement_score(user_id: str, feature_name: str, duration: int):
    """Update user engagement score."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Calculate engagement score (simplified)
        score = min(duration / 60, 10)  # 10 points per minute, capped at 10
        
        # Update user engagement
        engagement_key = f"analytics:user:{user_id}:engagement"
        await redis.zincrby("analytics:user_engagement_scores", score, user_id)
        
        # Update feature-specific engagement
        feature_key = f"analytics:feature:{feature_name}:engagement"
        await redis.zincrby(feature_key, score, user_id)
        
    except Exception as e:
        logger.error(f"Failed to update user engagement score: {e}")


async def _update_content_popularity(path: str, count: int = 1):
    """Update content popularity."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        key = f"analytics:content_popularity"
        await redis.zincrby(key, count, path)
        
    except Exception as e:
        logger.error(f"Failed to update content popularity: {e}")


async def _update_api_performance_metrics(api_name: str, endpoint: str, duration: int, success: bool):
    """Update API performance metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Track response times
        time_key = f"analytics:api:{api_name}:{endpoint}:response_times"
        await redis.rpush(time_key, duration)
        await redis.ltrim(time_key, -1000, -1)
        
        # Track success rate
        if success:
            success_key = f"analytics:api:{api_name}:{endpoint}:success"
            await redis.incr(success_key)
        else:
            failure_key = f"analytics:api:{api_name}:{endpoint}:failure"
            await redis.incr(failure_key)
        
    except Exception as e:
        logger.error(f"Failed to update API performance metrics: {e}")


async def _alert_slow_api(event_data: ExternalAPICalledEvent):
    """Alert on slow API calls."""
    try:
        logger.warning(
            f"Slow API call detected: {event_data.api_name} - {event_data.endpoint} "
            f"took {event_data.duration_ms}ms"
        )
        
        # Could send to alerting system
        # await alerting_service.send_alert(...)
        
    except Exception as e:
        logger.error(f"Failed to alert on slow API: {e}")


async def _alert_api_failure(event_data: ExternalAPICalledEvent):
    """Alert on API failures."""
    try:
        logger.error(
            f"API call failed: {event_data.api_name} - {event_data.endpoint} "
            f"status: {event_data.status_code}, error: {event_data.error_message}"
        )
        
        # Could send to alerting system
        # await alerting_service.send_alert(...)
        
    except Exception as e:
        logger.error(f"Failed to alert on API failure: {e}")


async def _update_conversion_metrics(conversion_type: str, funnel_stage: str, value: float):
    """Update conversion metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Track conversions by type
        type_key = f"analytics:conversions:{conversion_type}"
        await redis.incr(type_key)
        
        # Track funnel stage conversions
        if funnel_stage:
            stage_key = f"analytics:funnel:{funnel_stage}"
            await redis.incr(stage_key)
        
        # Track conversion value
        value_key = f"analytics:conversion_value:{conversion_type}"
        await redis.incrbyfloat(value_key, value)
        
    except Exception as e:
        logger.error(f"Failed to update conversion metrics: {e}")


async def _update_session_metrics(duration: int):
    """Update session metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Track session durations
        key = "analytics:session_durations"
        await redis.rpush(key, duration)
        await redis.ltrim(key, -10000, -1)
        
    except Exception as e:
        logger.error(f"Failed to update session metrics: {e}")


async def _update_retention_metrics_background(event_data: UserActivityEvent, event: Event):
    """Background task to update retention metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        user_id = event_data.user_id
        date = event.timestamp.date()
        
        # Track user activity date
        activity_key = f"analytics:user_activity:{user_id}"
        await redis.sadd(activity_key, date.isoformat())
        await redis.expire(activity_key, 86400 * 90)  # Keep for 90 days
        
        # Update retention cohorts
        # This would involve more complex cohort analysis
        # For now, just track that the user was active
        
        logger.debug(f"Updated retention metrics for user {user_id}")
        
    except Exception as e:
        logger.error(f"Failed to update retention metrics: {e}")


async def _update_geographic_aggregates(lat: float, lng: float, activity_type: str):
    """Update geographic aggregates."""
    try:
        # Round coordinates to reduce precision for aggregation
        lat_rounded = round(lat, 2)
        lng_rounded = round(lng, 2)
        
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Track activity by location
        location_key = f"analytics:location:{lat_rounded}:{lng_rounded}"
        await redis.incr(location_key)
        
        # Track activity type by location
        if activity_type:
            type_key = f"analytics:location:{lat_rounded}:{lng_rounded}:{activity_type}"
            await redis.incr(type_key)
        
    except Exception as e:
        logger.error(f"Failed to update geographic aggregates: {e}")


async def _update_device_usage_metrics(device_info: Dict[str, Any]):
    """Update device usage metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Track device types
        if device_info.get('type'):
            type_key = f"analytics:devices:type:{device_info['type']}"
            await redis.incr(type_key)
        
        # Track OS versions
        if device_info.get('os') and device_info.get('os_version'):
            os_key = f"analytics:devices:os:{device_info['os']}:{device_info['os_version']}"
            await redis.incr(os_key)
        
        # Track browsers
        if device_info.get('browser') and device_info.get('browser_version'):
            browser_key = f"analytics:devices:browser:{device_info['browser']}:{device_info['browser_version']}"
            await redis.incr(browser_key)
        
    except Exception as e:
        logger.error(f"Failed to update device usage metrics: {e}")


async def _update_ab_test_aggregates(test_id: str, variant: str, metric: str, value: float):
    """Update A/B test aggregates."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Track variant assignments
        assignment_key = f"analytics:ab_test:{test_id}:{variant}:assignments"
        await redis.incr(assignment_key)
        
        # Track metric values
        if metric and value is not None:
            metric_key = f"analytics:ab_test:{test_id}:{variant}:{metric}"
            await redis.rpush(metric_key, value)
            await redis.ltrim(metric_key, -10000, -1)
        
    except Exception as e:
        logger.error(f"Failed to update A/B test aggregates: {e}")


async def _update_feedback_metrics(feedback_type: str, rating: int, comments: str):
    """Update feedback metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Track feedback counts by type
        if feedback_type:
            type_key = f"analytics:feedback:{feedback_type}:count"
            await redis.incr(type_key)
        
        # Track ratings
        if rating is not None:
            rating_key = f"analytics:feedback:{feedback_type}:rating:{rating}"
            await redis.incr(rating_key)
            
            # Track average rating
            avg_key = f"analytics:feedback:{feedback_type}:average_rating"
            await redis.rpush(avg_key, rating)
            await redis.ltrim(avg_key, -1000, -1)
        
        # Track comment length if exists
        if comments:
            length_key = f"analytics:feedback:{feedback_type}:comment_length"
            await redis.rpush(length_key, len(comments))
            await redis.ltrim(length_key, -1000, -1)
        
    except Exception as e:
        logger.error(f"Failed to update feedback metrics: {e}")


async def _analyze_feedback_sentiment(comments: str, user_id: str):
    """Analyze feedback sentiment."""
    try:
        # Simple sentiment analysis (could be enhanced with ML)
        positive_words = ['good', 'great', 'excellent', 'awesome', 'love', 'thanks']
        negative_words = ['bad', 'poor', 'terrible', 'hate', 'disappointed']
        
        comment_lower = comments.lower()
        positive_count = sum(1 for word in positive_words if word in comment_lower)
        negative_count = sum(1 for word in negative_words if word in comment_lower)
        
        sentiment = "neutral"
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        
        # Store sentiment analysis
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        sentiment_key = f"analytics:feedback:sentiment:{sentiment}"
        await redis.incr(sentiment_key)
        
        logger.debug(f"Analyzed sentiment for feedback from user {user_id}: {sentiment}")
        
    except Exception as e:
        logger.error(f"Failed to analyze feedback sentiment: {e}")


async def _update_system_health_metrics(component: str, status: str, response_time: float):
    """Update system health metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Track component status
        status_key = f"analytics:system_health:{component}:status:{status}"
        await redis.incr(status_key)
        
        # Track response times
        if response_time is not None:
            time_key = f"analytics:system_health:{component}:response_times"
            await redis.rpush(time_key, response_time)
            await redis.ltrim(time_key, -1000, -1)
        
    except Exception as e:
        logger.error(f"Failed to update system health metrics: {e}")


async def _alert_system_health_issue(event_data: Dict[str, Any]):
    """Alert on system health issues."""
    try:
        logger.error(
            f"System health issue detected: {event_data.get('component')} - "
            f"status: {event_data.get('status')}, "
            f"response time: {event_data.get('response_time')}"
        )
        
        # Could send to alerting system
        # await alerting_service.send_alert(...)
        
    except Exception as e:
        logger.error(f"Failed to alert on system health issue: {e}")


async def _update_pipeline_metrics(pipeline_name: str, status: str, duration: float, records: int):
    """Update pipeline metrics."""
    try:
        from app.cache.redis_client import get_redis_client
        redis = await get_redis_client()
        
        # Track pipeline executions
        execution_key = f"analytics:pipeline:{pipeline_name}:executions"
        await redis.incr(execution_key)
        
        # Track status
        status_key = f"analytics:pipeline:{pipeline_name}:status:{status}"
        await redis.incr(status_key)
        
        # Track duration
        if duration is not None:
            duration_key = f"analytics:pipeline:{pipeline_name}:durations"
            await redis.rpush(duration_key, duration)
            await redis.ltrim(duration_key, -1000, -1)
        
        # Track records processed
        if records is not None:
            records_key = f"analytics:pipeline:{pipeline_name}:records"
            await redis.incrby(records_key, records)
        
    except Exception as e:
        logger.error(f"Failed to update pipeline metrics: {e}")


class AnalyticsSubscriber:
    """
    Main analytics subscriber class for managing analytics event handlers.
    
    This class provides methods for querying analytics data and generating reports.
    """
    
    def __init__(self):
        self._event_bus = None
    
    async def get_user_activity_report(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Generate user activity report."""
        try:
            async with get_db() as db:
                # Query user activities
                activities = await db.execute(
                    select(UserActivity)
                    .where(UserActivity.user_id == user_id)
                    .where(UserActivity.timestamp >= datetime.utcnow() - timedelta(days=days))
                    .order_by(UserActivity.timestamp.desc())
                )
                activities = activities.scalars().all()
                
                # Calculate metrics
                total_activities = len(activities)
                activity_types = defaultdict(int)
                for activity in activities:
                    activity_types[activity.activity_type] += 1
                
                return {
                    "user_id": user_id,
                    "period_days": days,
                    "total_activities": total_activities,
                    "activity_types": dict(activity_types),
                    "activities": [
                        {
                            "type": a.activity_type,
                            "timestamp": a.timestamp.isoformat(),
                            "data": a.activity_data
                        }
                        for a in activities[:100]  # Limit to 100 most recent
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to generate user activity report: {e}")
            raise
    
    async def get_system_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get system metrics for the last N hours."""
        try:
            from app.cache.redis_client import get_redis_client
            redis = await get_redis_client()
            
            # Get real-time metrics from Redis
            metrics = {}
            pattern = "analytics:realtime:*"
            keys = await redis.keys(pattern)
            
            for key in keys:
                metric_name = key.decode().split(":")[-1]
                value = await redis.get(key)
                metrics[metric_name] = int(value) if value else 0
            
            # Get DAU/WAU/MAU
            date_str = datetime.utcnow().date().isoformat()
            dau_key = f"analytics:dau:{date_str}"
            dau = await redis.scard(dau_key)
            
            # Calculate WAU (last 7 days)
            wau_dates = [(datetime.utcnow() - timedelta(days=i)).date().isoformat() for i in range(7)]
            wau_users = set()
            for day in wau_dates:
                wau_key = f"analytics:wau:{day}"
                day_users = await redis.smembers(wau_key)
                wau_users.update([uid.decode() for uid in day_users])
            wau = len(wau_users)
            
            # Calculate MAU (last 30 days)
            mau_dates = [(datetime.utcnow() - timedelta(days=i)).date().isoformat() for i in range(30)]
            mau_users = set()
            for day in mau_dates:
                mau_key = f"analytics:mau:{day}"
                day_users = await redis.smembers(mau_key)
                mau_users.update([uid.decode() for uid in day_users])
            mau = len(mau_users)
            
            metrics.update({
                "dau": dau,
                "wau": wau,
                "mau": mau,
                "retention_rate": round(dau / mau * 100, 2) if mau > 0 else 0
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            raise
    
    async def generate_daily_report(self, date: datetime.date) -> Dict[str, Any]:
        """Generate daily analytics report."""
        try:
            report = {
                "date": date.isoformat(),
                "generated_at": datetime.utcnow().isoformat(),
                "metrics": {}
            }
            
            # Get various metrics
            report["metrics"]["user_registrations"] = await self._get_daily_count(
                "registrations", date
            )
            report["metrics"]["incidents_reported"] = await self._get_daily_count(
                "incidents_reported", date
            )
            report["metrics"]["briefings_generated"] = await self._get_daily_count(
                "briefings_generated", date
            )
            report["metrics"]["chat_messages"] = await self._get_daily_count(
                "user_messages", date
            ) + await self._get_daily_count("ai_responses", date)
            
            # Get top content
            report["top_content"] = await self._get_top_content(date)
            
            # Get geographic distribution
            report["geographic_distribution"] = await self._get_geographic_distribution(date)
            
            # Get device usage
            report["device_usage"] = await self._get_device_usage(date)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            raise
    
    async def _get_daily_count(self, metric: str, date: datetime.date) -> int:
        """Get daily count for a metric."""
        try:
            from app.cache.redis_client import get_redis_client
            redis = await get_redis_client()
            
            key = f"analytics:daily:{metric}:{date.isoformat()}"
            value = await redis.get(key)
            return int(value) if value else 0
            
        except Exception as e:
            logger.error(f"Failed to get daily count for {metric}: {e}")
            return 0
    
    async def _get_top_content(self, date: datetime.date) -> List[Dict[str, Any]]:
        """Get top content for a date."""
        try:
            from app.cache.redis_client import get_redis_client
            redis = await get_redis_client()
            
            # This is a simplified implementation
            # In production, you'd query the database
            return []
            
        except Exception as e:
            logger.error(f"Failed to get top content: {e}")
            return []
    
    async def _get_geographic_distribution(self, date: datetime.date) -> Dict[str, Any]:
        """Get geographic distribution for a date."""
        try:
            # Simplified implementation
            return {"total_locations": 0, "top_countries": []}
            
        except Exception as e:
            logger.error(f"Failed to get geographic distribution: {e}")
            return {}
    
    async def _get_device_usage(self, date: datetime.date) -> Dict[str, Any]:
        """Get device usage for a date."""
        try:
            # Simplified implementation
            return {"devices": [], "browsers": [], "os": []}
            
        except Exception as e:
            logger.error(f"Failed to get device usage: {e}")
            return {}