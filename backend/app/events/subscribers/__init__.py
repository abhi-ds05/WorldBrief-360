"""
Event subscribers for WorldBrief 360.

This module contains all event subscribers that process specific events
and perform actions like notifications, analytics, moderation, etc.
"""

from app.events.subscribers.notification_subscriber import (
    NotificationSubscriber,
    send_welcome_email,
    send_password_reset_email,
    send_incident_verification_notification,
    send_briefing_ready_notification,
    send_wallet_transaction_notification,
    send_system_alert_notification,
    send_daily_digest,
    send_weekly_summary,
    send_topic_alert,
    send_community_update,
    send_chat_notification,
    send_content_moderation_alert,
    send_rate_limit_warning,
    send_backup_completion_notification,
    send_subscription_renewal_reminder,
    send_feature_announcement,
    send_maintenance_notification,
    send_security_alert,
    send_integration_error_alert
)

from app.events.subscribers.analytics_subscriber import (
    AnalyticsSubscriber,
    track_user_registration,
    track_user_login,
    track_incident_report,
    track_briefing_generation,
    track_chat_interaction,
    track_wallet_transaction,
    track_feature_usage,
    track_content_view,
    track_api_performance,
    track_conversion_funnel,
    track_session_duration,
    track_retention_metrics,
    track_geographic_distribution,
    track_device_usage,
    track_referral_sources,
    track_a_b_test_results,
    track_customer_feedback,
    track_system_health,
    track_data_pipeline_metrics
)

from app.events.subscribers.moderation_subscriber import (
    ModerationSubscriber,
    moderate_user_content,
    moderate_incident_reports,
    moderate_chat_messages,
    moderate_community_posts,
    moderate_media_uploads,
    auto_moderate_content,
    flag_suspicious_activity,
    review_flagged_content,
    escalate_moderation_cases,
    update_moderation_rules,
    generate_moderation_reports,
    handle_moderation_appeals,
    train_moderation_models,
    audit_moderation_decisions,
    sync_moderation_with_external_systems,
    send_moderation_warnings,
    apply_content_penalties,
    restore_moderated_content,
    track_moderation_metrics
)

from app.events.subscribers.audit_subscriber import (
    AuditSubscriber,
    audit_user_actions,
    audit_system_changes,
    audit_data_access,
    audit_configuration_changes,
    audit_security_events,
    audit_compliance_checks,
    audit_business_processes,
    audit_third_party_integrations,
    generate_audit_reports,
    archive_audit_logs,
    alert_on_audit_anomalies,
    sync_audit_with_external_systems,
    validate_audit_integrity,
    export_audit_data,
    analyze_audit_patterns,
    cleanup_old_audit_logs,
    monitor_audit_system_health,
    generate_compliance_certificates,
    audit_privacy_requests
)

from app.events.subscribers.background_task_subscriber import (
    BackgroundTaskSubscriber,
    process_background_jobs,
    handle_data_sync,
    process_batch_operations,
    generate_reports,
    clean_up_old_data,
    update_caches,
    send_bulk_notifications,
    process_data_exports,
    handle_file_processing,
    run_scheduled_maintenance,
    update_search_indexes,
    process_webhook_queue,
    handle_payment_reconciliation,
    update_analytics_aggregates,
    process_image_generation,
    handle_video_transcoding,
    process_document_conversion,
    update_recommendation_models,
    sync_with_external_apis,
    backup_database
)

from app.events.subscribers.wallet_subscriber import (
    WalletSubscriber,
    process_coin_rewards,
    handle_transaction_validation,
    update_wallet_balances,
    generate_wallet_statements,
    detect_fraudulent_transactions,
    process_refund_requests,
    handle_coin_transfers,
    award_achievement_badges,
    process_bonus_calculations,
    update_reward_tiers,
    generate_tax_reports,
    handle_payment_gateway_events,
    sync_wallet_with_blockchain,
    process_staking_rewards,
    handle_loyalty_program_updates,
    generate_wallet_analytics,
    notify_low_balance,
    process_subscription_payments,
    handle_currency_conversion
)

from app.events.subscribers.community_subscriber import (
    CommunitySubscriber,
    update_user_reputation,
    award_community_badges,
    update_leaderboards,
    handle_user_follows,
    process_community_votes,
    moderate_community_content,
    generate_community_insights,
    handle_group_membership,
    process_event_rsvps,
    update_discussion_threads,
    handle_poll_creation,
    process_survey_responses,
    generate_community_reports,
    handle_volunteer_signups,
    process_feedback_submissions,
    update_community_settings,
    handle_meetup_organization,
    generate_social_analytics,
    process_collaboration_requests
)

# Export all subscriber classes and functions
__all__ = [
    # Notification Subscriber
    'NotificationSubscriber',
    'send_welcome_email',
    'send_password_reset_email',
    'send_incident_verification_notification',
    'send_briefing_ready_notification',
    'send_wallet_transaction_notification',
    'send_system_alert_notification',
    'send_daily_digest',
    'send_weekly_summary',
    'send_topic_alert',
    'send_community_update',
    'send_chat_notification',
    'send_content_moderation_alert',
    'send_rate_limit_warning',
    'send_backup_completion_notification',
    'send_subscription_renewal_reminder',
    'send_feature_announcement',
    'send_maintenance_notification',
    'send_security_alert',
    'send_integration_error_alert',
    
    # Analytics Subscriber
    'AnalyticsSubscriber',
    'track_user_registration',
    'track_user_login',
    'track_incident_report',
    'track_briefing_generation',
    'track_chat_interaction',
    'track_wallet_transaction',
    'track_feature_usage',
    'track_content_view',
    'track_api_performance',
    'track_conversion_funnel',
    'track_session_duration',
    'track_retention_metrics',
    'track_geographic_distribution',
    'track_device_usage',
    'track_referral_sources',
    'track_a_b_test_results',
    'track_customer_feedback',
    'track_system_health',
    'track_data_pipeline_metrics',
    
    # Moderation Subscriber
    'ModerationSubscriber',
    'moderate_user_content',
    'moderate_incident_reports',
    'moderate_chat_messages',
    'moderate_community_posts',
    'moderate_media_uploads',
    'auto_moderate_content',
    'flag_suspicious_activity',
    'review_flagged_content',
    'escalate_moderation_cases',
    'update_moderation_rules',
    'generate_moderation_reports',
    'handle_moderation_appeals',
    'train_moderation_models',
    'audit_moderation_decisions',
    'sync_moderation_with_external_systems',
    'send_moderation_warnings',
    'apply_content_penalties',
    'restore_moderated_content',
    'track_moderation_metrics',
    
    # Audit Subscriber
    'AuditSubscriber',
    'audit_user_actions',
    'audit_system_changes',
    'audit_data_access',
    'audit_configuration_changes',
    'audit_security_events',
    'audit_compliance_checks',
    'audit_business_processes',
    'audit_third_party_integrations',
    'generate_audit_reports',
    'archive_audit_logs',
    'alert_on_audit_anomalies',
    'sync_audit_with_external_systems',
    'validate_audit_integrity',
    'export_audit_data',
    'analyze_audit_patterns',
    'cleanup_old_audit_logs',
    'monitor_audit_system_health',
    'generate_compliance_certificates',
    'audit_privacy_requests',
    
    # Background Task Subscriber
    'BackgroundTaskSubscriber',
    'process_background_jobs',
    'handle_data_sync',
    'process_batch_operations',
    'generate_reports',
    'clean_up_old_data',
    'update_caches',
    'send_bulk_notifications',
    'process_data_exports',
    'handle_file_processing',
    'run_scheduled_maintenance',
    'update_search_indexes',
    'process_webhook_queue',
    'handle_payment_reconciliation',
    'update_analytics_aggregates',
    'process_image_generation',
    'handle_video_transcoding',
    'process_document_conversion',
    'update_recommendation_models',
    'sync_with_external_apis',
    'backup_database',
    
    # Wallet Subscriber
    'WalletSubscriber',
    'process_coin_rewards',
    'handle_transaction_validation',
    'update_wallet_balances',
    'generate_wallet_statements',
    'detect_fraudulent_transactions',
    'process_refund_requests',
    'handle_coin_transfers',
    'award_achievement_badges',
    'process_bonus_calculations',
    'update_reward_tiers',
    'generate_tax_reports',
    'handle_payment_gateway_events',
    'sync_wallet_with_blockchain',
    'process_staking_rewards',
    'handle_loyalty_program_updates',
    'generate_wallet_analytics',
    'notify_low_balance',
    'process_subscription_payments',
    'handle_currency_conversion',
    
    # Community Subscriber
    'CommunitySubscriber',
    'update_user_reputation',
    'award_community_badges',
    'update_leaderboards',
    'handle_user_follows',
    'process_community_votes',
    'moderate_community_content',
    'generate_community_insights',
    'handle_group_membership',
    'process_event_rsvps',
    'update_discussion_threads',
    'handle_poll_creation',
    'process_survey_responses',
    'generate_community_reports',
    'handle_volunteer_signups',
    'process_feedback_submissions',
    'update_community_settings',
    'handle_meetup_organization',
    'generate_social_analytics',
    'process_collaboration_requests',
]

# Initialize subscribers on module import
def init_subscribers():
    """
    Initialize and register all subscribers with the event bus.
    This should be called during application startup.
    """
    from app.core.logging_config import logger
    from app.events.event_bus import get_event_bus
    from app.events import event_decorators
    
    try:
        # Import subscriber modules to trigger decorator registration
        # The @event_handler decorators auto-register with the event bus
        
        # Log initialization
        event_bus = get_event_bus()
        handler_count = len(event_bus.registry.get_all_handlers_info())
        
        logger.info(f"Event subscribers initialized. Total handlers: {handler_count}")
        
        # Log registered event types
        event_types = list(event_bus.registry._handlers.keys())
        logger.debug(f"Registered event types: {len(event_types)}")
        
        # Log some statistics
        metrics = event_bus.registry.get_handler_metrics()
        logger.debug(f"Subscriber metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Failed to initialize subscribers: {e}")
        raise

# Auto-initialize on module import
try:
    init_subscribers()
except Exception as e:
    # Don't crash if initialization fails - it will be retried on app startup
    from app.core.logging_config import logger
    logger.warning(f"Subscriber auto-initialization failed: {e}")
    logger.warning("Subscribers will be initialized during application startup")