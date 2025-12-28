"""
Notification Template model for managing reusable notification templates.
Supports variable substitution, multi-channel templates, and versioning.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Enum, ForeignKey, Boolean, JSON, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, validates
from datetime import datetime
import enum
import json
import re
from typing import Any, Dict, List, Optional

from app.db.base import Base
from app.core.exceptions import ValidationError


class TemplateType(str, enum.Enum):
    """
    Types of notification templates.
    """
    SYSTEM = "system"              # System notifications
    TRANSACTIONAL = "transactional" # Transactional emails/notifications
    MARKETING = "marketing"        # Marketing/promotional content
    ALERT = "alert"                # Alert/emergency notifications
    AUTOMATED = "automated"        # Automated system notifications
    CUSTOM = "custom"              # Custom user-defined templates


class TemplateLanguage(str, enum.Enum):
    """
    Supported languages for templates.
    """
    EN = "en"      # English
    ES = "es"      # Spanish
    FR = "fr"      # French
    DE = "de"      # German
    ZH = "zh"      # Chinese
    JA = "ja"      # Japanese
    KO = "ko"      # Korean
    RU = "ru"      # Russian
    AR = "ar"      # Arabic
    PT = "pt"      # Portuguese
    HI = "hi"      # Hindi
    BN = "bn"      # Bengali
    ALL = "all"    # All languages (wildcard)


class TemplateStatus(str, enum.Enum):
    """
    Status of a template.
    """
    DRAFT = "draft"                # Draft, not ready for use
    ACTIVE = "active"              # Active and available for use
    INACTIVE = "inactive"          # Inactive, not used
    ARCHIVED = "archived"          # Archived, kept for reference
    DEPRECATED = "deprecated"      # Deprecated, replaced by new version
    TESTING = "testing"            # In testing/QA phase


class VariableType(str, enum.Enum):
    """
    Types of template variables.
    """
    STRING = "string"              # Text string
    NUMBER = "number"              # Numeric value
    BOOLEAN = "boolean"            # True/False
    DATE = "date"                  # Date value
    DATETIME = "datetime"          # Date and time
    CURRENCY = "currency"          # Currency amount
    URL = "url"                    # URL/link
    EMAIL = "email"                # Email address
    PHONE = "phone"                # Phone number
    USER = "user"                  # User object
    CONTENT = "content"            # Content object
    LIST = "list"                  # List/array
    OBJECT = "object"              # JSON object
    HTML = "html"                  # HTML content
    MARKDOWN = "markdown"          # Markdown content


class NotificationTemplate(Base):
    """
    Model for storing notification templates.
    
    Templates support variable substitution, multi-channel content,
    and versioning for easy updates and rollbacks.
    """
    
    __tablename__ = "notification_templates"
    
    # Primary key and identification
    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(String(100), unique=True, index=True, nullable=False, 
                        comment="Unique identifier for the template (e.g., 'welcome_email')")
    name = Column(String(255), nullable=False, comment="Human-readable template name")
    description = Column(Text, nullable=True, comment="Template description")
    
    # Template configuration
    template_type = Column(Enum(TemplateType), default=TemplateType.SYSTEM, nullable=False, index=True)
    language = Column(Enum(TemplateLanguage), default=TemplateLanguage.EN, nullable=False, index=True)
    status = Column(Enum(TemplateStatus), default=TemplateStatus.DRAFT, nullable=False, index=True)
    
    # Versioning
    version = Column(Integer, default=1, nullable=False, comment="Template version")
    version_notes = Column(Text, nullable=True, comment="Notes about this version")
    previous_version_id = Column(Integer, ForeignKey("notification_templates.id"), nullable=True)
    is_latest = Column(Boolean, default=True, index=True, comment="Whether this is the latest version")
    
    # Template content per channel
    subject_template = Column(String(500), nullable=True, comment="Subject line template (for email)")
    title_template = Column(String(500), nullable=True, comment="Title template (for in-app/push)")
    message_template = Column(Text, nullable=False, comment="Main message template")
    summary_template = Column(String(1000), nullable=True, comment="Summary/snippet template")
    
    # Rich content templates
    html_template = Column(Text, nullable=True, comment="HTML template (for email)")
    markdown_template = Column(Text, nullable=True, comment="Markdown template")
    plain_text_template = Column(Text, nullable=True, comment="Plain text template")
    
    # Push notification specific
    push_title_template = Column(String(200), nullable=True, comment="Title template for push notifications")
    push_body_template = Column(Text, nullable=True, comment="Body template for push notifications")
    push_sound = Column(String(100), nullable=True, default="default", comment="Sound for push notifications")
    push_badge_template = Column(String(100), nullable=True, comment="Badge count template")
    
    # SMS specific
    sms_template = Column(String(500), nullable=True, comment="SMS template (max 500 chars)")
    
    # Webhook specific
    webhook_payload_template = Column(JSONB, nullable=True, comment="JSON template for webhook payload")
    
    # Actions and links
    action_url_template = Column(String(500), nullable=True, comment="Action URL template")
    action_text_template = Column(String(100), nullable=True, comment="Action text template")
    secondary_action_url_template = Column(String(500), nullable=True, comment="Secondary action URL template")
    secondary_action_text_template = Column(String(100), nullable=True, comment="Secondary action text template")
    unsubscribe_url_template = Column(String(500), nullable=True, comment="Unsubscribe URL template")
    
    # Default values (used when variables are not provided)
    default_subject = Column(String(500), nullable=True, comment="Default subject")
    default_title = Column(String(500), nullable=True, comment="Default title")
    default_message = Column(Text, nullable=True, comment="Default message")
    default_data = Column(JSONB, nullable=True, comment="Default data values")
    
    # Variable definitions
    variables = Column(JSONB, nullable=True, comment="JSON array of variable definitions")
    required_variables = Column(JSONB, nullable=True, comment="JSON array of required variable names")
    variable_defaults = Column(JSONB, nullable=True, comment="JSON map of variable -> default value")
    
    # Template metadata
    tags = Column(JSONB, nullable=True, comment="JSON array of tags for categorization")
    categories = Column(JSONB, nullable=True, comment="JSON array of categories")
    applicable_channels = Column(JSONB, nullable=True, comment="JSON array of channels this template applies to")
    supported_locales = Column(JSONB, nullable=True, comment="JSON array of supported locales")
    
    # Preview and testing
    preview_data = Column(JSONB, nullable=True, comment="Sample data for preview/testing")
    preview_image_url = Column(String(500), nullable=True, comment="Preview image URL")
    
    # Restrictions
    min_user_role = Column(String(50), nullable=True, comment="Minimum user role required")
    allowed_user_groups = Column(JSONB, nullable=True, comment="JSON array of allowed user group IDs")
    disallowed_user_groups = Column(JSONB, nullable=True, comment="JSON array of disallowed user group IDs")
    
    # Scheduling
    valid_from = Column(DateTime, nullable=True, index=True, comment="Template valid from date")
    valid_until = Column(DateTime, nullable=True, index=True, comment="Template valid until date")
    
    # Rate limiting
    rate_limit_per_user = Column(Integer, nullable=True, comment="Max sends per user per hour")
    rate_limit_global = Column(Integer, nullable=True, comment="Max global sends per hour")
    cooldown_period = Column(Integer, nullable=True, comment="Cooldown period in seconds")
    
    # Statistics
    usage_count = Column(Integer, default=0, comment="Number of times template used")
    last_used_at = Column(DateTime, nullable=True, comment="Last time template was used")
    success_rate = Column(Integer, nullable=True, comment="Success rate percentage (0-100)")
    
    # Audit
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    approved_at = Column(DateTime, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    # System flags
    is_system_template = Column(Boolean, default=False, index=True, comment="Whether template is system-defined")
    is_protected = Column(Boolean, default=False, comment="Whether template is protected from deletion")
    requires_approval = Column(Boolean, default=False, comment="Whether template requires approval")
    is_verified = Column(Boolean, default=False, comment="Whether template has been verified")
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])
    updater = relationship("User", foreign_keys=[updated_by])
    approver = relationship("User", foreign_keys=[approved_by])
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    previous_version_rel = relationship(
        "NotificationTemplate",
        remote_side=[id],
        primaryjoin="NotificationTemplate.previous_version_id == foreign(NotificationTemplate.id)",
        backref="next_version"
    )
    
    # Indexes
    __table_args__ = (
        Index('ix_notification_templates_type_status', 'template_type', 'status'),
        Index('ix_notification_templates_id_version', 'template_id', 'version', unique=True),
        Index('ix_notification_templates_created_by', 'created_by', 'created_at'),
        Index('ix_notification_templates_validity', 'valid_from', 'valid_until'),
        Index('ix_notification_templates_tags', 'tags', postgresql_using='gin'),
        Index('ix_notification_templates_categories', 'categories', postgresql_using='gin'),
    )
    
    @validates('template_id')
    def validate_template_id(self, key, template_id):
        """Validate template ID format."""
        if not template_id or len(template_id.strip()) == 0:
            raise ValidationError("Template ID cannot be empty")
        
        # Check format (alphanumeric, underscores, hyphens)
        if not re.match(r'^[a-zA-Z0-9_-]+$', template_id):
            raise ValidationError("Template ID can only contain letters, numbers, underscores, and hyphens")
        
        return template_id.strip()
    
    @validates('name')
    def validate_name(self, key, name):
        """Validate template name."""
        if not name or len(name.strip()) == 0:
            raise ValidationError("Template name cannot be empty")
        
        if len(name.strip()) > 255:
            raise ValidationError("Template name cannot exceed 255 characters")
        
        return name.strip()
    
    @validates('version')
    def validate_version(self, key, version):
        """Validate version number."""
        if version < 1:
            raise ValidationError("Version must be at least 1")
        return version
    
    @validates('rate_limit_per_user', 'rate_limit_global')
    def validate_rate_limit(self, key, limit):
        """Validate rate limit values."""
        if limit is not None and limit < 0:
            raise ValidationError("Rate limit cannot be negative")
        return limit
    
    def __repr__(self):
        return f"<NotificationTemplate {self.template_id} v{self.version} ({self.status})>"
    
    def to_dict(self, include_content: bool = True) -> Dict[str, Any]:
        """
        Convert template to dictionary representation.
        
        Args:
            include_content: Whether to include template content
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": self.id,
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "template_type": self.template_type.value,
            "language": self.language.value,
            "status": self.status.value,
            "version": self.version,
            "version_notes": self.version_notes,
            "is_latest": self.is_latest,
            "tags": self.tags,
            "categories": self.categories,
            "applicable_channels": self.applicable_channels,
            "supported_locales": self.supported_locales,
            "variables": self.variables,
            "required_variables": self.required_variables,
            "variable_defaults": self.variable_defaults,
            "min_user_role": self.min_user_role,
            "allowed_user_groups": self.allowed_user_groups,
            "disallowed_user_groups": self.disallowed_user_groups,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "rate_limit_per_user": self.rate_limit_per_user,
            "rate_limit_global": self.rate_limit_global,
            "cooldown_period": self.cooldown_period,
            "usage_count": self.usage_count,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "success_rate": self.success_rate,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "approved_by": self.approved_by,
            "reviewed_by": self.reviewed_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "is_system_template": self.is_system_template,
            "is_protected": self.is_protected,
            "requires_approval": self.requires_approval,
            "is_verified": self.is_verified,
            "preview_image_url": self.preview_image_url,
        }
        
        if include_content:
            result.update({
                "subject_template": self.subject_template,
                "title_template": self.title_template,
                "message_template": self.message_template,
                "summary_template": self.summary_template,
                "html_template": self.html_template,
                "markdown_template": self.markdown_template,
                "plain_text_template": self.plain_text_template,
                "push_title_template": self.push_title_template,
                "push_body_template": self.push_body_template,
                "push_sound": self.push_sound,
                "push_badge_template": self.push_badge_template,
                "sms_template": self.sms_template,
                "webhook_payload_template": self.webhook_payload_template,
                "action_url_template": self.action_url_template,
                "action_text_template": self.action_text_template,
                "secondary_action_url_template": self.secondary_action_url_template,
                "secondary_action_text_template": self.secondary_action_text_template,
                "unsubscribe_url_template": self.unsubscribe_url_template,
                "default_subject": self.default_subject,
                "default_title": self.default_title,
                "default_message": self.default_message,
                "default_data": self.default_data,
                "preview_data": self.preview_data,
            })
        
        return result
    
    @property
    def is_active(self) -> bool:
        """
        Check if template is currently active.
        
        Returns:
            True if template is active
        """
        if self.status != TemplateStatus.ACTIVE:
            return False
        
        current_time = datetime.utcnow()
        
        # Check validity period
        if self.valid_from and current_time < self.valid_from:
            return False
        
        if self.valid_until and current_time > self.valid_until:
            return False
        
        return True
    
    @property
    def has_expired(self) -> bool:
        """
        Check if template has expired.
        
        Returns:
            True if template has expired
        """
        if self.valid_until:
            return datetime.utcnow() > self.valid_until
        return False
    
    @property
    def is_available(self) -> bool:
        """
        Check if template is available for use.
        
        Returns:
            True if template can be used
        """
        return self.is_active and self.is_latest and not self.has_expired
    
    def get_variable_definitions(self) -> List[Dict[str, Any]]:
        """
        Get variable definitions as a list of dictionaries.
        
        Returns:
            List of variable definitions
        """
        if not self.variables:
            return []
        
        return self.variables
    
    def get_required_variables(self) -> List[str]:
        """
        Get list of required variable names.
        
        Returns:
            List of required variable names
        """
        if not self.required_variables:
            return []
        
        return self.required_variables
    
    def validate_variables(self, provided_variables: Dict[str, Any]) -> List[str]:
        """
        Validate provided variables against template requirements.
        
        Args:
            provided_variables: Dictionary of provided variables
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check required variables
        required_vars = self.get_required_variables()
        for var_name in required_vars:
            if var_name not in provided_variables or provided_variables[var_name] is None:
                errors.append(f"Required variable '{var_name}' is missing")
        
        # Check variable types (if variable definitions exist)
        variable_defs = self.get_variable_definitions()
        if variable_defs:
            for var_def in variable_defs:
                var_name = var_def.get('name')
                var_type = var_def.get('type')
                
                if var_name in provided_variables:
                    value = provided_variables[var_name]
                    
                    # Skip validation for None values (they'll use defaults)
                    if value is None:
                        continue
                    
                    # Type validation
                    try:
                        self._validate_variable_type(var_name, value, var_type)
                    except ValidationError as e:
                        errors.append(str(e))
        
        return errors
    
    def _validate_variable_type(self, var_name: str, value: Any, var_type: str) -> None:
        """
        Validate variable value against type.
        
        Args:
            var_name: Variable name
            value: Variable value
            var_type: Expected type
            
        Raises:
            ValidationError: If type validation fails
        """
        type_validators = {
            VariableType.STRING: lambda v: isinstance(v, str),
            VariableType.NUMBER: lambda v: isinstance(v, (int, float)),
            VariableType.BOOLEAN: lambda v: isinstance(v, bool),
            VariableType.DATE: lambda v: isinstance(v, datetime) or isinstance(v, str),
            VariableType.DATETIME: lambda v: isinstance(v, datetime) or isinstance(v, str),
            VariableType.CURRENCY: lambda v: isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()),
            VariableType.URL: lambda v: isinstance(v, str) and (v.startswith('http://') or v.startswith('https://')),
            VariableType.EMAIL: lambda v: isinstance(v, str) and '@' in v,
            VariableType.PHONE: lambda v: isinstance(v, str) and v.replace('+', '').replace(' ', '').replace('-', '').isdigit(),
            VariableType.USER: lambda v: isinstance(v, dict) and 'id' in v,
            VariableType.CONTENT: lambda v: isinstance(v, dict) and 'id' in v,
            VariableType.LIST: lambda v: isinstance(v, list),
            VariableType.OBJECT: lambda v: isinstance(v, dict),
            VariableType.HTML: lambda v: isinstance(v, str),
            VariableType.MARKDOWN: lambda v: isinstance(v, str),
        }
        
        if var_type not in type_validators:
            return  # Unknown type, skip validation
        
        if not type_validators[var_type](value):
            raise ValidationError(f"Variable '{var_name}' should be of type '{var_type}', got {type(value).__name__}")
    
    def render(self, variables: Dict[str, Any], channel: str = None) -> Dict[str, str]:
        """
        Render template with variables for specified channel.
        
        Args:
            variables: Dictionary of template variables
            channel: Optional specific channel to render for
            
        Returns:
            Dictionary of rendered content for different fields
        """
        # Validate variables first
        errors = self.validate_variables(variables)
        if errors:
            raise ValidationError(f"Template variable validation failed: {', '.join(errors)}")
        
        # Merge with defaults
        all_variables = self.variable_defaults or {}
        all_variables.update(variables)
        
        # Render templates
        rendered = {}
        
        # Helper function to render a template string
        def render_template(template: str) -> str:
            if not template:
                return ""
            
            result = template
            for key, value in all_variables.items():
                placeholder = f"{{{key}}}"
                if placeholder in result:
                    result = result.replace(placeholder, str(value))
            
            return result
        
        # Render based on channel or all channels
        if channel == 'email' or channel is None:
            if self.subject_template:
                rendered['subject'] = render_template(self.subject_template)
            if self.html_template:
                rendered['html_content'] = render_template(self.html_template)
            if self.plain_text_template:
                rendered['plain_text'] = render_template(self.plain_text_template)
        
        if channel == 'push' or channel is None:
            if self.push_title_template:
                rendered['push_title'] = render_template(self.push_title_template)
            if self.push_body_template:
                rendered['push_body'] = render_template(self.push_body_template)
            if self.push_badge_template:
                rendered['push_badge'] = render_template(self.push_badge_template)
        
        if channel == 'sms' or channel is None:
            if self.sms_template:
                rendered['sms'] = render_template(self.sms_template)
        
        if channel == 'webhook' or channel is None:
            if self.webhook_payload_template:
                rendered['webhook_payload'] = self._render_json_template(
                    self.webhook_payload_template, all_variables
                )
        
        # Always render common fields
        if self.title_template:
            rendered['title'] = render_template(self.title_template)
        if self.message_template:
            rendered['message'] = render_template(self.message_template)
        if self.summary_template:
            rendered['summary'] = render_template(self.summary_template)
        if self.markdown_template:
            rendered['markdown'] = render_template(self.markdown_template)
        if self.action_url_template:
            rendered['action_url'] = render_template(self.action_url_template)
        if self.action_text_template:
            rendered['action_text'] = render_template(self.action_text_template)
        
        # Add default values for any missing required fields
        if 'subject' not in rendered and self.default_subject:
            rendered['subject'] = self.default_subject
        if 'title' not in rendered and self.default_title:
            rendered['title'] = self.default_title
        if 'message' not in rendered and self.default_message:
            rendered['message'] = self.default_message
        
        return rendered
    
    def _render_json_template(self, json_template: Dict, variables: Dict[str, Any]) -> Dict:
        """
        Render a JSON template with variable substitution.
        
        Args:
            json_template: JSON template structure
            variables: Template variables
            
        Returns:
            Rendered JSON structure
        """
        import json as json_module
        
        # Convert to string, replace variables, then parse back
        json_str = json_module.dumps(json_template)
        
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in json_str:
                # Handle different types of values
                if isinstance(value, (dict, list)):
                    value_str = json_module.dumps(value)
                else:
                    value_str = str(value)
                json_str = json_str.replace(f'"{placeholder}"', value_str)
                json_str = json_str.replace(placeholder, value_str)
        
        return json_module.loads(json_str)
    
    def increment_usage(self):
        """Increment usage count and update last used timestamp."""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()
    
    def create_new_version(self, new_data: Dict[str, Any], updated_by: int = None) -> 'NotificationTemplate':
        """
        Create a new version of this template.
        
        Args:
            new_data: Dictionary of new template data
            updated_by: ID of user creating the new version
            
        Returns:
            New template version
        """
        # Mark current version as not latest
        self.is_latest = False
        
        # Create new version
        new_version = NotificationTemplate(
            template_id=self.template_id,
            name=new_data.get('name', self.name),
            description=new_data.get('description', self.description),
            template_type=new_data.get('template_type', self.template_type),
            language=new_data.get('language', self.language),
            status=TemplateStatus.DRAFT,
            version=self.version + 1,
            version_notes=new_data.get('version_notes'),
            previous_version_id=self.id,
            is_latest=True,
            
            # Copy or update content
            subject_template=new_data.get('subject_template', self.subject_template),
            title_template=new_data.get('title_template', self.title_template),
            message_template=new_data.get('message_template', self.message_template),
            summary_template=new_data.get('summary_template', self.summary_template),
            html_template=new_data.get('html_template', self.html_template),
            markdown_template=new_data.get('markdown_template', self.markdown_template),
            plain_text_template=new_data.get('plain_text_template', self.plain_text_template),
            push_title_template=new_data.get('push_title_template', self.push_title_template),
            push_body_template=new_data.get('push_body_template', self.push_body_template),
            push_sound=new_data.get('push_sound', self.push_sound),
            push_badge_template=new_data.get('push_badge_template', self.push_badge_template),
            sms_template=new_data.get('sms_template', self.sms_template),
            webhook_payload_template=new_data.get('webhook_payload_template', self.webhook_payload_template),
            action_url_template=new_data.get('action_url_template', self.action_url_template),
            action_text_template=new_data.get('action_text_template', self.action_text_template),
            secondary_action_url_template=new_data.get('secondary_action_url_template', self.secondary_action_url_template),
            secondary_action_text_template=new_data.get('secondary_action_text_template', self.secondary_action_text_template),
            unsubscribe_url_template=new_data.get('unsubscribe_url_template', self.unsubscribe_url_template),
            
            default_subject=new_data.get('default_subject', self.default_subject),
            default_title=new_data.get('default_title', self.default_title),
            default_message=new_data.get('default_message', self.default_message),
            default_data=new_data.get('default_data', self.default_data),
            
            variables=new_data.get('variables', self.variables),
            required_variables=new_data.get('required_variables', self.required_variables),
            variable_defaults=new_data.get('variable_defaults', self.variable_defaults),
            
            tags=new_data.get('tags', self.tags),
            categories=new_data.get('categories', self.categories),
            applicable_channels=new_data.get('applicable_channels', self.applicable_channels),
            supported_locales=new_data.get('supported_locales', self.supported_locales),
            
            preview_data=new_data.get('preview_data', self.preview_data),
            preview_image_url=new_data.get('preview_image_url', self.preview_image_url),
            
            min_user_role=new_data.get('min_user_role', self.min_user_role),
            allowed_user_groups=new_data.get('allowed_user_groups', self.allowed_user_groups),
            disallowed_user_groups=new_data.get('disallowed_user_groups', self.disallowed_user_groups),
            
            valid_from=new_data.get('valid_from', self.valid_from),
            valid_until=new_data.get('valid_until', self.valid_until),
            
            rate_limit_per_user=new_data.get('rate_limit_per_user', self.rate_limit_per_user),
            rate_limit_global=new_data.get('rate_limit_global', self.rate_limit_global),
            cooldown_period=new_data.get('cooldown_period', self.cooldown_period),
            
            created_by=self.created_by,
            updated_by=updated_by,
            
            is_system_template=self.is_system_template,
            is_protected=self.is_protected,
            requires_approval=self.requires_approval,
        )
        
        return new_version
    
    def activate(self, approved_by: int = None):
        """Activate this template version."""
        self.status = TemplateStatus.ACTIVE
        if approved_by:
            self.approved_by = approved_by
            self.approved_at = datetime.utcnow()
    
    def deactivate(self):
        """Deactivate this template version."""
        self.status = TemplateStatus.INACTIVE
    
    def archive(self):
        """Archive this template version."""
        self.status = TemplateStatus.ARCHIVED


class TemplateTranslation(Base):
    """
    Model for storing translations of notification templates.
    
    Each translation is linked to a base template and contains
    translated content for a specific language.
    """
    
    __tablename__ = "template_translations"
    
    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(Integer, ForeignKey("notification_templates.id", ondelete="CASCADE"), nullable=False, index=True)
    language = Column(Enum(TemplateLanguage), nullable=False, index=True)
    locale = Column(String(10), nullable=False, index=True, comment="Locale code (e.g., 'en_US', 'fr_CA')")
    
    # Translated content
    translated_subject = Column(String(500), nullable=True)
    translated_title = Column(String(500), nullable=True)
    translated_message = Column(Text, nullable=False)
    translated_summary = Column(String(1000), nullable=True)
    translated_html = Column(Text, nullable=True)
    translated_markdown = Column(Text, nullable=True)
    translated_plain_text = Column(Text, nullable=True)
    
    # Translated push content
    translated_push_title = Column(String(200), nullable=True)
    translated_push_body = Column(Text, nullable=True)
    
    # Translated SMS
    translated_sms = Column(String(500), nullable=True)
    
    # Translation metadata
    translator_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    translation_status = Column(String(50), default="draft", comment="draft, in_review, approved, published")
    translation_notes = Column(Text, nullable=True)
    
    # Quality metrics
    translation_quality = Column(Integer, nullable=True, comment="Quality score 0-100")
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    template = relationship("NotificationTemplate", backref="translations")
    translator = relationship("User", foreign_keys=[translator_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    
    # Unique constraint
    __table_args__ = (
        Index('ix_template_translations_unique', 'template_id', 'locale', unique=True),
    )
    
    def __repr__(self):
        return f"<TemplateTranslation template={self.template_id} locale={self.locale}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert translation to dictionary."""
        return {
            "id": self.id,
            "template_id": self.template_id,
            "language": self.language.value,
            "locale": self.locale,
            "translated_subject": self.translated_subject,
            "translated_title": self.translated_title,
            "translated_message": self.translated_message,
            "translated_summary": self.translated_summary,
            "translated_html": self.translated_html,
            "translated_markdown": self.translated_markdown,
            "translated_plain_text": self.translated_plain_text,
            "translated_push_title": self.translated_push_title,
            "translated_push_body": self.translated_push_body,
            "translated_sms": self.translated_sms,
            "translator_id": self.translator_id,
            "translation_status": self.translation_status,
            "translation_notes": self.translation_notes,
            "translation_quality": self.translation_quality,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TemplateVariable(Base):
    """
    Model for managing reusable template variables.
    
    Allows defining variables once and reusing them across multiple templates.
    """
    
    __tablename__ = "template_variables"
    
    id = Column(Integer, primary_key=True, index=True)
    variable_name = Column(String(100), unique=True, index=True, nullable=False, 
                          comment="Unique variable name (e.g., 'user_name', 'order_total')")
    display_name = Column(String(200), nullable=False, comment="Human-readable display name")
    description = Column(Text, nullable=True, comment="Variable description")
    
    # Variable definition
    variable_type = Column(Enum(VariableType), default=VariableType.STRING, nullable=False)
    default_value = Column(JSONB, nullable=True, comment="Default value in JSON format")
    validation_rules = Column(JSONB, nullable=True, comment="JSON validation rules")
    options = Column(JSONB, nullable=True, comment="JSON array of allowed values (for enums)")
    
    # Usage tracking
    used_in_templates = Column(JSONB, nullable=True, comment="JSON array of template IDs using this variable")
    usage_count = Column(Integer, default=0, comment="Number of times variable used")
    
    # System flags
    is_system_variable = Column(Boolean, default=False, index=True, comment="Whether variable is system-defined")
    is_required = Column(Boolean, default=False, comment="Whether variable is generally required")
    is_sensitive = Column(Boolean, default=False, comment="Whether variable contains sensitive data")
    
    # Metadata
    category = Column(String(100), nullable=True, index=True, comment="Variable category")
    tags = Column(JSONB, nullable=True, comment="JSON array of tags")
    
    # Audit
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])
    
    def __repr__(self):
        return f"<TemplateVariable {self.variable_name} ({self.variable_type})>"
    
    def validate_value(self, value: Any) -> List[str]:
        """
        Validate a value against this variable's rules.
        
        Args:
            value: Value to validate
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Type validation
        try:
            self._validate_type(value)
        except ValidationError as e:
            errors.append(str(e))
        
        # Custom validation rules
        if self.validation_rules:
            errors.extend(self._apply_validation_rules(value))
        
        # Check against allowed options
        if self.options and value not in self.options:
            errors.append(f"Value must be one of: {', '.join(str(opt) for opt in self.options)}")
        
        return errors
    
    def _validate_type(self, value: Any) -> None:
        """Validate value type."""
        type_map = {
            VariableType.STRING: str,
            VariableType.NUMBER: (int, float),
            VariableType.BOOLEAN: bool,
            VariableType.DATE: (str, datetime),
            VariableType.DATETIME: (str, datetime),
            VariableType.CURRENCY: (int, float, str),
            VariableType.URL: str,
            VariableType.EMAIL: str,
            VariableType.PHONE: str,
            VariableType.USER: dict,
            VariableType.CONTENT: dict,
            VariableType.LIST: list,
            VariableType.OBJECT: dict,
            VariableType.HTML: str,
            VariableType.MARKDOWN: str,
        }
        
        expected_type = type_map.get(self.variable_type)
        if not expected_type:
            return  # Unknown type, skip validation
        
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"Variable '{self.variable_name}' should be of type '{self.variable_type}', "
                f"got {type(value).__name__}"
            )
    
    def _apply_validation_rules(self, value: Any) -> List[str]:
        """Apply custom validation rules."""
        errors = []
        rules = self.validation_rules or {}
        
        # String length rules
        if isinstance(value, str):
            if 'min_length' in rules and len(value) < rules['min_length']:
                errors.append(f"Minimum length is {rules['min_length']} characters")
            if 'max_length' in rules and len(value) > rules['max_length']:
                errors.append(f"Maximum length is {rules['max_length']} characters")
            if 'pattern' in rules and not re.match(rules['pattern'], value):
                errors.append(f"Value must match pattern: {rules['pattern']}")
        
        # Number range rules
        if isinstance(value, (int, float)):
            if 'min_value' in rules and value < rules['min_value']:
                errors.append(f"Minimum value is {rules['min_value']}")
            if 'max_value' in rules and value > rules['max_value']:
                errors.append(f"Maximum value is {rules['max_value']}")
        
        # Array rules
        if isinstance(value, list):
            if 'min_items' in rules and len(value) < rules['min_items']:
                errors.append(f"Minimum {rules['min_items']} items required")
            if 'max_items' in rules and len(value) > rules['max_items']:
                errors.append(f"Maximum {rules['max_items']} items allowed")
        
        return errors