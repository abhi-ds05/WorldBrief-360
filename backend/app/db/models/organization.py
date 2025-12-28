"""
organization.py - Organization and Team Management Model

This module defines models for organization management, teams, roles, and permissions.
This includes:
- Organization creation and management
- Team structures and hierarchies
- Role-based access control (RBAC)
- User invitations and membership
- Organization settings and configuration
- Billing and subscription management
- Audit logs and compliance

Key Features:
- Multi-level organization hierarchy
- Flexible team structures
- Granular permissions system
- User invitation workflows
- Organization-wide settings
- Audit trail and logging
- Subscription and billing integration
- Compliance and security controls
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime,NUMERIC,BigInteger,Numeric,
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint,
    Index, Table, UniqueConstraint, LargeBinary
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func, expression
from sqlalchemy.ext.hybrid import hybrid_property

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from models.user import User
    from models.subscription import Subscription
    from models.incident import Incident
    from models.article import Article
    from models.dataset import Dataset


class OrganizationType(Enum):
    """Types of organizations."""
    COMPANY = "company"              # Business/company
    NON_PROFIT = "non_profit"        # Non-profit organization
    GOVERNMENT = "government"        # Government agency
    EDUCATIONAL = "educational"      # Educational institution
    RESEARCH = "research"            # Research organization
    COMMUNITY = "community"          # Community group
    PERSONAL = "personal"            # Personal organization
    OTHER = "other"                  # Other organization type


class OrganizationStatus(Enum):
    """Organization status."""
    ACTIVE = "active"                # Active organization
    PENDING = "pending"              # Pending setup/verification
    SUSPENDED = "suspended"          # Suspended
    DELETED = "deleted"              # Soft deleted
    ARCHIVED = "archived"            # Archived
    RESTRICTED = "restricted"        # Restricted access


class TeamType(Enum):
    """Types of teams within organizations."""
    DEPARTMENT = "department"        # Department team
    PROJECT = "project"              # Project team
    CROSS_FUNCTIONAL = "cross_functional"  # Cross-functional team
    SUPPORT = "support"              # Support team
    SECURITY = "security"            # Security team
    RESEARCH = "research"            # Research team
    ADMINISTRATION = "administration"  # Administration team
    DEVELOPMENT = "development"      # Development team
    OPERATIONS = "operations"        # Operations team
    OTHER = "other"                  # Other team type


class MemberRole(Enum):
    """Member roles within organizations."""
    OWNER = "owner"                  # Organization owner
    ADMIN = "admin"                  # Organization administrator
    MANAGER = "manager"              # Team manager
    MEMBER = "member"                # Regular member
    VIEWER = "viewer"                # Read-only viewer
    CONTRIBUTOR = "contributor"      # Contributor (limited write)
    GUEST = "guest"                  # Guest (temporary access)
    EXTERNAL = "external"            # External collaborator


class InvitationStatus(Enum):
    """Invitation status."""
    PENDING = "pending"              # Invitation sent, pending acceptance
    ACCEPTED = "accepted"            # Invitation accepted
    DECLINED = "declined"            # Invitation declined
    EXPIRED = "expired"              # Invitation expired
    REVOKED = "revoked"              # Invitation revoked
    CANCELLED = "cancelled"          # Invitation cancelled


class PermissionScope(Enum):
    """Permission scopes."""
    ORGANIZATION = "organization"    # Organization-wide permissions
    TEAM = "team"                    # Team-specific permissions
    PROJECT = "project"              # Project-specific permissions
    RESOURCE = "resource"            # Resource-specific permissions
    PERSONAL = "personal"            # Personal permissions


class Organization(Base, UUIDMixin, TimestampMixin):
    """
    Organization model.
    
    This model represents an organization (company, non-profit, etc.)
    that can have multiple users, teams, and resources.
    
    Attributes:
        id: Primary key UUID
        name: Organization name
        slug: URL-friendly organization identifier
        description: Organization description
        type: Organization type
        status: Organization status
        website: Organization website URL
        email: Organization contact email
        phone: Organization phone number
        logo_url: URL to organization logo
        banner_url: URL to organization banner
        location: Organization location
        timezone: Organization timezone
        locale: Default locale/language
        industry: Organization industry
        size: Number of employees
        founded_year: Year organization was founded
        tax_id: Tax identification number
        registration_number: Business registration number
        settings: Organization settings
        metadata: Additional metadata
        billing_email: Billing contact email
        billing_address: Billing address
        subscription_id: Active subscription
        is_verified: Whether organization is verified
        verification_data: Verification information
        deleted_at: When organization was soft deleted
        tags: Categorization tags
    """
    
    __tablename__ = "organizations"
    
    # Basic information
    name = Column(String(200), nullable=False, index=True)
    slug = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    type = Column(SQLEnum(OrganizationType), nullable=False, index=True)
    status = Column(SQLEnum(OrganizationStatus), default=OrganizationStatus.PENDING, nullable=False, index=True)
    
    # Contact information
    website = Column(String(2000), nullable=True)
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    
    # Media
    logo_url = Column(String(2000), nullable=True)
    banner_url = Column(String(2000), nullable=True)
    favicon_url = Column(String(2000), nullable=True)
    
    # Location and time
    location = Column(JSONB, nullable=True)
    timezone = Column(String(100), default="UTC", nullable=False)
    locale = Column(String(10), default="en-US", nullable=False)
    
    # Business information
    industry = Column(String(200), nullable=True)
    size = Column(String(50), nullable=True)  # e.g., "1-10", "11-50", "51-200", etc.
    founded_year = Column(Integer, nullable=True)
    tax_id = Column(String(100), nullable=True)
    registration_number = Column(String(100), nullable=True)
    
    # Settings and configuration
    settings = Column(JSONB, default=dict, nullable=False)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Billing information
    billing_email = Column(String(255), nullable=True, index=True)
    billing_address = Column(JSONB, nullable=True)
    subscription_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("subscriptions.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Verification
    is_verified = Column(Boolean, default=False, nullable=False, index=True)
    verification_data = Column(JSONB, nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    verified_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True
    )
    
    # Deletion (soft delete)
    deleted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Categorization
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Relationships
    subscription = relationship("Subscription", foreign_keys=[subscription_id])
    verifier = relationship("User", foreign_keys=[verified_by])
    members = relationship("OrganizationMember", back_populates="organization", cascade="all, delete-orphan")
    teams = relationship("Team", back_populates="organization", cascade="all, delete-orphan")
    invites = relationship("OrganizationInvitation", back_populates="organization", cascade="all, delete-orphan")
    incidents = relationship("Incident", back_populates="organization")
    articles = relationship("Article", back_populates="organization")
    datasets = relationship("Dataset", back_populates="organization")
    audit_logs = relationship("OrganizationAuditLog", back_populates="organization")
    roles = relationship("OrganizationRole", back_populates="organization", cascade="all, delete-orphan")
    settings_config = relationship("OrganizationSettings", back_populates="organization", uselist=False, cascade="all, delete-orphan")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'founded_year IS NULL OR founded_year >= 1800',
            name='check_founded_year_reasonable'
        ),
        CheckConstraint(
            'slug ~* \'^[a-z0-9]+(?:-[a-z0-9]+)*$\'',
            name='check_slug_format'
        ),
        Index('ix_organizations_type_status', 'type', 'status'),
        Index('ix_organizations_name_slug', 'name', 'slug'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Organization(id={self.id}, name={self.name}, slug={self.slug})>"
    
    @property
    def is_active(self) -> bool:
        """Check if organization is active."""
        return self.status == OrganizationStatus.ACTIVE
    
    @property
    def is_suspended(self) -> bool:
        """Check if organization is suspended."""
        return self.status == OrganizationStatus.SUSPENDED
    
    @property
    def is_deleted(self) -> bool:
        """Check if organization is deleted."""
        return self.deleted_at is not None or self.status == OrganizationStatus.DELETED
    
    @property
    def member_count(self) -> int:
        """Get number of active members."""
        return len([m for m in self.members if m.is_active])
    
    @property
    def team_count(self) -> int:
        """Get number of teams."""
        return len(self.teams)
    
    @property
    def owner(self) -> Optional['OrganizationMember']:
        """Get organization owner."""
        for member in self.members:
            if member.role == MemberRole.OWNER:
                return member
        return None
    
    @property
    def owners(self) -> List['OrganizationMember']:
        """Get all organization owners."""
        return [m for m in self.members if m.role == MemberRole.OWNER]
    
    @property
    def admins(self) -> List['OrganizationMember']:
        """Get all organization admins."""
        return [m for m in self.members if m.role == MemberRole.ADMIN]
    
    @property
    def active_subscription(self) -> Optional['Subscription']:
        """Get active subscription if exists."""
        if self.subscription and self.subscription.is_active:
            return self.subscription
        return None
    
    @validates('slug')
    def validate_slug(self, key: str, slug: str) -> str:
        """Validate organization slug."""
        slug = slug.strip().lower()
        if not slug:
            raise ValueError("Organization slug cannot be empty")
        if len(slug) > 100:
            raise ValueError("Organization slug cannot exceed 100 characters")
        # Ensure slug contains only letters, numbers, and hyphens
        import re
        if not re.match(r'^[a-z0-9]+(?:-[a-z0-9]+)*$', slug):
            raise ValueError("Organization slug can only contain lowercase letters, numbers, and hyphens")
        return slug
    
    @validates('name')
    def validate_name(self, key: str, name: str) -> str:
        """Validate organization name."""
        name = name.strip()
        if not name:
            raise ValueError("Organization name cannot be empty")
        if len(name) > 200:
            raise ValueError("Organization name cannot exceed 200 characters")
        return name
    
    def add_member(
        self,
        user: 'User',
        role: MemberRole = MemberRole.MEMBER,
        invited_by: Optional['User'] = None,
        join_date: Optional[datetime] = None
    ) -> 'OrganizationMember':
        """Add a member to the organization."""
        from models.organization import OrganizationMember
        
        # Check if user is already a member
        existing_member = next(
            (m for m in self.members if m.user_id == user.id),
            None
        )
        
        if existing_member:
            # Update existing member
            existing_member.role = role
            existing_member.is_active = True
            existing_member.joined_at = join_date or datetime.utcnow()
            return existing_member
        
        # Create new member
        member = OrganizationMember(
            organization_id=self.id,
            user_id=user.id,
            role=role,
            invited_by=invited_by.id if invited_by else None,
            joined_at=join_date or datetime.utcnow(),
            is_active=True
        )
        self.members.append(member)
        return member
    
    def remove_member(self, user_id: uuid.UUID) -> bool:
        """Remove a member from the organization."""
        for member in self.members:
            if member.user_id == user_id:
                member.is_active = False
                member.left_at = datetime.utcnow()
                return True
        return False
    
    def change_member_role(self, user_id: uuid.UUID, new_role: MemberRole) -> bool:
        """Change a member's role."""
        for member in self.members:
            if member.user_id == user_id and member.is_active:
                member.role = new_role
                return True
        return False
    
    def create_team(
        self,
        name: str,
        description: Optional[str] = None,
        team_type: TeamType = TeamType.DEPARTMENT,
        parent_team_id: Optional[uuid.UUID] = None,
        manager_id: Optional[uuid.UUID] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> 'Team':
        """Create a new team in the organization."""
        from models.organization import Team
        
        team = Team(
            organization_id=self.id,
            name=name,
            description=description,
            team_type=team_type,
            parent_team_id=parent_team_id,
            manager_id=manager_id,
            settings=settings or {},
            is_active=True
        )
        self.teams.append(team)
        return team
    
    def verify(self, verified_by: 'User', verification_data: Optional[Dict[str, Any]] = None) -> None:
        """Verify the organization."""
        self.is_verified = True
        self.verified_at = datetime.utcnow()
        self.verified_by = verified_by.id
        if verification_data:
            self.verification_data = verification_data
    
    def suspend(self, reason: Optional[str] = None) -> None:
        """Suspend the organization."""
        self.status = OrganizationStatus.SUSPENDED
        if reason:
            self.metadata["suspension_reason"] = reason
            self.metadata["suspended_at"] = datetime.utcnow().isoformat()
    
    def activate(self) -> None:
        """Activate the organization."""
        self.status = OrganizationStatus.ACTIVE
        if "suspension_reason" in self.metadata:
            del self.metadata["suspension_reason"]
        if "suspended_at" in self.metadata:
            del self.metadata["suspended_at"]
    
    def soft_delete(self) -> None:
        """Soft delete the organization."""
        self.status = OrganizationStatus.DELETED
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore a soft-deleted organization."""
        self.status = OrganizationStatus.ACTIVE
        self.deleted_at = None
    
    def to_dict(self, include_members: bool = False, include_teams: bool = False) -> Dict[str, Any]:
        """Convert organization to dictionary."""
        result = {
            "id": str(self.id),
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "type": self.type.value,
            "status": self.status.value,
            "website": self.website,
            "email": self.email,
            "phone": self.phone,
            "logo_url": self.logo_url,
            "banner_url": self.banner_url,
            "favicon_url": self.favicon_url,
            "location": self.location,
            "timezone": self.timezone,
            "locale": self.locale,
            "industry": self.industry,
            "size": self.size,
            "founded_year": self.founded_year,
            "tax_id": self.tax_id,
            "registration_number": self.registration_number,
            "is_verified": self.is_verified,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "is_active": self.is_active,
            "is_suspended": self.is_suspended,
            "is_deleted": self.is_deleted,
            "member_count": self.member_count,
            "team_count": self.team_count,
            "billing_email": self.billing_email,
            "billing_address": self.billing_address,
            "subscription_id": str(self.subscription_id) if self.subscription_id else None,
            "settings": self.settings,
            "metadata": self.metadata,
            "tags": self.tags,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_members:
            result["members"] = [
                member.to_dict(include_user=True, include_organization=False)
                for member in self.members
                if member.is_active
            ]
            result["owners"] = [
                member.to_dict(include_user=True, include_organization=False)
                for member in self.owners
            ]
            result["admins"] = [
                member.to_dict(include_user=True, include_organization=False)
                for member in self.admins
            ]
        
        if include_teams:
            result["teams"] = [
                team.to_dict(include_members=False, include_organization=False)
                for team in self.teams
            ]
        
        if self.subscription:
            result["subscription"] = self.subscription.to_dict(include_plan=True, include_subscriber=False)
        
        if self.verifier:
            result["verified_by"] = {
                "id": str(self.verifier.id),
                "username": self.verifier.username
            }
        
        return result


class OrganizationMember(Base, UUIDMixin, TimestampMixin):
    """
    Organization member model.
    
    This model represents a user's membership in an organization,
    including their role and membership status.
    
    Attributes:
        id: Primary key UUID
        organization_id: Organization ID
        user_id: User ID
        role: Member role
        invited_by: User who invited this member
        joined_at: When member joined
        left_at: When member left
        is_active: Whether membership is active
        permissions: Custom permissions
        metadata: Additional metadata
    """
    
    __tablename__ = "organization_members"
    
    # Organization and user
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Role and status
    role = Column(SQLEnum(MemberRole), default=MemberRole.MEMBER, nullable=False, index=True)
    invited_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True
    )
    
    # Membership dates
    joined_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    left_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Permissions and metadata
    permissions = Column(JSONB, default=dict, nullable=False)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="members")
    user = relationship("User")
    inviter = relationship("User", foreign_keys=[invited_by])
    team_memberships = relationship("TeamMember", back_populates="organization_member", cascade="all, delete-orphan")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('organization_id', 'user_id', name='uq_organization_user'),
        Index('ix_organization_members_role_active', 'role', 'is_active'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<OrganizationMember(id={self.id}, org={self.organization_id}, user={self.user_id}, role={self.role.value})>"
    
    @property
    def membership_duration_days(self) -> int:
        """Get membership duration in days."""
        end_date = self.left_at or datetime.utcnow()
        duration = end_date - self.joined_at
        return duration.days
    
    @property
    def is_owner(self) -> bool:
        """Check if member is an owner."""
        return self.role == MemberRole.OWNER
    
    @property
    def is_admin(self) -> bool:
        """Check if member is an admin."""
        return self.role in [MemberRole.OWNER, MemberRole.ADMIN]
    
    @property
    def is_manager(self) -> bool:
        """Check if member is a manager."""
        return self.role in [MemberRole.OWNER, MemberRole.ADMIN, MemberRole.MANAGER]
    
    def has_permission(self, permission: str) -> bool:
        """Check if member has specific permission."""
        # Owners have all permissions
        if self.is_owner:
            return True
        
        # Check custom permissions
        if permission in self.permissions.get('granted', []):
            return True
        if permission in self.permissions.get('denied', []):
            return False
        
        # Check role-based permissions
        role_permissions = self._get_role_permissions()
        return permission in role_permissions
    
    def _get_role_permissions(self) -> List[str]:
        """Get permissions for member's role."""
        # This would typically come from a configuration or database
        # For now, define basic role permissions
        role_permission_map = {
            MemberRole.OWNER: [
                "organization.manage",
                "organization.billing",
                "organization.settings",
                "members.manage",
                "teams.manage",
                "content.manage",
                "data.manage"
            ],
            MemberRole.ADMIN: [
                "organization.manage",
                "members.manage",
                "teams.manage",
                "content.manage",
                "data.manage"
            ],
            MemberRole.MANAGER: [
                "teams.manage",
                "content.manage",
                "data.manage"
            ],
            MemberRole.MEMBER: [
                "content.create",
                "content.edit",
                "data.view",
                "data.create"
            ],
            MemberRole.VIEWER: [
                "content.view",
                "data.view"
            ],
            MemberRole.CONTRIBUTOR: [
                "content.create",
                "content.edit",
                "data.view"
            ],
            MemberRole.GUEST: [
                "content.view"
            ]
        }
        
        return role_permission_map.get(self.role, [])
    
    def grant_permission(self, permission: str) -> None:
        """Grant a custom permission to member."""
        granted = self.permissions.get('granted', [])
        if permission not in granted:
            granted.append(permission)
            self.permissions['granted'] = granted
        
        # Remove from denied if present
        denied = self.permissions.get('denied', [])
        if permission in denied:
            denied.remove(permission)
            self.permissions['denied'] = denied
    
    def revoke_permission(self, permission: str) -> None:
        """Revoke a custom permission from member."""
        granted = self.permissions.get('granted', [])
        if permission in granted:
            granted.remove(permission)
            self.permissions['granted'] = granted
        
        denied = self.permissions.get('denied', [])
        if permission not in denied:
            denied.append(permission)
            self.permissions['denied'] = denied
    
    def leave_organization(self) -> None:
        """Member leaves the organization."""
        self.is_active = False
        self.left_at = datetime.utcnow()
    
    def to_dict(self, include_user: bool = True, include_organization: bool = False) -> Dict[str, Any]:
        """Convert organization member to dictionary."""
        result = {
            "id": str(self.id),
            "organization_id": str(self.organization_id),
            "user_id": str(self.user_id),
            "role": self.role.value,
            "invited_by": str(self.invited_by) if self.invited_by else None,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
            "left_at": self.left_at.isoformat() if self.left_at else None,
            "is_active": self.is_active,
            "is_owner": self.is_owner,
            "is_admin": self.is_admin,
            "is_manager": self.is_manager,
            "membership_duration_days": self.membership_duration_days,
            "permissions": self.permissions,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_user and self.user:
            result["user"] = {
                "id": str(self.user.id),
                "username": self.user.username,
                "email": getattr(self.user, 'email', None),
                "full_name": getattr(self.user, 'full_name', None),
                "avatar_url": getattr(self.user, 'avatar_url', None)
            }
        
        if include_organization and self.organization:
            result["organization"] = {
                "id": str(self.organization.id),
                "name": self.organization.name,
                "slug": self.organization.slug
            }
        
        if self.inviter:
            result["inviter"] = {
                "id": str(self.inviter.id),
                "username": self.inviter.username
            }
        
        return result


class Team(Base, UUIDMixin, TimestampMixin):
    """
    Team model.
    
    This model represents a team within an organization.
    Teams can have hierarchical structures and specific purposes.
    
    Attributes:
        id: Primary key UUID
        organization_id: Organization ID
        parent_team_id: Parent team ID (for hierarchy)
        name: Team name
        slug: URL-friendly team identifier
        description: Team description
        team_type: Team type
        manager_id: Team manager user ID
        is_active: Whether team is active
        settings: Team settings
        metadata: Additional metadata
        tags: Categorization tags
    """
    
    __tablename__ = "teams"
    
    # Organization and hierarchy
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    parent_team_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("teams.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Basic information
    name = Column(String(200), nullable=False, index=True)
    slug = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    team_type = Column(SQLEnum(TeamType), nullable=False, index=True)
    
    # Management
    manager_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Settings and metadata
    settings = Column(JSONB, default=dict, nullable=False)
    metadata = Column(JSONB, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Relationships
    organization = relationship("Organization", back_populates="teams")
    parent_team = relationship("Team", remote_side=[id], backref="child_teams")
    manager = relationship("User", foreign_keys=[manager_id])
    members = relationship("TeamMember", back_populates="team", cascade="all, delete-orphan")
    projects = relationship("TeamProject", back_populates="team", cascade="all, delete-orphan")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('organization_id', 'slug', name='uq_team_organization_slug'),
        CheckConstraint(
            'slug ~* \'^[a-z0-9]+(?:-[a-z0-9]+)*$\'',
            name='check_team_slug_format'
        ),
        Index('ix_teams_organization_type', 'organization_id', 'team_type'),
        Index('ix_teams_parent_hierarchy', 'parent_team_id', 'organization_id'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Team(id={self.id}, org={self.organization_id}, name={self.name}, type={self.team_type.value})>"
    
    @property
    def member_count(self) -> int:
        """Get number of active team members."""
        return len([m for m in self.members if m.is_active])
    
    @property
    def direct_member_count(self) -> int:
        """Get number of direct team members (excluding inherited)."""
        return len([m for m in self.members if m.is_active and not m.is_inherited])
    
    @property
    def all_members(self) -> List['TeamMember']:
        """Get all team members including inherited."""
        all_members = list(self.members)
        
        # Add members from child teams
        for child in self.child_teams:
            if child.is_active:
                all_members.extend(child.all_members)
        
        return all_members
    
    @property
    def hierarchy_level(self) -> int:
        """Get hierarchy level (0 for root teams)."""
        if not self.parent_team_id:
            return 0
        if not self.parent_team:
            return 1
        return self.parent_team.hierarchy_level + 1
    
    @property
    def full_path(self) -> str:
        """Get full team path in hierarchy."""
        if not self.parent_team:
            return self.name
        return f"{self.parent_team.full_path} / {self.name}"
    
    def add_member(
        self,
        user_id: uuid.UUID,
        role: str = "member",
        is_inherited: bool = False,
        permissions: Optional[Dict[str, Any]] = None
    ) -> 'TeamMember':
        """Add a member to the team."""
        from models.organization import TeamMember
        
        # Check if user is already a member
        existing_member = next(
            (m for m in self.members if m.user_id == user_id and not m.is_inherited),
            None
        )
        
        if existing_member:
            # Update existing member
            existing_member.role = role
            existing_member.is_active = True
            existing_member.permissions = permissions or {}
            return existing_member
        
        # Create new member
        member = TeamMember(
            team_id=self.id,
            user_id=user_id,
            role=role,
            is_inherited=is_inherited,
            permissions=permissions or {},
            is_active=True
        )
        self.members.append(member)
        return member
    
    def remove_member(self, user_id: uuid.UUID) -> bool:
        """Remove a member from the team."""
        for member in self.members:
            if member.user_id == user_id and not member.is_inherited:
                member.is_active = False
                return True
        return False
    
    def get_member(self, user_id: uuid.UUID) -> Optional['TeamMember']:
        """Get team member by user ID."""
        for member in self.members:
            if member.user_id == user_id and member.is_active:
                return member
        return None
    
    def has_member(self, user_id: uuid.UUID) -> bool:
        """Check if user is a member of this team."""
        return self.get_member(user_id) is not None
    
    def create_child_team(
        self,
        name: str,
        slug: str,
        description: Optional[str] = None,
        team_type: TeamType = TeamType.PROJECT
    ) -> 'Team':
        """Create a child team."""
        from models.organization import Team
        
        child_team = Team(
            organization_id=self.organization_id,
            parent_team_id=self.id,
            name=name,
            slug=slug,
            description=description,
            team_type=team_type,
            is_active=True
        )
        return child_team
    
    def to_dict(self, include_members: bool = False, include_organization: bool = False) -> Dict[str, Any]:
        """Convert team to dictionary."""
        result = {
            "id": str(self.id),
            "organization_id": str(self.organization_id),
            "parent_team_id": str(self.parent_team_id) if self.parent_team_id else None,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "team_type": self.team_type.value,
            "manager_id": str(self.manager_id) if self.manager_id else None,
            "is_active": self.is_active,
            "member_count": self.member_count,
            "direct_member_count": self.direct_member_count,
            "hierarchy_level": self.hierarchy_level,
            "full_path": self.full_path,
            "settings": self.settings,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_members:
            result["members"] = [
                member.to_dict(include_user=True, include_team=False)
                for member in self.members
                if member.is_active
            ]
        
        if include_organization and self.organization:
            result["organization"] = {
                "id": str(self.organization.id),
                "name": self.organization.name,
                "slug": self.organization.slug
            }
        
        if self.parent_team:
            result["parent_team"] = {
                "id": str(self.parent_team.id),
                "name": self.parent_team.name,
                "slug": self.parent_team.slug
            }
        
        if self.manager:
            result["manager"] = {
                "id": str(self.manager.id),
                "username": self.manager.username
            }
        
        return result


class TeamMember(Base, UUIDMixin, TimestampMixin):
    """
    Team member model.
    
    This model represents a user's membership in a team.
    
    Attributes:
        id: Primary key UUID
        team_id: Team ID
        user_id: User ID
        organization_member_id: Reference to organization membership
        role: Team role
        is_inherited: Whether membership is inherited from parent team
        is_active: Whether membership is active
        permissions: Team-specific permissions
        metadata: Additional metadata
    """
    
    __tablename__ = "team_members"
    
    # Team and user
    team_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("teams.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    organization_member_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organization_members.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    
    # Role and status
    role = Column(String(50), default="member", nullable=False, index=True)
    is_inherited = Column(Boolean, default=False, nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Permissions and metadata
    permissions = Column(JSONB, default=dict, nullable=False)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    team = relationship("Team", back_populates="members")
    user = relationship("User")
    organization_member = relationship("OrganizationMember", back_populates="team_memberships")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('team_id', 'user_id', name='uq_team_user'),
        Index('ix_team_members_org_member', 'organization_member_id'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<TeamMember(id={self.id}, team={self.team_id}, user={self.user_id}, role={self.role})>"
    
    @property
    def organization_id(self) -> Optional[uuid.UUID]:
        """Get organization ID through team."""
        return self.team.organization_id if self.team else None
    
    def to_dict(self, include_user: bool = True, include_team: bool = False) -> Dict[str, Any]:
        """Convert team member to dictionary."""
        result = {
            "id": str(self.id),
            "team_id": str(self.team_id),
            "user_id": str(self.user_id),
            "organization_member_id": str(self.organization_member_id) if self.organization_member_id else None,
            "role": self.role,
            "is_inherited": self.is_inherited,
            "is_active": self.is_active,
            "permissions": self.permissions,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_user and self.user:
            result["user"] = {
                "id": str(self.user.id),
                "username": self.user.username,
                "email": getattr(self.user, 'email', None)
            }
        
        if include_team and self.team:
            result["team"] = {
                "id": str(self.team.id),
                "name": self.team.name,
                "slug": self.team.slug
            }
        
        return result


class OrganizationInvitation(Base, UUIDMixin, TimestampMixin):
    """
    Organization invitation model.
    
    This model represents invitations sent to users to join an organization.
    
    Attributes:
        id: Primary key UUID
        organization_id: Organization ID
        email: Invited email address
        invited_by: User who sent invitation
        token: Invitation token
        status: Invitation status
        role: Role being offered
        expires_at: When invitation expires
        accepted_at: When invitation was accepted
        declined_at: When invitation was declined
        metadata: Additional metadata
    """
    
    __tablename__ = "organization_invitations"
    
    # Organization and invitee
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    email = Column(String(255), nullable=False, index=True)
    invited_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Invitation details
    token = Column(String(200), nullable=False, unique=True, index=True)
    status = Column(SQLEnum(InvitationStatus), default=InvitationStatus.PENDING, nullable=False, index=True)
    role = Column(SQLEnum(MemberRole), default=MemberRole.MEMBER, nullable=False, index=True)
    
    # Dates
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    declined_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="invites")
    inviter = relationship("User", foreign_keys=[invited_by])
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('organization_id', 'email', 'status', name='uq_organization_email_pending'),
        Index('ix_invitations_token_status', 'token', 'status'),
        Index('ix_invitations_email_status', 'email', 'status'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<OrganizationInvitation(id={self.id}, org={self.organization_id}, email={self.email}, status={self.status.value})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if invitation has expired."""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if invitation is valid (pending and not expired)."""
        return self.status == InvitationStatus.PENDING and not self.is_expired
    
    @property
    def days_until_expiry(self) -> int:
        """Get days until invitation expires."""
        remaining = self.expires_at - datetime.utcnow()
        return max(0, remaining.days)
    
    def accept(self) -> None:
        """Accept the invitation."""
        if not self.is_valid:
            raise ValueError("Invitation is not valid")
        
        self.status = InvitationStatus.ACCEPTED
        self.accepted_at = datetime.utcnow()
    
    def decline(self) -> None:
        """Decline the invitation."""
        if not self.is_valid:
            raise ValueError("Invitation is not valid")
        
        self.status = InvitationStatus.DECLINED
        self.declined_at = datetime.utcnow()
    
    def revoke(self) -> None:
        """Revoke the invitation."""
        if self.status != InvitationStatus.PENDING:
            raise ValueError("Only pending invitations can be revoked")
        
        self.status = InvitationStatus.REVOKED
    
    def extend(self, days: int = 7) -> None:
        """Extend invitation expiration."""
        if not self.is_valid:
            raise ValueError("Only valid invitations can be extended")
        
        self.expires_at = self.expires_at + timedelta(days=days)
    
    def to_dict(self, include_organization: bool = False, include_inviter: bool = False) -> Dict[str, Any]:
        """Convert invitation to dictionary."""
        result = {
            "id": str(self.id),
            "organization_id": str(self.organization_id),
            "email": self.email,
            "invited_by": str(self.invited_by),
            "token": self.token,
            "status": self.status.value,
            "role": self.role.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "declined_at": self.declined_at.isoformat() if self.declined_at else None,
            "is_expired": self.is_expired,
            "is_valid": self.is_valid,
            "days_until_expiry": self.days_until_expiry,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_organization and self.organization:
            result["organization"] = {
                "id": str(self.organization.id),
                "name": self.organization.name,
                "slug": self.organization.slug
            }
        
        if include_inviter and self.inviter:
            result["inviter"] = {
                "id": str(self.inviter.id),
                "username": self.inviter.username
            }
        
        return result


class OrganizationRole(Base, UUIDMixin, TimestampMixin):
    """
    Custom organization role model.
    
    This model allows organizations to define custom roles
    with specific permissions.
    
    Attributes:
        id: Primary key UUID
        organization_id: Organization ID
        name: Role name
        description: Role description
        permissions: Role permissions
        is_default: Whether this is a default role
        is_system: Whether this is a system role
        metadata: Additional metadata
    """
    
    __tablename__ = "organization_roles"
    
    # Organization
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Role information
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    permissions = Column(JSONB, default=dict, nullable=False)
    
    # Role type
    is_default = Column(Boolean, default=False, nullable=False, index=True)
    is_system = Column(Boolean, default=False, nullable=False, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="roles")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('organization_id', 'name', name='uq_organization_role_name'),
        Index('ix_organization_roles_type', 'is_default', 'is_system'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<OrganizationRole(id={self.id}, org={self.organization_id}, name={self.name})>"
    
    def has_permission(self, permission: str) -> bool:
        """Check if role has specific permission."""
        granted = self.permissions.get('granted', [])
        denied = self.permissions.get('denied', [])
        
        if permission in denied:
            return False
        
        return permission in granted or self.permissions.get('inherit_all', False)
    
    def grant_permission(self, permission: str) -> None:
        """Grant permission to role."""
        granted = self.permissions.get('granted', [])
        if permission not in granted:
            granted.append(permission)
            self.permissions['granted'] = granted
        
        # Remove from denied if present
        denied = self.permissions.get('denied', [])
        if permission in denied:
            denied.remove(permission)
            self.permissions['denied'] = denied
    
    def revoke_permission(self, permission: str) -> None:
        """Revoke permission from role."""
        granted = self.permissions.get('granted', [])
        if permission in granted:
            granted.remove(permission)
            self.permissions['granted'] = granted
        
        denied = self.permissions.get('denied', [])
        if permission not in denied:
            denied.append(permission)
            self.permissions['denied'] = denied
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert organization role to dictionary."""
        return {
            "id": str(self.id),
            "organization_id": str(self.organization_id),
            "name": self.name,
            "description": self.description,
            "permissions": self.permissions,
            "is_default": self.is_default,
            "is_system": self.is_system,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class OrganizationSettings(Base, UUIDMixin, TimestampMixin):
    """
    Organization settings model.
    
    This model stores organization-specific settings and configuration.
    
    Attributes:
        id: Primary key UUID
        organization_id: Organization ID
        settings: Organization settings
        preferences: User preferences defaults
        security_settings: Security configuration
        compliance_settings: Compliance configuration
        notification_settings: Notification configuration
        integration_settings: Integration configuration
        metadata: Additional metadata
    """
    
    __tablename__ = "organization_settings"
    
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="CASCADE"), 
        nullable=False,
        unique=True,
        index=True
    )
    
    # Settings categories
    settings = Column(JSONB, default=dict, nullable=False)
    preferences = Column(JSONB, default=dict, nullable=False)
    security_settings = Column(JSONB, default=dict, nullable=False)
    compliance_settings = Column(JSONB, default=dict, nullable=False)
    notification_settings = Column(JSONB, default=dict, nullable=False)
    integration_settings = Column(JSONB, default=dict, nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="settings_config")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<OrganizationSettings(id={self.id}, org={self.organization_id})>"
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        keys = key.split('.')
        current = self.settings
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value."""
        keys = key.split('.')
        current = self.settings
        
        for i, k in enumerate(keys[:-1]):
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert organization settings to dictionary."""
        return {
            "id": str(self.id),
            "organization_id": str(self.organization_id),
            "settings": self.settings,
            "preferences": self.preferences,
            "security_settings": self.security_settings,
            "compliance_settings": self.compliance_settings,
            "notification_settings": self.notification_settings,
            "integration_settings": self.integration_settings,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class OrganizationAuditLog(Base, UUIDMixin, TimestampMixin):
    """
    Organization audit log model.
    
    This model logs important actions and events within an organization
    for compliance and security purposes.
    
    Attributes:
        id: Primary key UUID
        organization_id: Organization ID
        user_id: User who performed action
        action: Action performed
        resource_type: Type of resource affected
        resource_id: ID of resource affected
        details: Action details
        ip_address: IP address of requester
        user_agent: User agent string
        metadata: Additional metadata
    """
    
    __tablename__ = "organization_audit_logs"
    
    # Organization and user
    organization_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("organizations.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Action details
    action = Column(String(200), nullable=False, index=True)
    resource_type = Column(String(100), nullable=True, index=True)
    resource_id = Column(String(100), nullable=True, index=True)
    details = Column(JSONB, default=dict, nullable=False)
    
    # Request information
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="audit_logs")
    user = relationship("User")
    
    # Check constraints
    __table_args__ = (
        Index('ix_audit_logs_action_resource', 'action', 'resource_type', 'resource_id'),
        Index('ix_audit_logs_organization_date', 'organization_id', 'created_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<OrganizationAuditLog(id={self.id}, org={self.organization_id}, action={self.action})>"
    
    @classmethod
    def log(
        cls,
        organization_id: uuid.UUID,
        user_id: Optional[uuid.UUID],
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'OrganizationAuditLog':
        """Create an audit log entry."""
        return cls(
            organization_id=organization_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {}
        )
    
    def to_dict(self, include_user: bool = False) -> Dict[str, Any]:
        """Convert audit log to dictionary."""
        result = {
            "id": str(self.id),
            "organization_id": str(self.organization_id),
            "user_id": str(self.user_id) if self.user_id else None,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
        
        if include_user and self.user:
            result["user"] = {
                "id": str(self.user.id),
                "username": self.user.username
            }
        
        return result


class TeamProject(Base, UUIDMixin, TimestampMixin):
    """
    Team project model.
    
    This model represents projects associated with teams.
    
    Attributes:
        id: Primary key UUID
        team_id: Team ID
        name: Project name
        description: Project description
        status: Project status
        start_date: Project start date
        end_date: Project end date
        budget: Project budget
        settings: Project settings
        metadata: Additional metadata
        tags: Categorization tags
    """
    
    __tablename__ = "team_projects"
    
    # Team
    team_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("teams.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Project information
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="active", nullable=False, index=True)
    
    # Dates and budget
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    budget = Column(Numeric(15, 2), nullable=True)
    
    # Settings and metadata
    settings = Column(JSONB, default=dict, nullable=False)
    metadata = Column(JSONB, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Relationships
    team = relationship("Team", back_populates="projects")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<TeamProject(id={self.id}, team={self.team_id}, name={self.name})>"


# Helper functions
def generate_invitation_token() -> str:
    """Generate unique invitation token."""
    import secrets
    import hashlib
    import base64
    
    random_bytes = secrets.token_bytes(32)
    token_hash = hashlib.sha256(random_bytes).digest()
    token = base64.urlsafe_b64encode(token_hash).decode('ascii')
    return token[:64]


def validate_organization_slug(slug: str) -> bool:
    """Validate organization slug format."""
    import re
    pattern = r'^[a-z0-9]+(?:-[a-z0-9]+)*$'
    return bool(re.match(pattern, slug))


def get_default_organization_settings() -> Dict[str, Any]:
    """Get default organization settings."""
    return {
        "general": {
            "allow_public_access": False,
            "require_approval_for_content": False,
            "auto_approve_members": False,
            "default_locale": "en-US",
            "timezone": "UTC"
        },
        "security": {
            "require_2fa": False,
            "password_policy": {
                "min_length": 8,
                "require_special": True,
                "require_number": True,
                "require_uppercase": True
            },
            "session_timeout_minutes": 120,
            "max_login_attempts": 5
        },
        "notifications": {
            "email_notifications": True,
            "in_app_notifications": True,
            "daily_digest": True,
            "weekly_report": True
        },
        "billing": {
            "auto_renew": True,
            "invoice_email": True,
            "receipt_email": True
        }
    }