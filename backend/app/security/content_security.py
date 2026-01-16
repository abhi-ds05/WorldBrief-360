"""
Content Security Policy and Security Headers Implementation

This module provides:
- Content Security Policy (CSP) generation and validation
- Security headers configuration
- XSS protection headers
- HSTS configuration
- Referrer policy
- Feature policy and permissions policy
- CORS security configurations
- Security header validation and testing
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, validator, HttpUrl

from app.core.config import get_settings
from app.security.audit_logger import AuditLogger, AuditEventType

# Get settings
settings = get_settings()


class DirectiveSource(str, Enum):
    """CSP directive source values."""
    NONE = "'none'"
    SELF = "'self'"
    UNSAFE_INLINE = "'unsafe-inline'"
    UNSAFE_EVAL = "'unsafe-eval'"
    UNSAFE_HASHES = "'unsafe-hashes'"
    STRICT_DYNAMIC = "'strict-dynamic'"
    REPORT_SAMPLE = "'report-sample'"
    BLOB = "blob:"
    DATA = "data:"
    HTTPS = "https:"
    HTTP = "http:"
    WS = "ws:"
    WSS = "wss:"


class ReportToGroup(str, Enum):
    """Report-To header groups."""
    CSP = "csp"
    NEL = "nel"
    DEFAULT = "default"


class FeaturePolicyDirective(str, Enum):
    """Feature Policy directives."""
    ACCELEROMETER = "accelerometer"
    AMBIENT_LIGHT_SENSOR = "ambient-light-sensor"
    AUTOPLAY = "autoplay"
    CAMERA = "camera"
    ENCRYPTED_MEDIA = "encrypted-media"
    FULLSCREEN = "fullscreen"
    GEOLOCATION = "geolocation"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    MICROPHONE = "microphone"
    MIDI = "midi"
    PAYMENT = "payment"
    PICTURE_IN_PICTURE = "picture-in-picture"
    SPEAKER = "speaker"
    USB = "usb"
    VR = "vr"
    WAKE_LOCK = "wake-lock"


class PermissionPolicyDirective(str, Enum):
    """Permissions Policy directives (replaces Feature Policy)."""
    ACCELEROMETER = "accelerometer"
    AMBIENT_LIGHT_SENSOR = "ambient-light-sensor"
    AUTOPLAY = "autoplay"
    BATTERY = "battery"
    CAMERA = "camera"
    DISPLAY_CAPTURE = "display-capture"
    DOCUMENT_DOMAIN = "document-domain"
    ENCRYPTED_MEDIA = "encrypted-media"
    EXECUTION_WHILE_NOT_RENDERED = "execution-while-not-rendered"
    EXECUTION_WHILE_OUT_OF_VIEWPORT = "execution-while-out-of-viewport"
    FULLSCREEN = "fullscreen"
    GAMEPAD = "gamepad"
    GEOLOCATION = "geolocation"
    GYROSCOPE = "gyroscope"
    HID = "hid"
    IDENTITY_CREDENTIALS_GET = "identity-credentials-get"
    IDLE_DETECTION = "idle-detection"
    LOCAL_FONTS = "local-fonts"
    MAGNETOMETER = "magnetometer"
    MICROPHONE = "microphone"
    MIDI = "midi"
    OTK_CREDENTIALS = "otk-credentials"
    PAYMENT = "payment"
    PICTURE_IN_PICTURE = "picture-in-picture"
    PUBLICKEY_CREDENTIALS_GET = "publickey-credentials-get"
    SCREEN_WAKE_LOCK = "screen-wake-lock"
    SERIAL = "serial"
    SPEAKER_SELECTION = "speaker-selection"
    STORAGE_ACCESS = "storage-access"
    USB = "usb"
    WEB_SHARE = "web-share"
    XR_SPATIAL_TRACKING = "xr-spatial-tracking"


@dataclass
class CSPDirective:
    """Content Security Policy directive configuration."""
    name: str
    sources: List[Union[str, DirectiveSource]] = field(default_factory=list)
    allow_unsafe_inline: bool = False
    allow_unsafe_eval: bool = False
    allow_data: bool = False
    allow_blob: bool = False
    allow_https: bool = False
    allow_http: bool = False
    allow_ws: bool = False
    allow_wss: bool = False
    enable_strict_dynamic: bool = False
    enable_report_sample: bool = False
    
    def build(self) -> str:
        """Build directive string."""
        values = []
        
        # Add special sources
        if self.allow_unsafe_inline:
            values.append(DirectiveSource.UNSAFE_INLINE.value)
        if self.allow_unsafe_eval:
            values.append(DirectiveSource.UNSAFE_EVAL.value)
        if self.enable_strict_dynamic:
            values.append(DirectiveSource.STRICT_DYNAMIC.value)
        if self.enable_report_sample:
            values.append(DirectiveSource.REPORT_SAMPLE.value)
        
        # Add protocol sources
        if self.allow_data:
            values.append(DirectiveSource.DATA.value)
        if self.allow_blob:
            values.append(DirectiveSource.BLOB.value)
        if self.allow_https:
            values.append(DirectiveSource.HTTPS.value)
        if self.allow_http:
            values.append(DirectiveSource.HTTP.value)
        if self.allow_ws:
            values.append(DirectiveSource.WS.value)
        if self.allow_wss:
            values.append(DirectiveSource.WSS.value)
        
        # Add custom sources
        for source in self.sources:
            if isinstance(source, DirectiveSource):
                values.append(source.value)
            else:
                # Validate source URL
                if self._is_valid_source(source):
                    values.append(source)
        
        # If no sources specified, default to 'none'
        if not values:
            values.append(DirectiveSource.NONE.value)
        
        return f"{self.name} {' '.join(values)}".strip()
    
    def _is_valid_source(self, source: str) -> bool:
        """Validate source URL or pattern."""
        # Check for special keywords
        if source in [s.value for s in DirectiveSource]:
            return True
        
        # Check for wildcard
        if source == "*":
            return True
        
        # Check for valid URL pattern
        if source.startswith(("http://", "https://", "ws://", "wss://")):
            try:
                result = urlparse(source)
                return all([result.scheme, result.netloc])
            except:
                return False
        
        # Check for domain pattern (e.g., *.example.com)
        if source.startswith("*."):
            domain = source[2:]
            return self._is_valid_domain(domain)
        
        # Check for scheme source (e.g., https:)
        if source.endswith(":"):
            return source[:-1] in ["http", "https", "ws", "wss", "data", "blob"]
        
        return True
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Validate domain name."""
        domain_pattern = re.compile(
            r'^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z0-9-]{1,63})*$'
        )
        return bool(domain_pattern.match(domain))


@dataclass
class SecurityPolicy:
    """Complete security policy configuration."""
    # CSP Configuration
    csp_enabled: bool = True
    csp_report_only: bool = False
    csp_report_uri: Optional[str] = None
    csp_report_to: Optional[str] = None
    csp_upgrade_insecure_requests: bool = True
    csp_block_all_mixed_content: bool = True
    csp_require_sri_for: List[str] = field(default_factory=list)  # 'script', 'style'
    
    # Security Headers
    enable_hsts: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False
    
    enable_x_frame_options: bool = True
    x_frame_options: str = "DENY"  # DENY, SAMEORIGIN, ALLOW-FROM uri
    
    enable_x_content_type_options: bool = True
    x_content_type_options: str = "nosniff"
    
    enable_x_xss_protection: bool = True
    x_xss_protection: str = "1; mode=block"
    
    enable_referrer_policy: bool = True
    referrer_policy: str = "strict-origin-when-cross-origin"
    
    enable_expect_ct: bool = False
    expect_ct_max_age: int = 86400  # 24 hours
    expect_ct_enforce: bool = False
    expect_ct_report_uri: Optional[str] = None
    
    enable_permissions_policy: bool = True
    enable_feature_policy: bool = False  # Deprecated, use permissions-policy
    
    enable_cross_origin_opener_policy: bool = True
    cross_origin_opener_policy: str = "same-origin"  # same-origin, same-origin-allow-popups, unsafe-none
    
    enable_cross_origin_embedder_policy: bool = True
    cross_origin_embedder_policy: str = "require-corp"  # require-corp, credentialless
    
    enable_cross_origin_resource_policy: bool = True
    cross_origin_resource_policy: str = "same-origin"  # same-origin, same-site, cross-origin
    
    # Other security headers
    enable_cache_control: bool = True
    cache_control: str = "no-store, no-cache, must-revalidate, max-age=0"
    
    enable_pragma: bool = True
    pragma: str = "no-cache"
    
    enable_x_permitted_cross_domain_policies: bool = True
    x_permitted_cross_domain_policies: str = "none"


class ContentSecurityPolicy:
    """
    Content Security Policy generator and validator.
    
    CSP helps prevent XSS, clickjacking, and other code injection attacks.
    """
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.directives: Dict[str, CSPDirective] = {}
        self.audit_logger = AuditLogger()
        
        # Initialize default directives
        self._init_default_directives()
    
    def _init_default_directives(self):
        """Initialize default CSP directives."""
        # Default strict policy
        self.directives = {
            "default-src": CSPDirective(
                name="default-src",
                sources=[DirectiveSource.SELF],
                allow_https=True,
                allow_wss=True,
            ),
            "script-src": CSPDirective(
                name="script-src",
                sources=[DirectiveSource.SELF],
                allow_unsafe_inline=False,
                allow_unsafe_eval=False,
            ),
            "style-src": CSPDirective(
                name="style-src",
                sources=[DirectiveSource.SELF],
                allow_unsafe_inline=True,  # Usually needed for inline styles
            ),
            "img-src": CSPDirective(
                name="img-src",
                sources=[DirectiveSource.SELF],
                allow_data=True,  # For data: URLs
                allow_https=True,
            ),
            "font-src": CSPDirective(
                name="font-src",
                sources=[DirectiveSource.SELF],
                allow_https=True,
            ),
            "connect-src": CSPDirective(
                name="connect-src",
                sources=[DirectiveSource.SELF],
                allow_wss=True,
            ),
            "frame-src": CSPDirective(
                name="frame-src",
                sources=[DirectiveSource.NONE],  # Block by default
            ),
            "object-src": CSPDirective(
                name="object-src",
                sources=[DirectiveSource.NONE],  # Block by default
            ),
            "media-src": CSPDirective(
                name="media-src",
                sources=[DirectiveSource.SELF],
            ),
            "manifest-src": CSPDirective(
                name="manifest-src",
                sources=[DirectiveSource.SELF],
            ),
            "form-action": CSPDirective(
                name="form-action",
                sources=[DirectiveSource.SELF],
            ),
            "base-uri": CSPDirective(
                name="base-uri",
                sources=[DirectiveSource.SELF],
            ),
            "frame-ancestors": CSPDirective(
                name="frame-ancestors",
                sources=[DirectiveSource.NONE],  # Prevent clickjacking
            ),
            "worker-src": CSPDirective(
                name="worker-src",
                sources=[DirectiveSource.SELF],
                allow_blob=True,
            ),
            "child-src": CSPDirective(
                name="child-src",
                sources=[DirectiveSource.SELF],
            ),
        }
    
    def add_directive(self, directive: CSPDirective):
        """Add or update a CSP directive."""
        self.directives[directive.name] = directive
    
    def add_source(self, directive_name: str, source: Union[str, DirectiveSource]):
        """Add a source to a directive."""
        if directive_name in self.directives:
            self.directives[directive_name].sources.append(source)
    
    def remove_directive(self, directive_name: str):
        """Remove a CSP directive."""
        if directive_name in self.directives:
            del self.directives[directive_name]
    
    def generate_csp_header(self) -> str:
        """Generate CSP header string."""
        directives = []
        
        # Add regular directives
        for directive in self.directives.values():
            directives.append(directive.build())
        
        # Add special directives
        if self.policy.csp_upgrade_insecure_requests:
            directives.append("upgrade-insecure-requests")
        
        if self.policy.csp_block_all_mixed_content:
            directives.append("block-all-mixed-content")
        
        if self.policy.csp_require_sri_for:
            require_str = " ".join(self.policy.csp_require_sri_for)
            directives.append(f"require-sri-for {require_str}")
        
        if self.policy.csp_report_uri:
            directives.append(f"report-uri {self.policy.csp_report_uri}")
        
        if self.policy.csp_report_to:
            directives.append(f"report-to {self.policy.csp_report_to}")
        
        return "; ".join(directives)
    
    def validate_csp(self, csp_string: str) -> Tuple[bool, List[str]]:
        """
        Validate a CSP string.
        
        Args:
            csp_string: CSP header string
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if CSP is empty
        if not csp_string or not csp_string.strip():
            errors.append("CSP string is empty")
            return False, errors
        
        # Parse directives
        directives = csp_string.split(";")
        
        for directive in directives:
            directive = directive.strip()
            if not directive:
                continue
            
            # Split directive name and values
            if " " in directive:
                name, values = directive.split(" ", 1)
                name = name.strip()
                values = values.strip()
            else:
                name = directive
                values = ""
            
            # Validate directive name
            if not self._is_valid_directive_name(name):
                errors.append(f"Invalid directive name: {name}")
            
            # Validate values for certain directives
            if name in ["default-src", "script-src", "style-src"]:
                if "'unsafe-inline'" in values and "'unsafe-eval'" in values:
                    errors.append(f"Directive {name} allows both unsafe-inline and unsafe-eval - security risk")
            
            if name == "frame-ancestors" and values in ["'none'", "'self'"]:
                # Good practice
                pass
            elif name == "frame-ancestors" and "*" in values:
                errors.append("frame-ancestors should not allow * - clickjacking risk")
        
        return len(errors) == 0, errors
    
    def _is_valid_directive_name(self, name: str) -> bool:
        """Check if directive name is valid."""
        valid_directives = {
            "default-src", "script-src", "style-src", "img-src", "font-src",
            "connect-src", "frame-src", "object-src", "media-src", "manifest-src",
            "form-action", "base-uri", "frame-ancestors", "worker-src", "child-src",
            "upgrade-insecure-requests", "block-all-mixed-content", "require-sri-for",
            "report-uri", "report-to", "sandbox", "script-src-attr", "script-src-elem",
            "style-src-attr", "style-src-elem", "prefetch-src", "navigate-to"
        }
        return name in valid_directives
    
    def generate_report_only_header(self) -> str:
        """Generate CSP-Report-Only header string."""
        csp_header = self.generate_csp_header()
        
        # Remove report-uri from regular CSP if present
        if self.policy.csp_report_uri:
            csp_header = csp_header.replace(f"report-uri {self.policy.csp_report_uri}", "")
            csp_header = csp_header.replace(";;", ";").strip(";")
        
        return csp_header
    
    def test_csp_violation(self, request: Request, violation_data: Dict[str, Any]) -> bool:
        """
        Test CSP violation (for development/testing).
        
        Args:
            request: FastAPI request
            violation_data: CSP violation report data
            
        Returns:
            True if violation is legitimate, False if false positive
        """
        # Log the violation for analysis
        self.audit_logger.log_security_event(
            event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
            description="CSP Violation Detected",
            details={
                "violation": violation_data,
                "path": request.url.path,
                "method": request.method,
                "user_agent": request.headers.get("User-Agent"),
                "ip_address": request.client.host if request.client else None,
            },
            severity="WARNING"
        )
        
        # Analyze violation
        blocked_uri = violation_data.get("blocked-uri", "")
        violated_directive = violation_data.get("violated-directive", "")
        
        # Check for common false positives
        if blocked_uri in ["inline", "eval", "data"]:
            # Might need to adjust CSP
            return True
        
        if "chrome-extension" in blocked_uri or "moz-extension" in blocked_uri:
            # Browser extensions - usually safe
            return False
        
        return True


class SecurityHeaders:
    """
    Security headers generator and middleware.
    """
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.csp = ContentSecurityPolicy(policy)
        self.audit_logger = AuditLogger()
    
    def generate_headers(self, request: Optional[Request] = None) -> Dict[str, str]:
        """
        Generate all security headers.
        
        Args:
            request: Optional request for context-aware headers
            
        Returns:
            Dictionary of security headers
        """
        headers = {}
        
        # Content Security Policy
        if self.policy.csp_enabled:
            if self.policy.csp_report_only:
                headers["Content-Security-Policy-Report-Only"] = self.csp.generate_report_only_header()
            else:
                headers["Content-Security-Policy"] = self.csp.generate_csp_header()
        
        # HTTP Strict Transport Security
        if self.policy.enable_hsts:
            hsts_value = f"max-age={self.policy.hsts_max_age}"
            if self.policy.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.policy.hsts_preload:
                hsts_value += "; preload"
            headers["Strict-Transport-Security"] = hsts_value
        
        # X-Frame-Options
        if self.policy.enable_x_frame_options:
            headers["X-Frame-Options"] = self.policy.x_frame_options
        
        # X-Content-Type-Options
        if self.policy.enable_x_content_type_options:
            headers["X-Content-Type-Options"] = self.policy.x_content_type_options
        
        # X-XSS-Protection
        if self.policy.enable_x_xss_protection:
            headers["X-XSS-Protection"] = self.policy.x_xss_protection
        
        # Referrer Policy
        if self.policy.enable_referrer_policy:
            headers["Referrer-Policy"] = self.policy.referrer_policy
        
        # Expect-CT
        if self.policy.enable_expect_ct:
            expect_ct_value = f"max-age={self.policy.expect_ct_max_age}"
            if self.policy.expect_ct_enforce:
                expect_ct_value += ", enforce"
            if self.policy.expect_ct_report_uri:
                expect_ct_value += f', report-uri="{self.policy.expect_ct_report_uri}"'
            headers["Expect-CT"] = expect_ct_value
        
        # Permissions Policy / Feature Policy
        if self.policy.enable_permissions_policy:
            headers["Permissions-Policy"] = self._generate_permissions_policy()
        
        if self.policy.enable_feature_policy:
            headers["Feature-Policy"] = self._generate_feature_policy()
        
        # Cross-Origin Policies
        if self.policy.enable_cross_origin_opener_policy:
            headers["Cross-Origin-Opener-Policy"] = self.policy.cross_origin_opener_policy
        
        if self.policy.enable_cross_origin_embedder_policy:
            headers["Cross-Origin-Embedder-Policy"] = self.policy.cross_origin_embedder_policy
        
        if self.policy.enable_cross_origin_resource_policy:
            headers["Cross-Origin-Resource-Policy"] = self.policy.cross_origin_resource_policy
        
        # Cache control
        if self.policy.enable_cache_control:
            headers["Cache-Control"] = self.policy.cache_control
        
        # Pragma
        if self.policy.enable_pragma:
            headers["Pragma"] = self.policy.pragma
        
        # X-Permitted-Cross-Domain-Policies
        if self.policy.enable_x_permitted_cross_domain_policies:
            headers["X-Permitted-Cross-Domain-Policies"] = self.policy.x_permitted_cross_domain_policies
        
        # Server header removal/obfuscation
        headers["Server"] = "WorldBrief360"  # Generic server name
        
        # X-Powered-By removal
        headers["X-Powered-By"] = "WorldBrief360"
        
        return headers
    
    def _generate_permissions_policy(self) -> str:
        """Generate Permissions-Policy header."""
        policies = []
        
        # Default restrictive policy
        default_policies = {
            PermissionPolicyDirective.ACCELEROMETER: "()",
            PermissionPolicyDirective.AMBIENT_LIGHT_SENSOR: "()",
            PermissionPolicyDirective.AUTOPLAY: "()",
            PermissionPolicyDirective.CAMERA: "()",
            PermissionPolicyDirective.ENCRYPTED_MEDIA: "()",
            PermissionPolicyDirective.FULLSCREEN: "()",
            PermissionPolicyDirective.GEOLOCATION: "()",
            PermissionPolicyDirective.GYROSCOPE: "()",
            PermissionPolicyDirective.MAGNETOMETER: "()",
            PermissionPolicyDirective.MICROPHONE: "()",
            PermissionPolicyDirective.MIDI: "()",
            PermissionPolicyDirective.PAYMENT: "()",
            PermissionPolicyDirective.PICTURE_IN_PICTURE: "()",
            PermissionPolicyDirective.USB: "()",
            PermissionPolicyDirective.VR: "()",
            PermissionPolicyDirective.WAKE_LOCK: "()",
        }
        
        for directive, value in default_policies.items():
            policies.append(f"{directive.value}={value}")
        
        return ", ".join(policies)
    
    def _generate_feature_policy(self) -> str:
        """Generate Feature-Policy header (deprecated but included for compatibility)."""
        policies = []
        
        default_policies = {
            FeaturePolicyDirective.ACCELEROMETER: "'none'",
            FeaturePolicyDirective.AMBIENT_LIGHT_SENSOR: "'none'",
            FeaturePolicyDirective.AUTOPLAY: "'none'",
            FeaturePolicyDirective.CAMERA: "'none'",
            FeaturePolicyDirective.ENCRYPTED_MEDIA: "'none'",
            FeaturePolicyDirective.FULLSCREEN: "'self'",
            FeaturePolicyDirective.GEOLOCATION: "'none'",
            FeaturePolicyDirective.GYROSCOPE: "'none'",
            FeaturePolicyDirective.MAGNETOMETER: "'none'",
            FeaturePolicyDirective.MICROPHONE: "'none'",
            FeaturePolicyDirective.MIDI: "'none'",
            FeaturePolicyDirective.PAYMENT: "'none'",
            FeaturePolicyDirective.PICTURE_IN_PICTURE: "'none'",
            FeaturePolicyDirective.SPEAKER: "'self'",
            FeaturePolicyDirective.USB: "'none'",
            FeaturePolicyDirective.VR: "'none'",
            FeaturePolicyDirective.WAKE_LOCK: "'none'",
        }
        
        for directive, value in default_policies.items():
            policies.append(f"{directive.value} {value}")
        
        return "; ".join(policies)
    
    def add_security_headers_to_response(self, response: Response, request: Optional[Request] = None):
        """Add security headers to FastAPI response."""
        headers = self.generate_headers(request)
        
        for header, value in headers.items():
            response.headers[header] = value
        
        return response
    
    def validate_headers(self, headers: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate security headers.
        
        Args:
            headers: Dictionary of headers to validate
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        # Check for missing critical headers
        critical_headers = [
            "Content-Security-Policy",
            "X-Frame-Options",
            "X-Content-Type-Options",
            "Referrer-Policy",
        ]
        
        for header in critical_headers:
            if header not in headers:
                warnings.append(f"Missing critical security header: {header}")
        
        # Validate specific headers
        if "Content-Security-Policy" in headers:
            csp_valid, csp_errors = self.csp.validate_csp(headers["Content-Security-Policy"])
            if not csp_valid:
                warnings.extend(csp_errors)
        
        if "X-Frame-Options" in headers:
            xfo_value = headers["X-Frame-Options"]
            if xfo_value not in ["DENY", "SAMEORIGIN"] and not xfo_value.startswith("ALLOW-FROM"):
                warnings.append(f"Invalid X-Frame-Options value: {xfo_value}")
        
        if "Strict-Transport-Security" in headers:
            hsts_value = headers["Strict-Transport-Security"]
            if "max-age=0" in hsts_value:
                warnings.append("HSTS max-age should not be 0 in production")
        
        # Check for insecure headers
        insecure_headers = ["Public-Key-Pins", "X-Powered-By", "X-AspNet-Version", "X-AspNetMvc-Version"]
        for header in insecure_headers:
            if header in headers:
                warnings.append(f"Insecure header present: {header}")
        
        return len(warnings) == 0, warnings
    
    def test_header_security(self, url: str) -> Dict[str, Any]:
        """
        Test security headers for a given URL (for security audits).
        
        Args:
            url: URL to test
            
        Returns:
            Dictionary with test results
        """
        # This would typically make an HTTP request and analyze headers
        # For now, return a template response
        return {
            "url": url,
            "headers_found": [],
            "missing_headers": [],
            "warnings": [],
            "score": 0,
            "grade": "F",
        }


class XSSProtection:
    """XSS protection utilities."""
    
    @staticmethod
    def sanitize_html(html: str) -> str:
        """
        Sanitize HTML to prevent XSS.
        
        Args:
            html: HTML string to sanitize
            
        Returns:
            Sanitized HTML
        """
        import html
        
        # Basic HTML escaping
        sanitized = html.escape(html)
        
        # Remove script tags and event handlers
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'on\w+="[^"]*"', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"on\w+='[^']*'", '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'on\w+=\w+', '', sanitized, flags=re.IGNORECASE)
        
        # Remove dangerous attributes
        dangerous_attrs = [
            'href="javascript:', 'src="javascript:', 'style="', 'formaction="',
            'poster="javascript:', 'background="javascript:', 'data="'
        ]
        
        for attr in dangerous_attrs:
            sanitized = sanitized.replace(attr, attr.replace('javascript:', ''))
        
        return sanitized
    
    @staticmethod
    def validate_input(input_string: str, input_type: str = "text") -> Tuple[bool, str]:
        """
        Validate input for XSS prevention.
        
        Args:
            input_string: Input to validate
            input_type: Type of input (text, html, url, email, etc.)
            
        Returns:
            Tuple of (is_valid, sanitized_string)
        """
        if not input_string:
            return True, ""
        
        sanitized = input_string
        
        if input_type == "html":
            sanitized = XSSProtection.sanitize_html(input_string)
            is_valid = sanitized == input_string
        elif input_type == "url":
            # Validate URL
            try:
                result = urlparse(input_string)
                is_valid = all([result.scheme, result.netloc])
                # Ensure safe scheme
                is_valid = is_valid and result.scheme in ["http", "https", "ftp", "ftps"]
            except:
                is_valid = False
        elif input_type == "email":
            # Simple email validation
            email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
            is_valid = bool(email_pattern.match(input_string))
        else:
            # Text input - basic XSS prevention
            dangerous_patterns = [
                r'<script', r'javascript:', r'on\w+=', r'vbscript:', r'expression\(',
                r'url\(', r'eval\(', r'exec\(', r'fromCharCode', r'alert\(', r'prompt\(',
                r'confirm\(', r'document\.', r'window\.', r'localStorage', r'sessionStorage'
            ]
            
            is_valid = True
            for pattern in dangerous_patterns:
                if re.search(pattern, input_string, re.IGNORECASE):
                    is_valid = False
                    break
        
        return is_valid, sanitized if is_valid else ""


class FrameOptions:
    """Frame options for clickjacking protection."""
    
    DENY = "DENY"
    SAMEORIGIN = "SAMEORIGIN"
    
    @staticmethod
    def allow_from(uri: str) -> str:
        """Generate ALLOW-FROM directive."""
        return f"ALLOW-FROM {uri}"
    
    @staticmethod
    def validate_frame_options(option: str) -> bool:
        """Validate frame option value."""
        valid_options = ["DENY", "SAMEORIGIN"]
        return option in valid_options or option.startswith("ALLOW-FROM ")


class HSTS:
    """HTTP Strict Transport Security utilities."""
    
    @staticmethod
    def generate_header(
        max_age: int = 31536000,
        include_subdomains: bool = True,
        preload: bool = False
    ) -> str:
        """Generate HSTS header."""
        header = f"max-age={max_age}"
        if include_subdomains:
            header += "; includeSubDomains"
        if preload:
            header += "; preload"
        return header
    
    @staticmethod
    def should_enable_hsts(request: Request) -> bool:
        """Determine if HSTS should be enabled for this request."""
        # Only enable HSTS for HTTPS requests
        return request.url.scheme == "https"


class ReferrerPolicy:
    """Referrer policy utilities."""
    
    # Valid referrer policies
    NO_REFERRER = "no-referrer"
    NO_REFERRER_WHEN_DOWNGRADE = "no-referrer-when-downgrade"
    ORIGIN = "origin"
    ORIGIN_WHEN_CROSS_ORIGIN = "origin-when-cross-origin"
    UNSAFE_URL = "unsafe-url"
    STRICT_ORIGIN = "strict-origin"
    STRICT_ORIGIN_WHEN_CROSS_ORIGIN = "strict-origin-when-cross-origin"
    
    @staticmethod
    def get_recommended_policy() -> str:
        """Get recommended referrer policy."""
        return ReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN


# FastAPI Middleware
class SecurityHeadersMiddleware:
    """FastAPI middleware for adding security headers."""
    
    def __init__(
        self,
        app,
        security_policy: Optional[SecurityPolicy] = None,
        enabled: bool = True
    ):
        self.app = app
        self.security_policy = security_policy or SecurityPolicy()
        self.enabled = enabled
        self.security_headers = SecurityHeaders(security_policy)
        self.audit_logger = AuditLogger()
    
    async def __call__(self, request: Request, call_next):
        """Process request and add security headers."""
        response = await call_next(request)
        
        if self.enabled:
            # Add security headers
            response = self.security_headers.add_security_headers_to_response(response, request)
            
            # Log if headers are missing or misconfigured
            if settings.ENVIRONMENT == "development":
                is_valid, warnings = self.security_headers.validate_headers(dict(response.headers))
                if warnings:
                    print(f"Security header warnings: {warnings}")
        
        return response


# CSP Violation Reporting Endpoint
async def csp_violation_report(request: Request):
    """
    Endpoint to receive CSP violation reports.
    
    This should be added to your API routes.
    """
    try:
        report_data = await request.json()
        
        # Log the violation
        audit_logger = AuditLogger()
        audit_logger.log_security_event(
            event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
            description="CSP Violation Reported",
            details={
                "report": report_data,
                "path": request.url.path,
                "user_agent": request.headers.get("User-Agent"),
                "ip_address": request.client.host if request.client else None,
            },
            severity="WARNING"
        )
        
        # You could also store in database or send to monitoring service
        return {"status": "received"}
    
    except Exception as e:
        print(f"Error processing CSP violation report: {e}")
        return {"status": "error", "message": str(e)}


# Configuration helpers
def get_default_security_policy(environment: str = "production") -> SecurityPolicy:
    """Get default security policy for environment."""
    policy = SecurityPolicy()
    
    if environment == "development":
        # More relaxed policy for development
        policy.csp_report_only = True
        policy.csp_report_uri = "/api/v1/security/csp-report"
        policy.hsts_max_age = 3600  # 1 hour for dev
        policy.hsts_preload = False
        policy.enable_expect_ct = False
        
        # Allow more sources for dev
        csp = ContentSecurityPolicy(policy)
        csp.add_source("script-src", "'unsafe-inline'")
        csp.add_source("script-src", "'unsafe-eval'")
        csp.add_source("style-src", "'unsafe-inline'")
        csp.add_source("connect-src", "ws://localhost:*")
        csp.add_source("connect-src", "http://localhost:*")
    
    elif environment == "testing":
        # Testing environment
        policy.csp_report_only = True
        policy.csp_report_uri = "/api/v1/security/csp-report"
    
    else:  # production
        # Strict policy for production
        policy.csp_report_only = False
        policy.csp_upgrade_insecure_requests = True
        policy.csp_block_all_mixed_content = True
        policy.hsts_max_age = 31536000  # 1 year
        policy.hsts_include_subdomains = True
        policy.hsts_preload = True
        policy.enable_expect_ct = True
        
        # Add SRI requirement
        policy.csp_require_sri_for = ["script", "style"]
    
    return policy


def setup_security_headers(app: FastAPI, policy: Optional[SecurityPolicy] = None):
    """
    Setup security headers for FastAPI application.
    
    Args:
        app: FastAPI application
        policy: Security policy configuration
    """
    if policy is None:
        policy = get_default_security_policy(settings.ENVIRONMENT)
    
    # Add middleware
    app.add_middleware(SecurityHeadersMiddleware, security_policy=policy)
    
    # Add CSP violation reporting endpoint
    @app.post("/api/v1/security/csp-report", include_in_schema=False)
    async def report_csp_violation(request: Request):
        return await csp_violation_report(request)
    
    # Add security headers test endpoint
    @app.get("/api/v1/security/test-headers", include_in_schema=False)
    async def test_security_headers(request: Request):
        """Test endpoint to check security headers."""
        security_headers = SecurityHeaders(policy)
        headers = security_headers.generate_headers(request)
        is_valid, warnings = security_headers.validate_headers(headers)
        
        return {
            "headers": headers,
            "valid": is_valid,
            "warnings": warnings,
            "grade": "A" if is_valid and not warnings else "F",
        }
    
    print(f"Security headers configured for {settings.ENVIRONMENT} environment")


# Export main components
__all__ = [
    # Classes
    "ContentSecurityPolicy",
    "SecurityHeaders",
    "SecurityPolicy",
    "CSPDirective",
    "XSSProtection",
    "FrameOptions",
    "HSTS",
    "ReferrerPolicy",
    "SecurityHeadersMiddleware",
    
    # Enums
    "DirectiveSource",
    "FeaturePolicyDirective",
    "PermissionPolicyDirective",
    "ReportToGroup",
    
    # Functions
    "get_default_security_policy",
    "setup_security_headers",
    "csp_violation_report",
    
    # Constants
    "ReferrerPolicy",
]