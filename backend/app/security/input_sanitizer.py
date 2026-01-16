"""
Input Sanitization and Validation System

This module provides comprehensive input sanitization and validation:
- SQL injection prevention
- XSS (Cross-Site Scripting) prevention
- Path traversal prevention
- Command injection prevention
- Input validation with schemas
- Data normalization and cleaning
- Malicious pattern detection
- Content security validation
"""

import json
import re
import html
import urllib.parse
import base64
import mimetypes
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union
from urllib.parse import urlparse, quote, unquote

import bleach
import jsonschema
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field, ValidationError, validator
from pydantic.networks import HttpUrl, IPvAnyAddress, EmailStr
from pydantic.types import conint, constr

from app.core.config import get_settings
from app.security.audit_logger import AuditLogger, AuditEventType, AuditSeverity

# Get settings
settings = get_settings()


class SanitizationLevel(str, Enum):
    """Levels of sanitization strictness."""
    MINIMAL = "minimal"      # Basic cleaning, allow most content
    MODERATE = "moderate"    # Remove unsafe content, keep safe HTML
    STRICT = "strict"        # Remove all HTML, only plain text
    PARANOID = "paranoid"    # Maximum security, aggressive filtering


class ValidationType(str, Enum):
    """Types of input validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    DATE = "date"
    DATETIME = "datetime"
    IP_ADDRESS = "ip_address"
    UUID = "uuid"
    JSON = "json"
    XML = "xml"
    HTML = "html"
    MARKDOWN = "markdown"
    FILE_PATH = "file_path"
    BASE64 = "base64"
    HEX = "hex"
    CREDIT_CARD = "credit_card"
    PASSWORD = "password"
    ZIP_CODE = "zip_code"
    COUNTRY_CODE = "country_code"
    CURRENCY = "currency"


class ThreatLevel(str, Enum):
    """Threat level for detected malicious input."""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    CRITICAL = "critical"


@dataclass
class SanitizationResult:
    """Result of sanitization operation."""
    sanitized_value: Any
    original_value: Any
    is_valid: bool
    threat_level: ThreatLevel
    warnings: List[str] = field(default_factory=list)
    removed_content: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """Validation rule definition."""
    validation_type: ValidationType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    error_message: Optional[str] = None
    sanitize: bool = True
    sanitization_level: SanitizationLevel = SanitizationLevel.MODERATE


class MaliciousPatternDetector:
    """Detects malicious patterns in input."""
    
    def __init__(self):
        # SQL Injection patterns
        self.sql_patterns = [
            # Basic SQL injection
            r"(?i)\b(union|select|insert|update|delete|drop|alter|create|truncate)\b.*\b(from|into|table|database|where)\b",
            r"(?i)\b(exec|execute|sp_executesql|xp_cmdshell)\b",
            r"(?i)\b(OR|AND)\s+['\"]?\d+['\"]?\s*[=<>]\s*['\"]?\d+['\"]?",
            r"(?i);\s*(--|#)",
            r"(?i)\b(SLEEP|WAITFOR|BENCHMARK)\b.*\(.*\)",
            r"(?i)\b(LOAD_FILE|INTO\s+(OUTFILE|DUMPFILE))\b",
            r"(?i)\b(LIKE|BETWEEN|IN)\b.*['\"].*%.*['\"]",
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"(?i)<script[^>]*>.*?</script>",
            r"(?i)javascript:",
            r"(?i)vbscript:",
            r"(?i)on\w+\s*=",
            r"(?i)expression\s*\(.*\)",
            r"(?i)url\s*\(.*\)",
            r"(?i)eval\s*\(.*\)",
            r"(?i)alert\s*\(.*\)",
            r"(?i)prompt\s*\(.*\)",
            r"(?i)confirm\s*\(.*\)",
            r"(?i)document\.(cookie|write|location)",
            r"(?i)window\.(location|open|alert)",
            r"(?i)localStorage|sessionStorage",
            r"(?i)<iframe[^>]*>.*?</iframe>",
            r"(?i)<embed[^>]*>.*?</embed>",
            r"(?i)<object[^>]*>.*?</object>",
            r"(?i)<applet[^>]*>.*?</applet>",
            r"(?i)<meta[^>]*>.*?</meta>",
            r"(?i)<link[^>]*>.*?</link>",
            r"(?i)<base[^>]*>.*?</base>",
        ]
        
        # Command injection patterns
        self.command_patterns = [
            r"(?i)[;&|`]\s*(ls|cat|rm|mkdir|chmod|chown|wget|curl|nc|netcat|python|perl|ruby|bash|sh|cmd|powershell)",
            r"(?i)\$(?:\{|\().*?(?:\}|\))",
            r"(?i)\b(exec|system|popen|shell_exec|passthru|proc_open)\b.*\(.*\)",
            r"(?i)\b(eval|assert|create_function)\b.*\(.*\)",
            r"(?i)\b(fopen|fwrite|file_put_contents|unlink|rename)\b.*\(.*\)",
            r"(?i)\b(include|require|include_once|require_once)\b.*\(.*\)",
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"(?i)\.\.(/|\\|%2f|%5c)",
            r"(?i)(/|\\|~)(etc|passwd|shadow|hosts|ssh)",
            r"(?i)(c:|d:|e:)(/|\\).*\.(exe|bat|cmd|vbs|ps1|sh)",
            r"(?i)/proc/.*/.*",
            r"(?i)/sys/.*/.*",
            r"(?i)/dev/.*",
            r"(?i)/bin/.*",
            r"(?i)/sbin/.*",
            r"(?i)/usr/.*",
            r"(?i)/var/.*",
            r"(?i)/tmp/.*",
            r"(?i)/home/.*",
            r"(?i)/root/.*",
        ]
        
        # File inclusion patterns
        self.file_inclusion_patterns = [
            r"(?i)\.\./\.\./",
            r"(?i)\.\.\\\.\.\\",
            r"(?i)\.\.%2f\.\.%2f",
            r"(?i)\.\.%5c\.\.%5c",
            r"(?i)\.\.%252f\.\.%252f",
            r"(?i)\.\.%255c\.\.%255c",
            r"(?i)(ftp|http|https|file|php|data):",
            r"(?i)zip://|phar://|data://",
            r"(?i)expect://|ssh2://",
        ]
        
        # SSRF patterns
        self.ssrf_patterns = [
            r"(?i)://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|\[::\])",
            r"(?i)://(169\.254\.\d+\.\d+|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+)",
            r"(?i)://(metadata\.google\.internal|169\.254\.169\.254)",
            r"(?i)://(instance-data|169\.254\.169\.254)",
            r"(?i)file://.*",
            r"(?i)gopher://.*",
            r"(?i)dict://.*",
            r"(?i)ldap://.*",
            r"(?i)ldaps://.*",
        ]
        
        # Malicious file patterns
        self.malicious_file_patterns = [
            r"(?i)\.(php|asp|aspx|jsp|cgi|pl|sh|exe|bat|cmd|vbs|ps1|jar|war|ear|sh|py|rb|pl|pm|tcl|lua)$",
            r"(?i)\.(phtml|pht|phps|phar|inc)$",
            r"(?i)\.(htaccess|htpasswd|ini|conf|cfg|config|yml|yaml|json|xml)$",
            r"(?i)\.(sql|db|dbf|mdb|accdb|sqlite)$",
            r"(?i)\.(log|txt|md|rst|doc|docx|xls|xlsx|ppt|pptx|pdf)$",
        ]
        
        # Compile all patterns
        self.all_patterns = {
            "sql_injection": [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.sql_patterns],
            "xss": [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.xss_patterns],
            "command_injection": [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.command_patterns],
            "path_traversal": [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.path_traversal_patterns],
            "file_inclusion": [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.file_inclusion_patterns],
            "ssrf": [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.ssrf_patterns],
            "malicious_file": [re.compile(p, re.IGNORECASE) for p in self.malicious_file_patterns],
        }
    
    def detect(self, input_data: Any, input_type: str = "text") -> Tuple[ThreatLevel, Dict[str, List[str]]]:
        """
        Detect malicious patterns in input.
        
        Args:
            input_data: Input to check
            input_type: Type of input (text, url, file, etc.)
            
        Returns:
            Tuple of (threat_level, detected_patterns)
        """
        if input_data is None:
            return ThreatLevel.SAFE, {}
        
        input_str = str(input_data)
        detected = {}
        threat_score = 0
        
        # Check patterns based on input type
        if input_type in ["text", "html", "markdown", "json", "xml"]:
            threat_score += self._check_patterns(input_str, "sql_injection", detected) * 10
            threat_score += self._check_patterns(input_str, "xss", detected) * 8
            threat_score += self._check_patterns(input_str, "command_injection", detected) * 10
        
        if input_type in ["file_path", "url", "path"]:
            threat_score += self._check_patterns(input_str, "path_traversal", detected) * 9
            threat_score += self._check_patterns(input_str, "file_inclusion", detected) * 8
            threat_score += self._check_patterns(input_str, "ssrf", detected) * 7
        
        if input_type in ["file", "file_path", "filename"]:
            threat_score += self._check_patterns(input_str, "malicious_file", detected) * 5
        
        # Determine threat level
        if threat_score >= 15:
            threat_level = ThreatLevel.CRITICAL
        elif threat_score >= 10:
            threat_level = ThreatLevel.MALICIOUS
        elif threat_score >= 5:
            threat_level = ThreatLevel.SUSPICIOUS
        else:
            threat_level = ThreatLevel.SAFE
        
        return threat_level, detected
    
    def _check_patterns(self, input_str: str, pattern_type: str, detected: Dict[str, List[str]]) -> int:
        """Check input against specific pattern type."""
        count = 0
        detected[pattern_type] = []
        
        for pattern in self.all_patterns[pattern_type]:
            matches = pattern.findall(input_str)
            if matches:
                detected[pattern_type].extend(matches)
                count += len(matches)
        
        return count


class InputSanitizer:
    """
    Main input sanitization and validation class.
    """
    
    def __init__(
        self,
        sanitization_level: SanitizationLevel = SanitizationLevel.MODERATE,
        enable_audit_logging: bool = True,
        max_input_length: int = 10000,
        custom_validators: Optional[Dict[str, Callable]] = None
    ):
        self.sanitization_level = sanitization_level
        self.enable_audit_logging = enable_audit_logging
        self.max_input_length = max_input_length
        self.detector = MaliciousPatternDetector()
        self.audit_logger = AuditLogger() if enable_audit_logging else None
        
        # Allowed HTML tags and attributes (for moderate level)
        self.allowed_tags = {
            "a", "abbr", "acronym", "b", "blockquote", "code", "em", "i", "li", "ol",
            "strong", "ul", "p", "br", "span", "div", "h1", "h2", "h3", "h4", "h5", "h6",
            "table", "thead", "tbody", "tr", "th", "td", "pre", "hr", "img"
        }
        
        self.allowed_attributes = {
            "a": ["href", "title", "target", "rel"],
            "img": ["src", "alt", "title", "width", "height"],
            "span": ["class", "style"],
            "div": ["class", "style"],
            "p": ["class", "style"],
            "table": ["class", "style", "border", "cellpadding", "cellspacing"],
            "th": ["class", "style", "colspan", "rowspan"],
            "td": ["class", "style", "colspan", "rowspan"],
        }
        
        # Custom validators
        self.custom_validators = custom_validators or {}
        self._init_default_validators()
        
        # Validation schemas cache
        self.validation_schemas: Dict[str, Dict[str, ValidationRule]] = {}
    
    def _init_default_validators(self):
        """Initialize default custom validators."""
        self.custom_validators.update({
            "phone": self._validate_phone,
            "zip_code": self._validate_zip_code,
            "country_code": self._validate_country_code,
            "currency": self._validate_currency,
            "credit_card": self._validate_credit_card,
            "password": self._validate_password,
        })
    
    def sanitize(
        self,
        input_data: Any,
        validation_type: Optional[ValidationType] = None,
        field_name: Optional[str] = None,
        request: Optional[Request] = None,
        **kwargs
    ) -> SanitizationResult:
        """
        Sanitize input data.
        
        Args:
            input_data: Input data to sanitize
            validation_type: Type of validation to apply
            field_name: Name of the field (for logging)
            request: FastAPI request (for context)
            **kwargs: Additional sanitization options
            
        Returns:
            SanitizationResult object
        """
        original_value = input_data
        
        # Handle None/empty input
        if input_data is None:
            return SanitizationResult(
                sanitized_value=None,
                original_value=None,
                is_valid=True,
                threat_level=ThreatLevel.SAFE
            )
        
        # Convert to string for pattern detection
        input_str = str(input_data)
        
        # Check max length
        if len(input_str) > self.max_input_length:
            return SanitizationResult(
                sanitized_value=input_str[:self.max_input_length],
                original_value=original_value,
                is_valid=False,
                threat_level=ThreatLevel.SUSPICIOUS,
                warnings=[f"Input truncated: exceeds max length of {self.max_input_length}"],
                validation_errors=["Input too long"]
            )
        
        # Detect malicious patterns
        threat_level, detected_patterns = self.detector.detect(
            input_str,
            validation_type.value if validation_type else "text"
        )
        
        # Apply sanitization based on type
        if validation_type:
            sanitized_value, is_valid, warnings, errors = self._sanitize_by_type(
                input_data, validation_type, **kwargs
            )
        else:
            sanitized_value, is_valid, warnings, errors = self._sanitize_generic(input_str, **kwargs)
        
        # Log security events if malicious patterns detected
        if self.enable_audit_logging and self.audit_logger and threat_level != ThreatLevel.SAFE:
            self._log_sanitization_event(
                field_name=field_name,
                original_value=original_value,
                sanitized_value=sanitized_value,
                threat_level=threat_level,
                detected_patterns=detected_patterns,
                request=request
            )
        
        # Create result
        result = SanitizationResult(
            sanitized_value=sanitized_value,
            original_value=original_value,
            is_valid=is_valid and threat_level != ThreatLevel.CRITICAL,
            threat_level=threat_level,
            warnings=warnings,
            validation_errors=errors,
            metadata={
                "detected_patterns": detected_patterns,
                "validation_type": validation_type.value if validation_type else None,
                "field_name": field_name,
            }
        )
        
        # Remove malicious content if critical
        if threat_level == ThreatLevel.CRITICAL and not is_valid:
            result.sanitized_value = self._sanitize_critical(input_str)
            result.removed_content = self._extract_malicious_content(input_str, detected_patterns)
        
        return result
    
    def _sanitize_by_type(
        self,
        input_data: Any,
        validation_type: ValidationType,
        **kwargs
    ) -> Tuple[Any, bool, List[str], List[str]]:
        """Sanitize input based on validation type."""
        sanitization_level = kwargs.get('sanitization_level', self.sanitization_level)
        
        if validation_type == ValidationType.STRING:
            return self._sanitize_string(input_data, sanitization_level)
        elif validation_type == ValidationType.INTEGER:
            return self._sanitize_integer(input_data)
        elif validation_type == ValidationType.FLOAT:
            return self._sanitize_float(input_data)
        elif validation_type == ValidationType.BOOLEAN:
            return self._sanitize_boolean(input_data)
        elif validation_type == ValidationType.EMAIL:
            return self._sanitize_email(input_data)
        elif validation_type == ValidationType.URL:
            return self._sanitize_url(input_data)
        elif validation_type == ValidationType.HTML:
            return self._sanitize_html(input_data, sanitization_level)
        elif validation_type == ValidationType.JSON:
            return self._sanitize_json(input_data)
        elif validation_type == ValidationType.XML:
            return self._sanitize_xml(input_data)
        elif validation_type == ValidationType.MARKDOWN:
            return self._sanitize_markdown(input_data, sanitization_level)
        elif validation_type == ValidationType.FILE_PATH:
            return self._sanitize_file_path(input_data)
        elif validation_type == ValidationType.BASE64:
            return self._sanitize_base64(input_data)
        elif validation_type == ValidationType.IP_ADDRESS:
            return self._sanitize_ip_address(input_data)
        elif validation_type == ValidationType.UUID:
            return self._sanitize_uuid(input_data)
        elif validation_type == ValidationType.DATE:
            return self._sanitize_date(input_data)
        elif validation_type == ValidationType.DATETIME:
            return self._sanitize_datetime(input_data)
        else:
            # Use custom validator if available
            if validation_type.value in self.custom_validators:
                validator_func = self.custom_validators[validation_type.value]
                return validator_func(input_data)
            else:
                return self._sanitize_generic(str(input_data))
    
    def _sanitize_string(self, input_data: Any, level: SanitizationLevel) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize string input."""
        input_str = str(input_data)
        warnings = []
        errors = []
        
        # Apply sanitization based on level
        if level == SanitizationLevel.PARANOID:
            # Remove all non-alphanumeric characters
            sanitized = re.sub(r'[^a-zA-Z0-9\s\-_\.@]', '', input_str)
            if sanitized != input_str:
                warnings.append("Non-alphanumeric characters removed")
        elif level == SanitizationLevel.STRICT:
            # Remove HTML tags and special characters
            sanitized = html.escape(input_str)
            sanitized = re.sub(r'[<>"\']', '', sanitized)
        elif level == SanitizationLevel.MODERATE:
            # Allow safe HTML
            sanitized = bleach.clean(
                input_str,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )
        else:  # MINIMAL
            # Just basic cleaning
            sanitized = input_str.strip()
            sanitized = sanitized.replace('\x00', '')  # Remove null bytes
        
        is_valid = bool(sanitized.strip()) if input_str.strip() else True
        
        return sanitized, is_valid, warnings, errors
    
    def _sanitize_html(self, input_data: Any, level: SanitizationLevel) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize HTML input."""
        input_str = str(input_data)
        warnings = []
        errors = []
        
        if level in [SanitizationLevel.PARANOID, SanitizationLevel.STRICT]:
            # Convert HTML to plain text
            sanitized = bleach.clean(input_str, tags=[], attributes={}, strip=True)
            warnings.append("HTML tags removed")
        else:
            # Allow safe HTML
            sanitized = bleach.clean(
                input_str,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )
        
        is_valid = True
        return sanitized, is_valid, warnings, errors
    
    def _sanitize_url(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize URL input."""
        input_str = str(input_data)
        warnings = []
        errors = []
        
        try:
            # Parse URL
            parsed = urlparse(input_str)
            
            # Validate scheme
            if parsed.scheme not in ['http', 'https', 'ftp', 'ftps', 'mailto', 'tel']:
                errors.append(f"Unsupported URL scheme: {parsed.scheme}")
                return input_str, False, warnings, errors
            
            # Validate hostname
            if not parsed.netloc:
                errors.append("Invalid URL: missing hostname")
                return input_str, False, warnings, errors
            
            # Reconstruct sanitized URL
            sanitized = parsed.geturl()
            
            # URL encode if needed
            if parsed.query:
                sanitized = self._sanitize_query_string(sanitized)
            
            is_valid = True
            
        except Exception as e:
            errors.append(f"Invalid URL: {str(e)}")
            return input_str, False, warnings, errors
        
        return sanitized, is_valid, warnings, errors
    
    def _sanitize_query_string(self, url: str) -> str:
        """Sanitize URL query string."""
        try:
            parsed = urlparse(url)
            if parsed.query:
                # Parse query parameters
                query_params = urllib.parse.parse_qs(parsed.query)
                
                # Sanitize each parameter
                sanitized_params = {}
                for key, values in query_params.items():
                    sanitized_values = []
                    for value in values:
                        sanitized_value = html.escape(value)
                        sanitized_value = re.sub(r'[<>"\']', '', sanitized_value)
                        sanitized_values.append(sanitized_value)
                    sanitized_params[key] = sanitized_values
                
                # Rebuild query string
                sanitized_query = urllib.parse.urlencode(sanitized_params, doseq=True)
                
                # Reconstruct URL
                sanitized_url = urllib.parse.urlunparse((
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.params,
                    sanitized_query,
                    parsed.fragment
                ))
                
                return sanitized_url
        except:
            pass
        
        return url
    
    def _sanitize_json(self, input_data: Any) -> Tuple[Any, bool, List[str], List[str]]:
        """Sanitize JSON input."""
        warnings = []
        errors = []
        
        if isinstance(input_data, (dict, list)):
            # Already parsed JSON
            data = input_data
        else:
            try:
                # Parse JSON string
                data = json.loads(str(input_data))
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON: {str(e)}")
                return input_data, False, warnings, errors
        
        # Recursively sanitize JSON data
        sanitized = self._sanitize_json_recursive(data)
        
        is_valid = True
        return sanitized, is_valid, warnings, errors
    
    def _sanitize_json_recursive(self, data: Any) -> Any:
        """Recursively sanitize JSON data."""
        if isinstance(data, dict):
            return {k: self._sanitize_json_recursive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_json_recursive(item) for item in data]
        elif isinstance(data, str):
            # Sanitize strings in JSON
            return html.escape(data)
        else:
            return data
    
    def _sanitize_file_path(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize file path input."""
        input_str = str(input_data)
        warnings = []
        errors = []
        
        # Normalize path
        path = Path(input_str).resolve()
        
        # Check for path traversal
        if '..' in str(path):
            errors.append("Path traversal detected")
            return input_str, False, warnings, errors
        
        # Check for unsafe characters
        unsafe_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in unsafe_chars:
            if char in input_str:
                errors.append(f"Unsafe character in path: {char}")
                return input_str, False, warnings, errors
        
        # Convert to string
        sanitized = str(path)
        
        is_valid = True
        return sanitized, is_valid, warnings, errors
    
    def _sanitize_base64(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize base64 input."""
        input_str = str(input_data)
        warnings = []
        errors = []
        
        try:
            # Validate base64
            decoded = base64.b64decode(input_str)
            
            # Check for binary content
            # You might want to add more specific checks here
            
            # Re-encode to ensure valid base64
            sanitized = base64.b64encode(decoded).decode('utf-8')
            
            is_valid = True
            
        except Exception as e:
            errors.append(f"Invalid base64: {str(e)}")
            return input_str, False, warnings, errors
        
        return sanitized, is_valid, warnings, errors
    
    def _sanitize_integer(self, input_data: Any) -> Tuple[int, bool, List[str], List[str]]:
        """Sanitize integer input."""
        try:
            value = int(input_data)
            is_valid = True
            return value, is_valid, [], []
        except (ValueError, TypeError):
            return 0, False, [], ["Invalid integer"]
    
    def _sanitize_float(self, input_data: Any) -> Tuple[float, bool, List[str], List[str]]:
        """Sanitize float input."""
        try:
            value = float(input_data)
            is_valid = True
            return value, is_valid, [], []
        except (ValueError, TypeError):
            return 0.0, False, [], ["Invalid float"]
    
    def _sanitize_boolean(self, input_data: Any) -> Tuple[bool, bool, List[str], List[str]]:
        """Sanitize boolean input."""
        if isinstance(input_data, bool):
            return input_data, True, [], []
        
        input_str = str(input_data).lower().strip()
        
        true_values = ['true', '1', 'yes', 'on', 'y']
        false_values = ['false', '0', 'no', 'off', 'n']
        
        if input_str in true_values:
            return True, True, [], []
        elif input_str in false_values:
            return False, True, [], []
        else:
            return False, False, [], ["Invalid boolean value"]
    
    def _sanitize_email(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize email input."""
        input_str = str(input_data).strip()
        
        # Simple email validation regex
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if re.match(email_regex, input_str):
            # Sanitize by escaping special characters
            sanitized = html.escape(input_str)
            return sanitized, True, [], []
        else:
            return input_str, False, [], ["Invalid email format"]
    
    def _sanitize_ip_address(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize IP address input."""
        input_str = str(input_data).strip()
        
        # IP address regex (IPv4 and IPv6)
        ipv4_regex = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        ipv6_regex = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        
        if re.match(ipv4_regex, input_str) or re.match(ipv6_regex, input_str):
            return input_str, True, [], []
        else:
            return input_str, False, [], ["Invalid IP address"]
    
    def _sanitize_uuid(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize UUID input."""
        input_str = str(input_data).strip()
        
        # UUID regex (version 1-5)
        uuid_regex = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$'
        
        if re.match(uuid_regex, input_str):
            return input_str.lower(), True, [], []
        else:
            return input_str, False, [], ["Invalid UUID"]
    
    def _sanitize_date(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize date input."""
        # This is a simplified version - in production, use dateutil or similar
        input_str = str(input_data).strip()
        
        # Common date formats
        date_formats = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
        ]
        
        for fmt in date_formats:
            if re.match(fmt, input_str):
                return input_str, True, [], []
        
        return input_str, False, [], ["Invalid date format"]
    
    def _sanitize_datetime(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize datetime input."""
        # Simplified - in production, use proper datetime parsing
        input_str = str(input_data).strip()
        
        # ISO 8601 format
        iso_regex = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$'
        
        if re.match(iso_regex, input_str):
            return input_str, True, [], []
        
        return input_str, False, [], ["Invalid datetime format"]
    
    def _sanitize_xml(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize XML input."""
        input_str = str(input_data)
        warnings = []
        errors = []
        
        # Remove potentially dangerous tags
        dangerous_tags = ['<!DOCTYPE', '<?xml', '<!ENTITY']
        for tag in dangerous_tags:
            if tag in input_str.upper():
                errors.append(f"Dangerous XML tag found: {tag}")
                return input_str, False, warnings, errors
        
        # Basic XML structure check
        if not (input_str.strip().startswith('<') and input_str.strip().endswith('>')):
            errors.append("Invalid XML structure")
            return input_str, False, warnings, errors
        
        # Escape special characters
        sanitized = html.escape(input_str)
        
        return sanitized, True, warnings, errors
    
    def _sanitize_markdown(self, input_data: Any, level: SanitizationLevel) -> Tuple[str, bool, List[str], List[str]]:
        """Sanitize markdown input."""
        input_str = str(input_data)
        warnings = []
        errors = []
        
        # First sanitize as HTML (markdown can contain HTML)
        html_sanitized = bleach.clean(
            input_str,
            tags=self.allowed_tags if level != SanitizationLevel.STRICT else [],
            attributes=self.allowed_attributes,
            strip=True
        )
        
        # Remove dangerous markdown patterns
        dangerous_md = [
            r'\!\[.*\]\(javascript:.*\)',  # JavaScript in images
            r'\[.*\]\(javascript:.*\)',    # JavaScript in links
            r'`.*`.*\b(eval|exec|system)\b',  # Code execution
        ]
        
        for pattern in dangerous_md:
            html_sanitized = re.sub(pattern, '', html_sanitized, flags=re.IGNORECASE)
        
        return html_sanitized, True, warnings, errors
    
    def _sanitize_generic(self, input_str: str, **kwargs) -> Tuple[str, bool, List[str], List[str]]:
        """Generic sanitization for unknown types."""
        sanitization_level = kwargs.get('sanitization_level', self.sanitization_level)
        
        if sanitization_level in [SanitizationLevel.PARANOID, SanitizationLevel.STRICT]:
            sanitized = html.escape(input_str)
            sanitized = re.sub(r'[<>"\']', '', sanitized)
        else:
            sanitized = input_str
        
        return sanitized, True, [], []
    
    def _sanitize_critical(self, input_str: str) -> str:
        """Aggressive sanitization for critical threats."""
        # Remove all non-alphanumeric characters
        sanitized = re.sub(r'[^a-zA-Z0-9\s\-_\.@]', '', input_str)
        
        # Truncate if still long
        if len(sanitized) > 100:
            sanitized = sanitized[:100] + "..."
        
        return sanitized
    
    def _extract_malicious_content(self, input_str: str, detected_patterns: Dict[str, List[str]]) -> List[str]:
        """Extract malicious content from input."""
        removed = []
        
        for pattern_type, matches in detected_patterns.items():
            for match in matches:
                if match not in removed:
                    removed.append(match)
        
        return removed
    
    def _log_sanitization_event(
        self,
        field_name: Optional[str],
        original_value: Any,
        sanitized_value: Any,
        threat_level: ThreatLevel,
        detected_patterns: Dict[str, List[str]],
        request: Optional[Request]
    ):
        """Log sanitization event for security monitoring."""
        if not self.audit_logger:
            return
        
        description = f"Input sanitization: {threat_level.value} threat detected"
        if field_name:
            description += f" in field '{field_name}'"
        
        details = {
            "field_name": field_name,
            "threat_level": threat_level.value,
            "detected_patterns": detected_patterns,
            "original_value_preview": str(original_value)[:100],
            "sanitized_value_preview": str(sanitized_value)[:100],
        }
        
        if request:
            details.update({
                "path": request.url.path,
                "method": request.method,
                "ip_address": request.client.host if request.client else None,
                "user_agent": request.headers.get("User-Agent"),
            })
        
        self.audit_logger.log_security_event(
            event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
            description=description,
            details=details,
            severity=AuditSeverity.WARNING if threat_level == ThreatLevel.SUSPICIOUS else AuditSeverity.ERROR
        )
    
    # Custom validators
    def _validate_phone(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Validate phone number."""
        input_str = str(input_data).strip()
        
        # International phone number regex (simplified)
        phone_regex = r'^\+?[1-9]\d{1,14}$'
        
        if re.match(phone_regex, input_str):
            return input_str, True, [], []
        else:
            return input_str, False, [], ["Invalid phone number"]
    
    def _validate_zip_code(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Validate ZIP/postal code."""
        input_str = str(input_data).strip()
        
        # US ZIP code regex (5 digits or 5+4)
        us_zip_regex = r'^\d{5}(-\d{4})?$'
        
        if re.match(us_zip_regex, input_str):
            return input_str, True, [], []
        else:
            # Could add other country formats here
            return input_str, False, [], ["Invalid ZIP code"]
    
    def _validate_country_code(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Validate country code (ISO 3166-1 alpha-2)."""
        input_str = str(input_data).strip().upper()
        
        # ISO 3166-1 alpha-2 country codes
        valid_codes = {
            'US', 'CA', 'MX', 'GB', 'FR', 'DE', 'IT', 'ES', 'JP', 'CN',
            'IN', 'BR', 'AU', 'RU', 'KR', 'SA', 'AE', 'SG', 'MY', 'ID',
            # Add more as needed
        }
        
        if input_str in valid_codes:
            return input_str, True, [], []
        else:
            return input_str, False, [], ["Invalid country code"]
    
    def _validate_currency(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Validate currency code (ISO 4217)."""
        input_str = str(input_data).strip().upper()
        
        # Common currency codes
        valid_currencies = {
            'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'HKD',
            'SGD', 'SEK', 'NZD', 'MXN', 'INR', 'RUB', 'ZAR', 'TRY', 'BRL',
        }
        
        if input_str in valid_currencies:
            return input_str, True, [], []
        else:
            return input_str, False, [], ["Invalid currency code"]
    
    def _validate_credit_card(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Validate credit card number (Luhn algorithm)."""
        input_str = str(input_data).replace(' ', '').replace('-', '')
        
        # Check if all digits
        if not input_str.isdigit():
            return input_str, False, [], ["Credit card must contain only digits"]
        
        # Luhn algorithm
        def luhn_checksum(card_number):
            def digits_of(n):
                return [int(d) for d in str(n)]
            digits = digits_of(card_number)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10
        
        if luhn_checksum(input_str) == 0:
            return input_str, True, [], []
        else:
            return input_str, False, [], ["Invalid credit card number"]
    
    def _validate_password(self, input_data: Any) -> Tuple[str, bool, List[str], List[str]]:
        """Validate password strength."""
        input_str = str(input_data)
        errors = []
        warnings = []
        
        # Minimum length
        if len(input_str) < 8:
            errors.append("Password must be at least 8 characters long")
        
        # Check for common patterns
        common_passwords = [
            'password', '123456', 'qwerty', 'admin', 'welcome',
            'password123', '12345678', '123456789', '1234567890'
        ]
        
        if input_str.lower() in common_passwords:
            errors.append("Password is too common")
        
        # Strength requirements
        has_upper = any(c.isupper() for c in input_str)
        has_lower = any(c.islower() for c in input_str)
        has_digit = any(c.isdigit() for c in input_str)
        has_special = any(not c.isalnum() for c in input_str)
        
        if not has_upper:
            warnings.append("Password should contain uppercase letters")
        if not has_lower:
            warnings.append("Password should contain lowercase letters")
        if not has_digit:
            warnings.append("Password should contain digits")
        if not has_special:
            warnings.append("Password should contain special characters")
        
        is_valid = len(errors) == 0
        
        return input_str, is_valid, warnings, errors
    
    # Schema-based validation
    def register_schema(self, schema_name: str, schema: Dict[str, ValidationRule]):
        """Register a validation schema."""
        self.validation_schemas[schema_name] = schema
    
    def validate_with_schema(
        self,
        data: Dict[str, Any],
        schema_name: str,
        request: Optional[Request] = None
    ) -> Dict[str, SanitizationResult]:
        """
        Validate data against a registered schema.
        
        Args:
            data: Data to validate
            schema_name: Name of registered schema
            request: FastAPI request for context
            
        Returns:
            Dictionary of field names to SanitizationResult
        """
        if schema_name not in self.validation_schemas:
            raise ValueError(f"Schema '{schema_name}' not found")
        
        schema = self.validation_schemas[schema_name]
        results = {}
        
        for field_name, rule in schema.items():
            value = data.get(field_name)
            
            if value is None and rule.required:
                results[field_name] = SanitizationResult(
                    sanitized_value=None,
                    original_value=None,
                    is_valid=False,
                    threat_level=ThreatLevel.SAFE,
                    validation_errors=["Field is required"]
                )
            elif value is not None:
                # Apply validation
                result = self.sanitize(
                    value,
                    validation_type=rule.validation_type,
                    field_name=field_name,
                    request=request,
                    sanitization_level=rule.sanitization_level if rule.sanitize else SanitizationLevel.MINIMAL
                )
                
                # Apply additional validation rules
                if result.is_valid:
                    result = self._apply_validation_rules(result, rule)
                
                results[field_name] = result
        
        return results
    
    def _apply_validation_rules(self, result: SanitizationResult, rule: ValidationRule) -> SanitizationResult:
        """Apply additional validation rules to result."""
        value = result.sanitized_value
        
        # Length validation
        if rule.min_length is not None and len(str(value)) < rule.min_length:
            result.is_valid = False
            result.validation_errors.append(f"Minimum length is {rule.min_length}")
        
        if rule.max_length is not None and len(str(value)) > rule.max_length:
            result.is_valid = False
            result.validation_errors.append(f"Maximum length is {rule.max_length}")
        
        # Pattern validation
        if rule.pattern and not re.match(rule.pattern, str(value)):
            result.is_valid = False
            result.validation_errors.append("Pattern validation failed")
        
        # Allowed values
        if rule.allowed_values and value not in rule.allowed_values:
            result.is_valid = False
            result.validation_errors.append("Value not in allowed list")
        
        # Custom validator
        if rule.custom_validator:
            try:
                is_valid, error = rule.custom_validator(value)
                if not is_valid:
                    result.is_valid = False
                    result.validation_errors.append(error or "Custom validation failed")
            except Exception as e:
                result.is_valid = False
                result.validation_errors.append(f"Validation error: {str(e)}")
        
        return result


# FastAPI Integration
class SanitizedInput:
    """FastAPI dependency for sanitized input."""
    
    def __init__(self, sanitizer: Optional[InputSanitizer] = None):
        self.sanitizer = sanitizer or InputSanitizer()
    
    async def __call__(
        self,
        request: Request,
        schema_name: Optional[str] = None,
        validation_rules: Optional[Dict[str, ValidationRule]] = None
    ) -> Dict[str, Any]:
        """
        Sanitize request data.
        
        Args:
            request: FastAPI request
            schema_name: Optional schema name
            validation_rules: Optional validation rules
            
        Returns:
            Sanitized data dictionary
        """
        # Get request data
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            try:
                data = await request.json()
            except:
                data = {}
        elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            data = await request.form()
            data = dict(data)
        else:
            data = {}
        
        # Apply validation
        if schema_name:
            results = self.sanitizer.validate_with_schema(data, schema_name, request)
        elif validation_rules:
            # Create temporary schema
            temp_schema = {k: v for k, v in validation_rules.items()}
            results = self.sanitizer.validate_with_schema(data, "temp", request)
        else:
            # Basic sanitization of all fields
            results = {}
            for key, value in data.items():
                result = self.sanitizer.sanitize(value, field_name=key, request=request)
                results[key] = result
        
        # Check if any validation failed
        failed_fields = []
        sanitized_data = {}
        
        for field_name, result in results.items():
            if not result.is_valid:
                failed_fields.append({
                    "field": field_name,
                    "errors": result.validation_errors,
                    "threat_level": result.threat_level.value
                })
            else:
                sanitized_data[field_name] = result.sanitized_value
        
        # Raise HTTPException if validation failed
        if failed_fields:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Input validation failed",
                    "failed_fields": failed_fields
                }
            )
        
        return sanitized_data


# Decorator for automatic sanitization
def sanitize_input(
    schema_name: Optional[str] = None,
    validation_rules: Optional[Dict[str, ValidationRule]] = None,
    sanitizer: Optional[InputSanitizer] = None
):
    """
    Decorator to sanitize function inputs.
    
    Args:
        schema_name: Schema name for validation
        validation_rules: Validation rules dictionary
        sanitizer: Custom sanitizer instance
    """
    def decorator(func):
        from functools import wraps
        import inspect
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Find request parameter
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                for value in kwargs.values():
                    if isinstance(value, Request):
                        request = value
                        break
            
            if request:
                # Sanitize input
                sanitizer_instance = sanitizer or InputSanitizer()
                dependency = SanitizedInput(sanitizer_instance)
                
                try:
                    sanitized_data = await dependency(request, schema_name, validation_rules)
                    
                    # Merge sanitized data with kwargs
                    kwargs.update(sanitized_data)
                    
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Input sanitization failed: {str(e)}"
                    )
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            return func(*args, **kwargs)
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Default schemas
def get_default_schemas() -> Dict[str, Dict[str, ValidationRule]]:
    """Get default validation schemas."""
    return {
        "user_registration": {
            "username": ValidationRule(
                validation_type=ValidationType.STRING,
                required=True,
                min_length=3,
                max_length=50,
                pattern=r'^[a-zA-Z0-9_\-\.]+$',
                error_message="Username can only contain letters, numbers, dots, dashes, and underscores"
            ),
            "email": ValidationRule(
                validation_type=ValidationType.EMAIL,
                required=True,
                error_message="Invalid email address"
            ),
            "password": ValidationRule(
                validation_type=ValidationType.PASSWORD,
                required=True,
                min_length=8,
                error_message="Password must be at least 8 characters"
            ),
            "full_name": ValidationRule(
                validation_type=ValidationType.STRING,
                required=False,
                max_length=100,
                sanitization_level=SanitizationLevel.STRICT
            ),
        },
        "incident_report": {
            "title": ValidationRule(
                validation_type=ValidationType.STRING,
                required=True,
                min_length=5,
                max_length=200,
                sanitization_level=SanitizationLevel.STRICT
            ),
            "description": ValidationRule(
                validation_type=ValidationType.HTML,
                required=True,
                min_length=10,
                max_length=5000,
                sanitization_level=SanitizationLevel.MODERATE
            ),
            "location": ValidationRule(
                validation_type=ValidationType.STRING,
                required=True,
                max_length=200
            ),
            "category": ValidationRule(
                validation_type=ValidationType.STRING,
                required=True,
                allowed_values=["emergency", "hazard", "crime", "accident", "other"]
            ),
            "images": ValidationRule(
                validation_type=ValidationType.JSON,
                required=False
            ),
        },
        "api_request": {
            "query": ValidationRule(
                validation_type=ValidationType.STRING,
                required=False,
                max_length=500,
                sanitization_level=SanitizationLevel.STRICT
            ),
            "page": ValidationRule(
                validation_type=ValidationType.INTEGER,
                required=False,
                custom_validator=lambda x: (1 <= int(x) <= 100, "Page must be between 1 and 100")
            ),
            "limit": ValidationRule(
                validation_type=ValidationType.INTEGER,
                required=False,
                custom_validator=lambda x: (1 <= int(x) <= 100, "Limit must be between 1 and 100")
            ),
            "sort_by": ValidationRule(
                validation_type=ValidationType.STRING,
                required=False,
                allowed_values=["created_at", "updated_at", "title", "priority"]
            ),
            "sort_order": ValidationRule(
                validation_type=ValidationType.STRING,
                required=False,
                allowed_values=["asc", "desc"]
            ),
        },
    }


# Initialize default sanitizer
default_sanitizer = InputSanitizer(
    sanitization_level=SanitizationLevel.MODERATE,
    enable_audit_logging=True
)

# Register default schemas
for schema_name, schema in get_default_schemas().items():
    default_sanitizer.register_schema(schema_name, schema)


# Utility functions
def sanitize_string(input_str: str, level: SanitizationLevel = SanitizationLevel.MODERATE) -> str:
    """Quick string sanitization utility."""
    sanitizer = InputSanitizer(sanitization_level=level, enable_audit_logging=False)
    result = sanitizer.sanitize(input_str, ValidationType.STRING)
    return str(result.sanitized_value)


def sanitize_html(input_html: str, level: SanitizationLevel = SanitizationLevel.MODERATE) -> str:
    """Quick HTML sanitization utility."""
    sanitizer = InputSanitizer(sanitization_level=level, enable_audit_logging=False)
    result = sanitizer.sanitize(input_html, ValidationType.HTML)
    return str(result.sanitized_value)


def sanitize_url(input_url: str) -> str:
    """Quick URL sanitization utility."""
    sanitizer = InputSanitizer(enable_audit_logging=False)
    result = sanitizer.sanitize(input_url, ValidationType.URL)
    return str(result.sanitized_value)


def is_safe_input(input_data: Any, validation_type: ValidationType = ValidationType.STRING) -> bool:
    """Quick safety check utility."""
    sanitizer = InputSanitizer(enable_audit_logging=False)
    result = sanitizer.sanitize(input_data, validation_type)
    return result.is_valid and result.threat_level == ThreatLevel.SAFE


# Export main components
__all__ = [
    # Classes
    "InputSanitizer",
    "MaliciousPatternDetector",
    "SanitizedInput",
    "ValidationRule",
    "SanitizationResult",
    
    # Enums
    "SanitizationLevel",
    "ValidationType",
    "ThreatLevel",
    
    # Functions
    "sanitize_input",
    "sanitize_string",
    "sanitize_html",
    "sanitize_url",
    "is_safe_input",
    "get_default_schemas",
    
    # Default instances
    "default_sanitizer",
]