"""
Security utilities including encryption, hashing, and JWT handling.
"""

import hashlib
import hmac
import base64
import secrets
import string
from typing import Any, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import jwt
import bcrypt

from .config import get_config, SecurityConfig
from .constants import SecurityConstants
from .exceptions import SecurityError


class PasswordHasher:
    """Password hashing utilities."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
            
        Raises:
            SecurityError: If hashing fails
        """
        try:
            # Validate password
            if len(password) < SecurityConstants.PASSWORD_MIN_LENGTH:
                raise SecurityError(
                    f"Password must be at least {SecurityConstants.PASSWORD_MIN_LENGTH} characters"
                )
            
            if len(password) > SecurityConstants.PASSWORD_MAX_LENGTH:
                raise SecurityError(
                    f"Password must be at most {SecurityConstants.PASSWORD_MAX_LENGTH} characters"
                )
            
            # Hash using bcrypt
            salt = bcrypt.gensalt(rounds=get_config().security.password_hash_rounds)
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
            
        except Exception as e:
            raise SecurityError(f"Failed to hash password: {str(e)}")
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """
        Verify a password against a hash.
        
        Args:
            password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches hash
            
        Raises:
            SecurityError: If verification fails
        """
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            raise SecurityError(f"Failed to verify password: {str(e)}")
    
    @staticmethod
    def generate_random_password(length: int = 16) -> str:
        """
        Generate a random secure password.
        
        Args:
            length: Password length
            
        Returns:
            Random password
        """
        if length < 8:
            length = 8
        
        # Character sets
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        symbols = "!@#$%^&*"
        
        # Ensure at least one character from each set
        password = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits),
            secrets.choice(symbols),
        ]
        
        # Fill the rest
        all_chars = lowercase + uppercase + digits + symbols
        password += [secrets.choice(all_chars) for _ in range(length - 4)]
        
        # Shuffle
        secrets.SystemRandom().shuffle(password)
        
        return ''.join(password)


class EncryptionManager:
    """Data encryption/decryption utilities."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize encryption manager.
        
        Args:
            secret_key: Encryption key (defaults from config)
        """
        config = get_config()
        self.secret_key = secret_key or config.security.secret_key.encode('utf-8')
        self._validate_key()
    
    def _validate_key(self) -> None:
        """Validate encryption key."""
        if len(self.secret_key) < SecurityConstants.ENCRYPTION_KEY_LENGTH:
            raise SecurityError(
                f"Encryption key must be at least {SecurityConstants.ENCRYPTION_KEY_LENGTH} bytes"
            )
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Base64 encoded encrypted data
            
        Raises:
            SecurityError: If encryption fails
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Generate random IV
            iv = secrets.token_bytes(SecurityConstants.IV_LENGTH)
            
            # Create AES-GCM cipher
            cipher = AESGCM(self.secret_key[:SecurityConstants.ENCRYPTION_KEY_LENGTH])
            
            # Encrypt
            encrypted_data = cipher.encrypt(iv, data, None)
            
            # Combine IV and encrypted data
            combined = iv + encrypted_data
            
            # Return base64 encoded
            return base64.urlsafe_b64encode(combined).decode('utf-8')
            
        except Exception as e:
            raise SecurityError(f"Failed to encrypt data: {str(e)}")
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted string
            
        Raises:
            SecurityError: If decryption fails
        """
        try:
            # Decode base64
            combined = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            
            # Extract IV and encrypted data
            iv = combined[:SecurityConstants.IV_LENGTH]
            encrypted = combined[SecurityConstants.IV_LENGTH:]
            
            # Create AES-GCM cipher
            cipher = AESGCM(self.secret_key[:SecurityConstants.ENCRYPTION_KEY_LENGTH])
            
            # Decrypt
            decrypted_data = cipher.decrypt(iv, encrypted, None)
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            raise SecurityError(f"Failed to decrypt data: {str(e)}")


class TokenManager:
    """JWT token management."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        Initialize token manager.
        
        Args:
            config: Security configuration
        """
        self.config = config or get_config().security
        self.algorithm = self.config.algorithm
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create access token.
        
        Args:
            data: Token payload data
            expires_delta: Token expiration time
            
        Returns:
            JWT token
            
        Raises:
            SecurityError: If token creation fails
        """
        try:
            to_encode = data.copy()
            
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(
                    minutes=self.config.access_token_expire_minutes
                )
            
            to_encode.update({
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "access",
            })
            
            return jwt.encode(
                to_encode,
                self.config.secret_key,
                algorithm=self.algorithm
            )
            
        except Exception as e:
            raise SecurityError(f"Failed to create access token: {str(e)}")
    
    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create refresh token.
        
        Args:
            data: Token payload data
            expires_delta: Token expiration time
            
        Returns:
            JWT refresh token
            
        Raises:
            SecurityError: If token creation fails
        """
        try:
            to_encode = data.copy()
            
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(
                    days=self.config.refresh_token_expire_days
                )
            
            to_encode.update({
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "refresh",
            })
            
            return jwt.encode(
                to_encode,
                self.config.secret_key,
                algorithm=self.algorithm
            )
            
        except Exception as e:
            raise SecurityError(f"Failed to create refresh token: {str(e)}")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Decoded token payload
            
        Raises:
            SecurityError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
            
        except jwt.ExpiredSignatureError:
            raise SecurityError("Token has expired", details={"error": "token_expired"})
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Invalid token: {str(e)}", details={"error": "invalid_token"})
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token
            
        Raises:
            SecurityError: If refresh fails
        """
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token)
            
            # Check token type
            if payload.get("type") != "refresh":
                raise SecurityError("Invalid token type", details={"error": "invalid_token_type"})
            
            # Create new access token with same data (excluding token-specific fields)
            access_token_data = {
                k: v for k, v in payload.items()
                if k not in ["exp", "iat", "type"]
            }
            
            return self.create_access_token(access_token_data)
            
        except SecurityError:
            raise
        except Exception as e:
            raise SecurityError(f"Failed to refresh token: {str(e)}")


class SecurityManager:
    """Main security manager combining all security utilities."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        Initialize security manager.
        
        Args:
            config: Security configuration
        """
        self.config = config or get_config().security
        self.password_hasher = PasswordHasher()
        self.encryption_manager = EncryptionManager()
        self.token_manager = TokenManager(self.config)
    
    def generate_api_key(self, prefix: str = "sk_", length: int = 32) -> str:
        """
        Generate a secure API key.
        
        Args:
            prefix: Key prefix
            length: Key length (excluding prefix)
            
        Returns:
            API key
        """
        # Generate random bytes
        random_bytes = secrets.token_bytes(length)
        
        # Convert to URL-safe base64
        key = base64.urlsafe_b64encode(random_bytes).decode('utf-8')
        
        # Remove padding and truncate to desired length
        key = key.rstrip('=')[:length]
        
        return prefix + key
    
    def generate_csrf_token(self) -> Tuple[str, str]:
        """
        Generate CSRF token and its signed version.
        
        Returns:
            Tuple of (token, signed_token)
        """
        token = secrets.token_urlsafe(32)
        signed_token = self._sign_data(token)
        return token, signed_token
    
    def verify_csrf_token(self, token: str, signed_token: str) -> bool:
        """
        Verify CSRF token.
        
        Args:
            token: Original token
            signed_token: Signed token
            
        Returns:
            True if token is valid
        """
        try:
            expected_signed = self._sign_data(token)
            return hmac.compare_digest(expected_signed, signed_token)
        except Exception:
            return False
    
    def _sign_data(self, data: str) -> str:
        """
        Sign data using HMAC.
        
        Args:
            data: Data to sign
            
        Returns:
            HMAC signature
        """
        signature = hmac.new(
            self.config.secret_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.urlsafe_b64encode(signature).decode('utf-8')
    
    def sanitize_input(self, input_str: str) -> str:
        """
        Sanitize user input to prevent XSS.
        
        Args:
            input_str: Input string
            
        Returns:
            Sanitized string
        """
        import html
        
        # Escape HTML special characters
        sanitized = html.escape(input_str)
        
        # Remove potentially dangerous characters
        dangerous_patterns = [
            r"<script.*?>.*?</script>",
            r"javascript:",
            r"on\w+=",
        ]
        
        import re
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "score": 0,
            "issues": [],
            "strength": "weak",
        }
        
        # Check length
        if len(password) < SecurityConstants.PASSWORD_MIN_LENGTH:
            results["valid"] = False
            results["issues"].append(
                f"Password must be at least {SecurityConstants.PASSWORD_MIN_LENGTH} characters"
            )
        
        # Check character variety
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?/" for c in password)
        
        # Calculate score
        if has_lower:
            results["score"] += 1
        if has_upper:
            results["score"] += 1
        if has_digit:
            results["score"] += 1
        if has_special:
            results["score"] += 1
        if len(password) >= 12:
            results["score"] += 1
        
        # Determine strength
        if results["score"] >= 4:
            results["strength"] = "strong"
        elif results["score"] >= 3:
            results["strength"] = "medium"
        else:
            results["strength"] = "weak"
            if results["valid"]:
                results["valid"] = False
                results["issues"].append("Password is too weak")
        
        return results


# Global instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """
    Get or create security manager instance.
    
    Returns:
        SecurityManager instance
    """
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


# Convenience functions
def hash_password(password: str) -> str:
    """Hash a password."""
    return get_security_manager().password_hasher.hash_password(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password."""
    return get_security_manager().password_hasher.verify_password(password, hashed_password)


def encrypt_data(data: Union[str, bytes]) -> str:
    """Encrypt data."""
    return get_security_manager().encryption_manager.encrypt(data)


def decrypt_data(encrypted_data: str) -> str:
    """Decrypt data."""
    return get_security_manager().encryption_manager.decrypt(encrypted_data)


def create_access_token(data: Dict[str, Any]) -> str:
    """Create access token."""
    return get_security_manager().token_manager.create_access_token(data)


def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode token."""
    return get_security_manager().token_manager.verify_token(token)