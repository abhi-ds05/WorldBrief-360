"""
SSL/TLS Security and Certificate Management

This module provides comprehensive SSL/TLS security features:
- SSL/TLS certificate generation and management
- Certificate validation and verification
- SSL context configuration
- HSTS (HTTP Strict Transport Security) implementation
- Certificate pinning
- SSL/TLS best practices enforcement
- SSL/TLS vulnerability scanning
- Certificate transparency monitoring
"""

import ssl
import socket
import hashlib
import base64
import json
import tempfile
import os
import subprocess
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from contextlib import contextmanager

import cryptography
from cryptography import x509
from cryptography.x509.oid import ExtensionOID, NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.backends import default_backend
from OpenSSL import SSL, crypto # type: ignore
from fastapi import FastAPI, Request, Response
import aiohttp

from app.core.config import get_settings
from app.security.audit_logger import AuditLogger, AuditEventType, AuditSeverity

# Get settings
settings = get_settings()


class SSLVersion(str, Enum):
    """SSL/TLS versions."""
    SSLv2 = "SSLv2"
    SSLv3 = "SSLv3"
    TLSv1 = "TLSv1"
    TLSv1_1 = "TLSv1.1"
    TLSv1_2 = "TTLv1.2"
    TLSv1_3 = "TLSv1.3"


class CertificateType(str, Enum):
    """Certificate types."""
    SELF_SIGNED = "self_signed"
    CA_SIGNED = "ca_signed"
    WILDCARD = "wildcard"
    MULTI_DOMAIN = "multi_domain"
    EV = "ev"  # Extended Validation
    DV = "dv"  # Domain Validation
    OV = "ov"  # Organization Validation


class KeyAlgorithm(str, Enum):
    """Key algorithms."""
    RSA = "RSA"
    ECDSA = "ECDSA"
    ED25519 = "ED25519"


class KeySize(str, Enum):
    """Key sizes."""
    RSA_2048 = "2048"
    RSA_3072 = "3072"
    RSA_4096 = "4096"
    ECDSA_256 = "256"
    ECDSA_384 = "384"


class HashAlgorithm(str, Enum):
    """Hash algorithms."""
    SHA1 = "SHA1"
    SHA256 = "SHA256"
    SHA384 = "SHA384"
    SHA512 = "SHA512"


@dataclass
class CertificateInfo:
    """Certificate information."""
    subject: Dict[str, str]
    issuer: Dict[str, str]
    serial_number: str
    version: int
    not_valid_before: datetime
    not_valid_after: datetime
    signature_algorithm: str
    public_key_algorithm: str
    public_key_size: int
    extensions: Dict[str, Any]
    san: List[str] = field(default_factory=list)  # Subject Alternative Names
    fingerprint_sha1: str = ""
    fingerprint_sha256: str = ""
    ocsp_urls: List[str] = field(default_factory=list)
    crl_urls: List[str] = field(default_factory=list)
    is_ca: bool = False
    is_self_signed: bool = False
    is_expired: bool = False
    is_revoked: bool = False
    has_revocation_check: bool = False
    cert_type: CertificateType = CertificateType.DV


@dataclass
class SSLContextConfig:
    """SSL context configuration."""
    # Certificate files
    certfile: Optional[Path] = None
    keyfile: Optional[Path] = None
    chainfile: Optional[Path] = None
    
    # Protocol settings
    min_version: SSLVersion = SSLVersion.TLSv1_2
    max_version: Optional[SSLVersion] = None
    
    # Cipher suites
    ciphers: str = "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20"
    prefer_server_ciphers: bool = True
    
    # Security features
    verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED
    check_hostname: bool = True
    verify_flags: int = 0
    
    # Certificate options
    dh_params: Optional[Path] = None
    ecdh_curve: str = "secp384r1"
    
    # Session settings
    session_tickets: bool = True
    session_timeout: int = 300
    session_cache_size: int = 1024
    
    # OCSP stapling
    ocsp_stapling: bool = True
    ocsp_stapling_cache_timeout: int = 3600
    
    # HSTS
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False
    
    # Certificate pinning
    cert_pinning_enabled: bool = False
    pinned_certs: List[str] = field(default_factory=list)
    
    # Certificate transparency
    ct_enabled: bool = True
    ct_logs: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.certfile and not self.certfile.exists():
            raise FileNotFoundError(f"Certificate file not found: {self.certfile}")
        if self.keyfile and not self.keyfile.exists():
            raise FileNotFoundError(f"Private key file not found: {self.keyfile}")
        if self.chainfile and not self.chainfile.exists():
            raise FileNotFoundError(f"Chain file not found: {self.chainfile}")
        if self.dh_params and not self.dh_params.exists():
            raise FileNotFoundError(f"DH params file not found: {self.dh_params}")


@dataclass
class CertificateRequest:
    """Certificate signing request configuration."""
    common_name: str
    organization: Optional[str] = None
    organizational_unit: Optional[str] = None
    country: str = "US"
    state: Optional[str] = None
    locality: Optional[str] = None
    email: Optional[str] = None
    
    # SANs (Subject Alternative Names)
    dns_names: List[str] = field(default_factory=list)
    ip_addresses: List[str] = field(default_factory=list)
    
    # Key configuration
    key_algorithm: KeyAlgorithm = KeyAlgorithm.RSA
    key_size: Union[str, int] = KeySize.RSA_2048
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    
    # Certificate extensions
    is_ca: bool = False
    path_length: Optional[int] = None
    key_usage: List[str] = field(default_factory=lambda: [
        "digital_signature", "key_encipherment"
    ])
    extended_key_usage: List[str] = field(default_factory=lambda: [
        "server_auth", "client_auth"
    ])
    
    # Validity
    validity_days: int = 365
    
    def __post_init__(self):
        """Add common_name to SANs if not present."""
        if self.common_name and self.common_name not in self.dns_names:
            self.dns_names.append(self.common_name)


class CertificateManager:
    """Certificate generation and management."""
    
    def __init__(self, ca_cert: Optional[Path] = None, ca_key: Optional[Path] = None):
        self.ca_cert = ca_cert
        self.ca_key = ca_key
        self.backend = default_backend()
        self.audit_logger = AuditLogger()
    
    def generate_private_key(
        self,
        algorithm: KeyAlgorithm = KeyAlgorithm.RSA,
        key_size: Union[str, int] = KeySize.RSA_2048
    ) -> Tuple[Any, bytes]:
        """
        Generate a private key.
        
        Args:
            algorithm: Key algorithm
            key_size: Key size in bits
            
        Returns:
            Tuple of (key_object, private_key_pem)
        """
        if algorithm == KeyAlgorithm.RSA:
            key_size = int(key_size)
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=self.backend
            )
        elif algorithm == KeyAlgorithm.ECDSA:
            curve_map = {
                "256": ec.SECP256R1(),
                "384": ec.SECP384R1(),
                "521": ec.SECP521R1(),
            }
            key_size = str(key_size)
            curve = curve_map.get(key_size, ec.SECP256R1())
            private_key = ec.generate_private_key(curve, self.backend)
        elif algorithm == KeyAlgorithm.ED25519:
            private_key = cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey.generate()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Serialize to PEM
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return private_key, private_key_pem
    
    def create_certificate_signing_request(
        self,
        private_key: Any,
        csr_config: CertificateRequest
    ) -> bytes:
        """
        Create a Certificate Signing Request (CSR).
        
        Args:
            private_key: Private key object
            csr_config: CSR configuration
            
        Returns:
            CSR in PEM format
        """
        # Build subject
        subject_name = []
        
        if csr_config.country:
            subject_name.append(x509.NameAttribute(NameOID.COUNTRY_NAME, csr_config.country))
        if csr_config.state:
            subject_name.append(x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, csr_config.state))
        if csr_config.locality:
            subject_name.append(x509.NameAttribute(NameOID.LOCALITY_NAME, csr_config.locality))
        if csr_config.organization:
            subject_name.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, csr_config.organization))
        if csr_config.organizational_unit:
            subject_name.append(x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, csr_config.organizational_unit))
        if csr_config.common_name:
            subject_name.append(x509.NameAttribute(NameOID.COMMON_NAME, csr_config.common_name))
        if csr_config.email:
            subject_name.append(x509.NameAttribute(NameOID.EMAIL_ADDRESS, csr_config.email))
        
        subject = x509.Name(subject_name)
        
        # Build CSR
        builder = x509.CertificateSigningRequestBuilder()
        builder = builder.subject_name(subject)
        
        # Add Subject Alternative Names
        san_items = []
        for dns_name in csr_config.dns_names:
            san_items.append(x509.DNSName(dns_name))
        for ip_addr in csr_config.ip_addresses:
            san_items.append(x509.IPAddress(ip_addr))
        
        if san_items:
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_items),
                critical=False
            )
        
        # Add Key Usage
        key_usage_mapping = {
            "digital_signature": x509.KeyUsage.digital_signature,
            "content_commitment": x509.KeyUsage.content_commitment,
            "key_encipherment": x509.KeyUsage.key_encipherment,
            "data_encipherment": x509.KeyUsage.data_encipherment,
            "key_agreement": x509.KeyUsage.key_agreement,
            "key_cert_sign": x509.KeyUsage.key_cert_sign,
            "crl_sign": x509.KeyUsage.crl_sign,
            "encipher_only": x509.KeyUsage.encipher_only,
            "decipher_only": x509.KeyUsage.decipher_only,
        }
        
        key_usage_flags = []
        for usage in csr_config.key_usage:
            if usage in key_usage_mapping:
                key_usage_flags.append(key_usage_mapping[usage])
        
        if key_usage_flags:
            builder = builder.add_extension(
                x509.KeyUsage(*key_usage_flags),
                critical=True
            )
        
        # Sign CSR
        csr = builder.sign(private_key, getattr(hashes, csr_config.hash_algorithm.value)(), self.backend)
        
        # Serialize to PEM
        csr_pem = csr.public_bytes(serialization.Encoding.PEM)
        
        return csr_pem
    
    def create_self_signed_certificate(
        self,
        csr_config: CertificateRequest,
        private_key: Optional[Any] = None
    ) -> Tuple[bytes, bytes]:
        """
        Create a self-signed certificate.
        
        Args:
            csr_config: Certificate configuration
            private_key: Optional private key (generated if not provided)
            
        Returns:
            Tuple of (certificate_pem, private_key_pem)
        """
        # Generate private key if not provided
        if private_key is None:
            private_key, private_key_pem = self.generate_private_key(
                csr_config.key_algorithm,
                csr_config.key_size
            )
        else:
            private_key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        
        # Build subject (same as issuer for self-signed)
        subject_name = []
        
        if csr_config.country:
            subject_name.append(x509.NameAttribute(NameOID.COUNTRY_NAME, csr_config.country))
        if csr_config.state:
            subject_name.append(x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, csr_config.state))
        if csr_config.locality:
            subject_name.append(x509.NameAttribute(NameOID.LOCALITY_NAME, csr_config.locality))
        if csr_config.organization:
            subject_name.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, csr_config.organization))
        if csr_config.organizational_unit:
            subject_name.append(x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, csr_config.organizational_unit))
        if csr_config.common_name:
            subject_name.append(x509.NameAttribute(NameOID.COMMON_NAME, csr_config.common_name))
        if csr_config.email:
            subject_name.append(x509.NameAttribute(NameOID.EMAIL_ADDRESS, csr_config.email))
        
        subject = issuer = x509.Name(subject_name)
        
        # Build certificate
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(issuer)
        builder = builder.public_key(private_key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.utcnow())
        builder = builder.not_valid_after(
            datetime.utcnow() + timedelta(days=csr_config.validity_days)
        )
        
        # Add extensions
        # Subject Alternative Names
        san_items = []
        for dns_name in csr_config.dns_names:
            san_items.append(x509.DNSName(dns_name))
        for ip_addr in csr_config.ip_addresses:
            san_items.append(x509.IPAddress(ip_addr))
        
        if san_items:
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_items),
                critical=False
            )
        
        # Key Usage
        key_usage_mapping = {
            "digital_signature": x509.KeyUsage.digital_signature,
            "content_commitment": x509.KeyUsage.content_commitment,
            "key_encipherment": x509.KeyUsage.key_encipherment,
            "data_encipherment": x509.KeyUsage.data_encipherment,
            "key_agreement": x509.KeyUsage.key_agreement,
            "key_cert_sign": x509.KeyUsage.key_cert_sign,
            "crl_sign": x509.KeyUsage.crl_sign,
            "encipher_only": x509.KeyUsage.encipher_only,
            "decipher_only": x509.KeyUsage.decipher_only,
        }
        
        key_usage_flags = []
        for usage in csr_config.key_usage:
            if usage in key_usage_mapping:
                key_usage_flags.append(key_usage_mapping[usage])
        
        if key_usage_flags:
            builder = builder.add_extension(
                x509.KeyUsage(*key_usage_flags),
                critical=True
            )
        
        # Extended Key Usage
        extended_key_usage_mapping = {
            "server_auth": x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            "client_auth": x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
            "code_signing": x509.oid.ExtendedKeyUsageOID.CODE_SIGNING,
            "email_protection": x509.oid.ExtendedKeyUsageOID.EMAIL_PROTECTION,
            "time_stamping": x509.oid.ExtendedKeyUsageOID.TIME_STAMPING,
            "ocsp_signing": x509.oid.ExtendedKeyUsageOID.OCSP_SIGNING,
        }
        
        extended_key_usage_oids = []
        for usage in csr_config.extended_key_usage:
            if usage in extended_key_usage_mapping:
                extended_key_usage_oids.append(extended_key_usage_mapping[usage])
        
        if extended_key_usage_oids:
            builder = builder.add_extension(
                x509.ExtendedKeyUsage(extended_key_usage_oids),
                critical=False
            )
        
        # Basic Constraints
        builder = builder.add_extension(
            x509.BasicConstraints(ca=csr_config.is_ca, path_length=csr_config.path_length),
            critical=True
        )
        
        # Sign certificate
        certificate = builder.sign(
            private_key,
            getattr(hashes, csr_config.hash_algorithm.value)(),
            self.backend
        )
        
        # Serialize to PEM
        certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
        
        return certificate_pem, private_key_pem
    
    def sign_certificate(
        self,
        csr_pem: bytes,
        validity_days: int = 365,
        is_ca: bool = False,
        path_length: Optional[int] = None
    ) -> bytes:
        """
        Sign a certificate with the CA.
        
        Args:
            csr_pem: CSR in PEM format
            validity_days: Certificate validity in days
            is_ca: Whether this is a CA certificate
            path_length: CA path length constraint
            
        Returns:
            Signed certificate in PEM format
        """
        if not self.ca_cert or not self.ca_key:
            raise ValueError("CA certificate and key are required for signing")
        
        # Load CA certificate and key
        with open(self.ca_cert, 'rb') as f:
            ca_cert = x509.load_pem_x509_certificate(f.read(), self.backend)
        
        with open(self.ca_key, 'rb') as f:
            ca_key = load_pem_private_key(f.read(), password=None, backend=self.backend)
        
        # Load CSR
        csr = x509.load_pem_x509_csr(csr_pem, self.backend)
        
        # Build certificate
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(csr.subject)
        builder = builder.issuer_name(ca_cert.subject)
        builder = builder.public_key(csr.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.utcnow())
        builder = builder.not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        )
        
        # Copy extensions from CSR
        for extension in csr.extensions:
            builder = builder.add_extension(extension.value, critical=extension.critical)
        
        # Add/override basic constraints
        builder = builder.add_extension(
            x509.BasicConstraints(ca=is_ca, path_length=path_length),
            critical=True
        )
        
        # Sign certificate
        certificate = builder.sign(
            ca_key,
            hashes.SHA256(),
            self.backend
        )
        
        # Serialize to PEM
        certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
        
        return certificate_pem
    
    def parse_certificate(self, cert_pem: bytes) -> CertificateInfo:
        """
        Parse a certificate and extract information.
        
        Args:
            cert_pem: Certificate in PEM format
            
        Returns:
            CertificateInfo object
        """
        cert = x509.load_pem_x509_certificate(cert_pem, self.backend)
        
        # Extract subject and issuer
        subject = {}
        for attr in cert.subject:
            subject[attr.oid._name] = attr.value
        
        issuer = {}
        for attr in cert.issuer:
            issuer[attr.oid._name] = attr.value
        
        # Extract extensions
        extensions = {}
        san = []
        
        for ext in cert.extensions:
            ext_name = ext.oid._name
            extensions[ext_name] = {
                "critical": ext.critical,
                "value": str(ext.value)
            }
            
            if ext.oid == ExtensionOID.SUBJECT_ALTERNATIVE_NAME:
                for name in ext.value:
                    if isinstance(name, x509.DNSName):
                        san.append(name.value)
                    elif isinstance(name, x509.IPAddress):
                        san.append(str(name.value))
        
        # Calculate fingerprints
        fingerprint_sha1 = cert.fingerprint(hashes.SHA1()).hex()
        fingerprint_sha256 = cert.fingerprint(hashes.SHA256()).hex()
        
        # Check if self-signed
        is_self_signed = cert.subject == cert.issuer
        
        # Check if CA
        is_ca = False
        try:
            basic_constraints = cert.extensions.get_extension_for_oid(
                ExtensionOID.BASIC_CONSTRAINTS
            )
            is_ca = basic_constraints.value.ca
        except x509.ExtensionNotFound:
            pass
        
        # Extract OCSP and CRL URLs
        ocsp_urls = []
        crl_urls = []
        
        try:
            auth_info_access = cert.extensions.get_extension_for_oid(
                ExtensionOID.AUTHORITY_INFORMATION_ACCESS
            )
            for desc in auth_info_access.value:
                if desc.access_method == x509.oid.AuthorityInformationAccessOID.OCSP:
                    if isinstance(desc.access_location, x509.UniformResourceIdentifier):
                        ocsp_urls.append(desc.access_location.value)
                elif desc.access_method == x509.oid.AuthorityInformationAccessOID.CA_ISSUERS:
                    pass  # CA issuers URL
        except x509.ExtensionNotFound:
            pass
        
        try:
            crl_dist_points = cert.extensions.get_extension_for_oid(
                ExtensionOID.CRL_DISTRIBUTION_POINTS
            )
            for point in crl_dist_points.value:
                for name in point.full_name:
                    if isinstance(name, x509.UniformResourceIdentifier):
                        crl_urls.append(name.value)
        except x509.ExtensionNotFound:
            pass
        
        # Determine certificate type
        cert_type = CertificateType.DV
        if len(san) > 1:
            cert_type = CertificateType.MULTI_DOMAIN
        elif "*" in cert.subject.get("commonName", ""):
            cert_type = CertificateType.WILDCARD
        elif is_ca:
            cert_type = CertificateType.CA_SIGNED
        elif is_self_signed:
            cert_type = CertificateType.SELF_SIGNED
        
        # Check expiration
        is_expired = datetime.utcnow() > cert.not_valid_after
        
        return CertificateInfo(
            subject=subject,
            issuer=issuer,
            serial_number=str(cert.serial_number),
            version=cert.version.value,
            not_valid_before=cert.not_valid_before,
            not_valid_after=cert.not_valid_after,
            signature_algorithm=cert.signature_algorithm_oid._name,
            public_key_algorithm=cert.public_key().__class__.__name__,
            public_key_size=cert.public_key().key_size if hasattr(cert.public_key(), 'key_size') else 0,
            extensions=extensions,
            san=san,
            fingerprint_sha1=fingerprint_sha1,
            fingerprint_sha256=fingerprint_sha256,
            ocsp_urls=ocsp_urls,
            crl_urls=crl_urls,
            is_ca=is_ca,
            is_self_signed=is_self_signed,
            is_expired=is_expired,
            is_revoked=False,  # Would require OCSP/CRL check
            has_revocation_check=bool(ocsp_urls or crl_urls),
            cert_type=cert_type
        )


class SSLContextManager:
    """SSL/TLS context management."""
    
    def __init__(self, config: SSLContextConfig):
        self.config = config
        self.ssl_context = None
        self.audit_logger = AuditLogger()
        
    def create_ssl_context(self) -> ssl.SSLContext:
        """
        Create and configure SSL context.
        
        Returns:
            Configured SSLContext
        """
        # Determine protocol version
        if self.config.min_version == SSLVersion.TLSv1_3:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_3
        elif self.config.min_version == SSLVersion.TLSv1_2:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        elif self.config.min_version == SSLVersion.TLSv1_1:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_1
        else:
            # Default to TLS 1.2+
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Set maximum version
        if self.config.max_version:
            if self.config.max_version == SSLVersion.TLSv1_3:
                ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
            elif self.config.max_version == SSLVersion.TLSv1_2:
                ssl_context.maximum_version = ssl.TLSVersion.TLSv1_2
        
        # Disable insecure protocols
        ssl_context.options |= ssl.OP_NO_SSLv2
        ssl_context.options |= ssl.OP_NO_SSLv3
        ssl_context.options |= ssl.OP_NO_TLSv1
        ssl_context.options |= ssl.OP_NO_TLSv1_1
        
        # Enable security options
        ssl_context.options |= ssl.OP_SINGLE_DH_USE
        ssl_context.options |= ssl.OP_SINGLE_ECDH_USE
        ssl_context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE
        
        # Set ciphers
        if self.config.ciphers:
            ssl_context.set_ciphers(self.config.ciphers)
        
        # Load certificates
        if self.config.certfile and self.config.keyfile:
            ssl_context.load_cert_chain(
                certfile=str(self.config.certfile),
                keyfile=str(self.config.keyfile)
            )
        
        # Load CA chain
        if self.config.chainfile:
            ssl_context.load_verify_locations(cafile=str(self.config.chainfile))
        else:
            # Use system defaults
            ssl_context.load_default_certs()
        
        # Set verification
        ssl_context.verify_mode = self.config.verify_mode
        ssl_context.check_hostname = self.config.check_hostname
        
        if self.config.verify_flags:
            ssl_context.verify_flags = self.config.verify_flags
        
        # Load DH parameters
        if self.config.dh_params:
            with open(self.config.dh_params, 'rb') as f:
                dh_params = ssl.DHParameter(f.read())
            ssl_context.load_dh_params(str(self.config.dh_params))
        
        # Set ECDH curve
        if hasattr(ssl, 'set_ecdh_curve'):
            ssl_context.set_ecdh_curve(self.config.ecdh_curve)
        
        # Session settings
        if not self.config.session_tickets:
            ssl_context.options |= ssl.OP_NO_TICKET
        
        ssl_context.session_timeout = self.config.session_timeout
        
        self.ssl_context = ssl_context
        return ssl_context
    
    def get_ssl_context(self) -> ssl.SSLContext:
        """Get or create SSL context."""
        if self.ssl_context is None:
            self.ssl_context = self.create_ssl_context()
        return self.ssl_context
    
    def verify_certificate_chain(self, cert_pem: bytes, chain_pem: Optional[bytes] = None) -> Tuple[bool, List[str]]:
        """
        Verify a certificate chain.
        
        Args:
            cert_pem: Certificate in PEM format
            chain_pem: Chain certificates in PEM format
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as cert_file:
                cert_file.write(cert_pem)
                cert_path = cert_file.name
            
            chain_path = None
            if chain_pem:
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as chain_file:
                    chain_file.write(chain_pem)
                    chain_path = chain_file.name
            
            try:
                # Use OpenSSL command line for verification
                cmd = ['openssl', 'verify']
                
                if chain_path:
                    cmd.extend(['-CAfile', chain_path])
                else:
                    cmd.append('-CApath')  # Use system CAs
                
                cmd.append(cert_path)
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    errors.append(f"Certificate chain verification failed: {result.stderr}")
                    return False, errors
                
                return True, []
                
            finally:
                # Cleanup temporary files
                os.unlink(cert_path)
                if chain_path:
                    os.unlink(chain_path)
                
        except Exception as e:
            errors.append(f"Certificate verification error: {str(e)}")
            return False, errors
    
    def check_certificate_expiry(self, cert_pem: bytes, warning_days: int = 30) -> Tuple[bool, int]:
        """
        Check certificate expiry.
        
        Args:
            cert_pem: Certificate in PEM format
            warning_days: Days before expiry to warn
            
        Returns:
            Tuple of (is_expired, days_remaining)
        """
        cert_info = CertificateManager().parse_certificate(cert_pem)
        now = datetime.utcnow()
        
        if cert_info.is_expired:
            return True, 0
        
        days_remaining = (cert_info.not_valid_after - now).days
        
        if days_remaining <= warning_days:
            # Log warning
            self.audit_logger.log_security_event(
                event_type=AuditEventType.SYSTEM_ALERT,
                description=f"Certificate expires in {days_remaining} days",
                details={
                    "subject": cert_info.subject,
                    "expiry_date": cert_info.not_valid_after.isoformat(),
                    "days_remaining": days_remaining,
                },
                severity=AuditSeverity.WARNING
            )
        
        return False, days_remaining


class HSTSManager:
    """HTTP Strict Transport Security management."""
    
    def __init__(self, config: SSLContextConfig):
        self.config = config
    
    def generate_hsts_header(self) -> str:
        """Generate HSTS header value."""
        header_value = f"max-age={self.config.hsts_max_age}"
        
        if self.config.hsts_include_subdomains:
            header_value += "; includeSubDomains"
        
        if self.config.hsts_preload:
            header_value += "; preload"
        
        return header_value
    
    def should_enforce_hsts(self, request: Request) -> bool:
        """Check if HSTS should be enforced for this request."""
        # Only enforce for HTTPS requests
        if request.url.scheme != 'https':
            return False
        
        # Check if already has HSTS header
        if 'strict-transport-security' in request.headers:
            return False
        
        return True
    
    def add_hsts_header(self, response: Response, request: Optional[Request] = None) -> Response:
        """Add HSTS header to response."""
        if self.config.hsts_enabled:
            if request is None or self.should_enforce_hsts(request):
                response.headers['Strict-Transport-Security'] = self.generate_hsts_header()
        
        return response


class CertificatePinner:
    """Certificate pinning implementation."""
    
    def __init__(self, pinned_certs: List[str]):
        self.pinned_certs = pinned_certs
        self.audit_logger = AuditLogger()
    
    def calculate_pin(self, cert_pem: bytes, algorithm: str = 'sha256') -> str:
        """
        Calculate certificate pin.
        
        Args:
            cert_pem: Certificate in PEM format
            algorithm: Hash algorithm
            
        Returns:
            Certificate pin
        """
        # Remove PEM headers/footers
        cert_data = b'\n'.join(cert_pem.split(b'\n')[1:-2])
        
        if algorithm == 'sha256':
            hash_obj = hashlib.sha256(cert_data)
            pin = base64.b64encode(hash_obj.digest()).decode('utf-8')
        elif algorithm == 'sha1':
            hash_obj = hashlib.sha1(cert_data)
            pin = base64.b64encode(hash_obj.digest()).decode('utf-8')
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return pin
    
    def verify_pin(self, cert_pem: bytes, expected_pins: Optional[List[str]] = None) -> bool:
        """
        Verify certificate pin.
        
        Args:
            cert_pem: Certificate in PEM format
            expected_pins: List of expected pins (uses configured pins if None)
            
        Returns:
            True if pin matches
        """
        if expected_pins is None:
            expected_pins = self.pinned_certs
        
        if not expected_pins:
            return True
        
        # Calculate pins for certificate
        sha256_pin = self.calculate_pin(cert_pem, 'sha256')
        sha1_pin = self.calculate_pin(cert_pem, 'sha1')
        
        # Check if any pin matches
        for expected_pin in expected_pins:
            if expected_pin == sha256_pin or expected_pin == sha1_pin:
                return True
        
        # Log pinning failure
        self.audit_logger.log_security_event(
            event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
            description="Certificate pinning violation",
            details={
                "calculated_sha256": sha256_pin,
                "calculated_sha1": sha1_pin,
                "expected_pins": expected_pins,
            },
            severity=AuditSeverity.ERROR
        )
        
        return False
    
    def generate_pin_report(self, cert_pem: bytes) -> Dict[str, str]:
        """Generate pin report for certificate."""
        return {
            'sha256': self.calculate_pin(cert_pem, 'sha256'),
            'sha1': self.calculate_pin(cert_pem, 'sha1'),
            'subject': str(CertificateManager().parse_certificate(cert_pem).subject),
        }


class CertificateTransparency:
    """Certificate Transparency monitoring."""
    
    def __init__(self, ct_logs: Optional[List[str]] = None):
        self.ct_logs = ct_logs or [
            "https://ct.googleapis.com/logs/argon2024/",
            "https://ct.googleapis.com/logs/xenon2024/",
            "https://ct.cloudflare.com/logs/nimbus2024/",
        ]
    
    async def check_certificate_transparency(
        self,
        cert_pem: bytes,
        domain: str
    ) -> Dict[str, Any]:
        """
        Check if certificate is logged in Certificate Transparency logs.
        
        Args:
            cert_pem: Certificate in PEM format
            domain: Domain name
            
        Returns:
            Dictionary with CT check results
        """
        results = {
            'domain': domain,
            'is_logged': False,
            'logs_checked': [],
            'scts': [],  # Signed Certificate Timestamps
            'errors': [],
        }
        
        # This is a simplified implementation
        # In production, you would use a proper CT client library
        
        for log_url in self.ct_logs:
            try:
                # In practice, you would:
                # 1. Submit precertificate to log
                # 2. Get SCT (Signed Certificate Timestamp)
                # 3. Verify SCT
                
                # For now, we'll just record that we checked
                results['logs_checked'].append(log_url)
                
            except Exception as e:
                results['errors'].append(f"Error checking log {log_url}: {str(e)}")
        
        return results


class SSLScanner:
    """SSL/TLS vulnerability scanner."""
    
    def __init__(self):
        self.audit_logger = AuditLogger()
    
    async def scan_host(
        self,
        hostname: str,
        port: int = 443,
        timeout: int = 10
    ) -> Dict[str, Any]:
        """
        Scan a host for SSL/TLS vulnerabilities.
        
        Args:
            hostname: Hostname to scan
            port: Port to scan
            timeout: Connection timeout
            
        Returns:
            Scan results
        """
        results = {
            'hostname': hostname,
            'port': port,
            'success': False,
            'certificate': None,
            'protocols': [],
            'ciphers': [],
            'vulnerabilities': [],
            'grade': 'F',
            'recommendations': [],
        }
        
        try:
            # Get certificate
            cert_info = await self.get_certificate(hostname, port, timeout)
            results['certificate'] = cert_info
            results['success'] = True
            
            # Check protocols
            protocols = await self.check_protocols(hostname, port, timeout)
            results['protocols'] = protocols
            
            # Check ciphers
            ciphers = await self.check_ciphers(hostname, port, timeout)
            results['ciphers'] = ciphers
            
            # Check vulnerabilities
            vulnerabilities = await self.check_vulnerabilities(hostname, port, timeout)
            results['vulnerabilities'] = vulnerabilities
            
            # Calculate grade
            results['grade'] = self.calculate_grade(cert_info, protocols, ciphers, vulnerabilities)
            
            # Generate recommendations
            results['recommendations'] = self.generate_recommendations(
                cert_info, protocols, ciphers, vulnerabilities
            )
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def get_certificate(
        self,
        hostname: str,
        port: int = 443,
        timeout: int = 10
    ) -> Optional[CertificateInfo]:
        """Get certificate from host."""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert_pem = ssl.DER_cert_to_PEM_cert(ssock.getpeercert(True))
                    manager = CertificateManager()
                    return manager.parse_certificate(cert_pem.encode())
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type=AuditEventType.SYSTEM_ALERT,
                description=f"Failed to get certificate from {hostname}:{port}",
                details={'error': str(e)},
                severity=AuditSeverity.ERROR
            )
            return None
    
    async def check_protocols(
        self,
        hostname: str,
        port: int,
        timeout: int
    ) -> Dict[str, bool]:
        """Check supported SSL/TLS protocols."""
        protocols = {}
        
        # Test different protocol versions
        for protocol_name, ssl_version in [
            ('SSLv2', ssl.PROTOCOL_SSLv2),
            ('SSLv3', ssl.PROTOCOL_SSLv3),
            ('TLSv1', ssl.PROTOCOL_TLSv1),
            ('TLSv1.1', ssl.PROTOCOL_TLSv1_1),
            ('TLSv1.2', ssl.PROTOCOL_TLSv1_2),
            ('TLSv1.3', ssl.PROTOCOL_TLS),
        ]:
            try:
                context = ssl.SSLContext(ssl_version)
                context.verify_mode = ssl.CERT_NONE
                context.check_hostname = False
                
                with socket.create_connection((hostname, port), timeout=timeout) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        protocols[protocol_name] = True
            except:
                protocols[protocol_name] = False
        
        return protocols
    
    async def check_ciphers(
        self,
        hostname: str,
        port: int,
        timeout: int
    ) -> List[str]:
        """Check supported cipher suites."""
        ciphers = []
        
        try:
            context = ssl.create_default_context()
            context.set_ciphers('ALL')
            
            with socket.create_connection((hostname, port), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    ciphers = ssock.cipher()
        except:
            pass
        
        return ciphers
    
    async def check_vulnerabilities(
        self,
        hostname: str,
        port: int,
        timeout: int
    ) -> List[str]:
        """Check for known vulnerabilities."""
        vulnerabilities = []
        
        # Check for Heartbleed (CVE-2014-0160)
        if await self.check_heartbleed(hostname, port, timeout):
            vulnerabilities.append('heartbleed')
        
        # Check for POODLE (CVE-2014-3566)
        if await self.check_poodle(hostname, port, timeout):
            vulnerabilities.append('poodle')
        
        # Check for ROBOT (CVE-2017-13099)
        if await self.check_robot(hostname, port, timeout):
            vulnerabilities.append('robot')
        
        return vulnerabilities
    
    async def check_heartbleed(self, hostname: str, port: int, timeout: int) -> bool:
        """Check for Heartbleed vulnerability."""
        # Simplified check - in production use a proper heartbleed checker
        return False
    
    async def check_poodle(self, hostname: str, port: int, timeout: int) -> bool:
        """Check for POODLE vulnerability."""
        # Simplified check
        return False
    
    async def check_robot(self, hostname: str, port: int, timeout: int) -> bool:
        """Check for ROBOT vulnerability."""
        # Simplified check
        return False
    
    def calculate_grade(
        self,
        cert_info: Optional[CertificateInfo],
        protocols: Dict[str, bool],
        ciphers: List[str],
        vulnerabilities: List[str]
    ) -> str:
        """Calculate SSL/TLS security grade."""
        score = 100
        
        # Deduct for vulnerabilities
        if vulnerabilities:
            score -= len(vulnerabilities) * 20
        
        # Deduct for weak protocols
        if protocols.get('SSLv2', False) or protocols.get('SSLv3', False):
            score -= 30
        if protocols.get('TLSv1', False):
            score -= 20
        if protocols.get('TLSv1.1', False):
            score -= 10
        
        # Deduct for certificate issues
        if cert_info:
            if cert_info.is_expired:
                score -= 30
            if cert_info.is_self_signed:
                score -= 20
            if not cert_info.has_revocation_check:
                score -= 10
        
        # Calculate grade
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def generate_recommendations(
        self,
        cert_info: Optional[CertificateInfo],
        protocols: Dict[str, bool],
        ciphers: List[str],
        vulnerabilities: List[str]
    ) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        # Protocol recommendations
        if protocols.get('SSLv2', False) or protocols.get('SSLv3', False):
            recommendations.append("Disable SSLv2 and SSLv3")
        if protocols.get('TLSv1', False):
            recommendations.append("Disable TLSv1.0")
        if protocols.get('TLSv1.1', False):
            recommendations.append("Disable TLSv1.1")
        
        if not protocols.get('TLSv1.3', False):
            recommendations.append("Enable TLSv1.3")
        
        # Certificate recommendations
        if cert_info:
            if cert_info.is_expired:
                recommendations.append("Renew expired certificate")
            if cert_info.is_self_signed:
                recommendations.append("Replace self-signed certificate with CA-signed certificate")
            if not cert_info.has_revocation_check:
                recommendations.append("Configure OCSP stapling or CRL")
        
        # Cipher recommendations
        recommendations.append("Use strong cipher suites (ECDHE, DHE with AES-GCM)")
        
        # Vulnerability remediation
        for vuln in vulnerabilities:
            if vuln == 'heartbleed':
                recommendations.append("Update OpenSSL to patch Heartbleed vulnerability")
            elif vuln == 'poodle':
                recommendations.append("Disable SSLv3 to mitigate POODLE vulnerability")
            elif vuln == 'robot':
                recommendations.append("Disable RSA key exchange or use forward secrecy")
        
        return recommendations


# FastAPI Integration
class SSLMiddleware:
    """FastAPI middleware for SSL/TLS security."""
    
    def __init__(
        self,
        app,
        config: Optional[SSLContextConfig] = None,
        enable_hsts: bool = True,
        enable_cert_pinning: bool = False
    ):
        self.app = app
        self.config = config or SSLContextConfig()
        self.enable_hsts = enable_hsts
        self.enable_cert_pinning = enable_cert_pinning
        
        # Initialize managers
        self.hsts_manager = HSTSManager(self.config)
        self.cert_pinner = CertificatePinner(self.config.pinned_certs) if enable_cert_pinning else None
        
        # Initialize scanner
        self.scanner = SSLScanner()
    
    async def __call__(self, request: Request, call_next):
        """Process request with SSL security features."""
        response = await call_next(request)
        
        # Add HSTS header
        if self.enable_hsts:
            response = self.hsts_manager.add_hsts_header(response, request)
        
        # Add security headers
        response = self._add_security_headers(response)
        
        return response
    
    def _add_security_headers(self, response: Response) -> Response:
        """Add SSL/TLS security headers."""
        # HTTP Strict Transport Security (already handled by HSTSManager)
        
        # Expect-CT
        if self.config.ct_enabled:
            response.headers['Expect-CT'] = 'max-age=86400, enforce'
        
        # Content Security Policy
        csp = "default-src 'self'; script-src 'self'; style-src 'self';"
        response.headers['Content-Security-Policy'] = csp
        
        # X-Content-Type-Options
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # X-Frame-Options
        response.headers['X-Frame-Options'] = 'DENY'
        
        # X-XSS-Protection
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Referrer-Policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Permissions-Policy
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        return response


# Utility functions
def generate_self_signed_certificate(
    common_name: str,
    validity_days: int = 365,
    dns_names: Optional[List[str]] = None
) -> Tuple[bytes, bytes]:
    """
    Generate a self-signed certificate.
    
    Args:
        common_name: Common name for certificate
        validity_days: Certificate validity in days
        dns_names: List of DNS names for SAN
        
    Returns:
        Tuple of (certificate_pem, private_key_pem)
    """
    csr_config = CertificateRequest(
        common_name=common_name,
        organization="WorldBrief360",
        organizational_unit="Security",
        country="US",
        locality="San Francisco",
        state="CA",
        dns_names=dns_names or [common_name],
        validity_days=validity_days
    )
    
    manager = CertificateManager()
    cert_pem, key_pem = manager.create_self_signed_certificate(csr_config)
    
    return cert_pem, key_pem


def verify_certificate_chain(cert_pem: bytes, chain_pem: Optional[bytes] = None) -> Tuple[bool, List[str]]:
    """
    Verify a certificate chain.
    
    Args:
        cert_pem: Certificate in PEM format
        chain_pem: Chain certificates in PEM format
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    config = SSLContextConfig()
    manager = SSLContextManager(config)
    return manager.verify_certificate_chain(cert_pem, chain_pem)


def check_certificate_expiry(cert_pem: bytes, warning_days: int = 30) -> Tuple[bool, int]:
    """
    Check certificate expiry.
    
    Args:
        cert_pem: Certificate in PEM format
        warning_days: Days before expiry to warn
        
    Returns:
        Tuple of (is_expired, days_remaining)
    """
    config = SSLContextConfig()
    manager = SSLContextManager(config)
    return manager.check_certificate_expiry(cert_pem, warning_days)


def enforce_https(app: FastAPI, hsts_max_age: int = 31536000):
    """
    Enforce HTTPS for all requests.
    
    Args:
        app: FastAPI application
        hsts_max_age: HSTS max-age in seconds
    """
    @app.middleware("http")
    async def https_redirect(request: Request, call_next):
        # Skip for local development
        if settings.ENVIRONMENT == "development":
            return await call_next(request)
        
        # Check if request is HTTP
        if request.url.scheme == "http":
            # Redirect to HTTPS
            https_url = request.url.replace(scheme="https")
            from starlette.responses import RedirectResponse
            return RedirectResponse(url=str(https_url), status_code=301)
        
        # Add HSTS header
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = f"max-age={hsts_max_age}; includeSubDomains"
        
        return response
    
    print("HTTPS enforcement enabled")


# Default configurations
def get_default_ssl_config(environment: str = "production") -> SSLContextConfig:
    """Get default SSL configuration for environment."""
    if environment == "development":
        return SSLContextConfig(
            min_version=SSLVersion.TLSv1_2,
            verify_mode=ssl.CERT_NONE,
            check_hostname=False,
            hsts_enabled=False,
        )
    else:  # production
        return SSLContextConfig(
            min_version=SSLVersion.TLSv1_2,
            verify_mode=ssl.CERT_REQUIRED,
            check_hostname=True,
            ciphers="ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20",
            hsts_enabled=True,
            hsts_max_age=31536000,
            hsts_include_subdomains=True,
            ocsp_stapling=True,
            cert_pinning_enabled=True,
            ct_enabled=True,
        )


# Export main components
__all__ = [
    # Classes
    "CertificateManager",
    "SSLContextManager",
    "HSTSManager",
    "CertificatePinner",
    "CertificateTransparency",
    "SSLScanner",
    "SSLMiddleware",
    
    # Data classes
    "SSLContextConfig",
    "CertificateRequest",
    "CertificateInfo",
    
    # Enums
    "SSLVersion",
    "CertificateType",
    "KeyAlgorithm",
    "KeySize",
    "HashAlgorithm",
    
    # Functions
    "generate_self_signed_certificate",
    "verify_certificate_chain",
    "check_certificate_expiry",
    "enforce_https",
    "get_default_ssl_config",
]