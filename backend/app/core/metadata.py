"""
Application metadata and information.
"""

import platform
import sys
import socket
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import importlib.metadata

from .config import get_config, Environment
from .version import __version__
from .constants import AppConstants


class SystemInfo:
    """System information collector."""
    
    @staticmethod
    def get_python_info() -> Dict[str, Any]:
        """Get Python runtime information."""
        return {
            "version": sys.version,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
                "releaselevel": sys.version_info.releaselevel,
            },
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
        }
    
    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """Get platform information."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "node": platform.node(),
        }
    
    @staticmethod
    def get_network_info() -> Dict[str, Any]:
        """Get network information."""
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            return {
                "hostname": hostname,
                "ip_address": ip_address,
                "fqdn": socket.getfqdn(),
            }
        except Exception:
            return {"hostname": "unknown", "ip_address": "unknown"}
    
    @staticmethod
    def get_package_info() -> Dict[str, str]:
        """Get installed package information."""
        packages = {}
        try:
            # Get app package info
            packages["app"] = __version__
            
            # Get dependencies from pyproject.toml or requirements.txt
            # This is a simplified version - you might want to load from actual files
            for package in ["fastapi", "pydantic", "redis", "sqlalchemy", "aioredis"]:
                try:
                    version = importlib.metadata.version(package)
                    packages[package] = version
                except importlib.metadata.PackageNotFoundError:
                    packages[package] = "not installed"
        except Exception:
            pass
        
        return packages


class AppMetadata:
    """
    Application metadata container.
    
    Attributes:
        name: Application name
        version: Application version
        description: Application description
        environment: Current environment
        start_time: When the application started
        system_info: System information
        config_summary: Configuration summary
    """
    
    def __init__(self):
        self.name = AppConstants.APP_NAME
        self.version = __version__
        self.description = AppConstants.APP_DESCRIPTION
        self.environment: Optional[Environment] = None
        self.start_time = datetime.now()
        self.system_info: Dict[str, Any] = {}
        self.config_summary: Dict[str, Any] = {}
        
        self._collect_metadata()
    
    def _collect_metadata(self) -> None:
        """Collect all metadata."""
        config = get_config()
        self.environment = config.environment
        
        # System information
        self.system_info = {
            "python": SystemInfo.get_python_info(),
            "platform": SystemInfo.get_platform_info(),
            "network": SystemInfo.get_network_info(),
            "packages": SystemInfo.get_package_info(),
        }
        
        # Configuration summary (excluding sensitive data)
        self.config_summary = {
            "app_name": config.app_name,
            "environment": config.environment.value,
            "debug": config.debug,
            "host": config.host,
            "port": config.port,
            "cache_backend": config.cache.backend.value,
            "database_type": config.database.type.value,
            "log_level": config.logging.level.value,
        }
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert metadata to dictionary.
        
        Args:
            include_sensitive: Include sensitive configuration data
            
        Returns:
            Dictionary representation
        """
        data = {
            "application": {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "environment": self.environment.value if self.environment else None,
                "uptime": str(datetime.now() - self.start_time),
                "start_time": self.start_time.isoformat(),
            },
            "system": self.system_info,
            "config": self.config_summary,
        }
        
        if include_sensitive:
            config = get_config()
            data["config"].update({
                "cache": {
                    "redis_url": config.cache.redis_url,
                    "default_ttl": config.cache.default_ttl,
                },
                "database": {
                    "host": config.database.host,
                    "port": config.database.port,
                    "name": config.database.name,
                },
                "security": {
                    "algorithm": config.security.algorithm,
                    "access_token_expire_minutes": config.security.access_token_expire_minutes,
                },
            })
        
        return data
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get application health status.
        
        Returns:
            Health status dictionary
        """
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
        }
        
        # System checks
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                status["checks"]["python_version"] = {
                    "status": "warning",
                    "message": f"Python {sys.version} is below recommended 3.8+",
                }
            else:
                status["checks"]["python_version"] = {
                    "status": "healthy",
                    "message": f"Python {sys.version}",
                }
            
            # Check disk space (simplified)
            try:
                disk_usage = Path.cwd().stat()
                status["checks"]["disk"] = {
                    "status": "healthy",
                    "message": "Disk accessible",
                }
            except Exception as e:
                status["checks"]["disk"] = {
                    "status": "unhealthy",
                    "message": f"Disk error: {e}",
                }
            
            # Check event loop
            try:
                loop = asyncio.get_event_loop()
                status["checks"]["event_loop"] = {
                    "status": "healthy",
                    "message": f"Event loop running: {type(loop).__name__}",
                }
            except Exception as e:
                status["checks"]["event_loop"] = {
                    "status": "unhealthy",
                    "message": f"Event loop error: {e}",
                }
            
            # Update overall status if any check failed
            for check in status["checks"].values():
                if check["status"] == "unhealthy":
                    status["status"] = "unhealthy"
                elif check["status"] == "warning" and status["status"] == "healthy":
                    status["status"] = "degraded"
                    
        except Exception as e:
            status["status"] = "unhealthy"
            status["error"] = str(e)
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get application metrics.
        
        Returns:
            Metrics dictionary
        """
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        return {
            "memory": {
                "rss": process.memory_info().rss,  # Resident Set Size
                "vms": process.memory_info().vms,  # Virtual Memory Size
                "percent": process.memory_percent(),
            },
            "cpu": {
                "percent": process.cpu_percent(interval=0.1),
                "count": psutil.cpu_count(),
                "load": psutil.getloadavg(),
            },
            "python": {
                "gc_count": gc.get_count(),
                "gc_threshold": gc.get_threshold(),
            },
            "threads": process.num_threads(),
            "connections": len(process.connections()) if hasattr(process, 'connections') else 0,
        }


# Singleton instance
_app_metadata: Optional[AppMetadata] = None


def get_app_metadata() -> AppMetadata:
    """
    Get or create application metadata instance.
    
    Returns:
        AppMetadata instance
    """
    global _app_metadata
    if _app_metadata is None:
        _app_metadata = AppMetadata()
    return _app_metadata


# Convenience functions
def get_metadata_dict(include_sensitive: bool = False) -> Dict[str, Any]:
    """Get metadata as dictionary."""
    return get_app_metadata().to_dict(include_sensitive)


def get_health_status() -> Dict[str, Any]:
    """Get health status."""
    return get_app_metadata().get_health_status()


def get_metrics() -> Dict[str, Any]:
    """Get application metrics."""
    return get_app_metadata().get_metrics()


def get_start_time() -> datetime:
    """Get application start time."""
    return get_app_metadata().start_time


def get_uptime() -> str:
    """Get formatted uptime."""
    uptime = datetime.now() - get_app_metadata().start_time
    return str(uptime)