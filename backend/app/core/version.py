"""
Version information for the application.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Version should be updated according to semantic versioning
# Major.Minor.Patch
__version__ = "1.0.0"


class VersionInfo:
    """
    Detailed version information.
    
    Attributes:
        version: Version string
        major: Major version number
        minor: Minor version number
        patch: Patch version number
        prerelease: Prerelease identifier
        build: Build metadata
        commit_hash: Git commit hash
        commit_date: Git commit date
        build_date: Build date
        python_version: Python version
    """
    
    def __init__(self, version: str = __version__):
        """
        Initialize version info.
        
        Args:
            version: Version string
        """
        self.version = version
        self.major, self.minor, self.patch = self._parse_version(version)
        self.prerelease: Optional[str] = None
        self.build: Optional[str] = None
        
        # Parse prerelease and build metadata if present
        if "-" in version:
            version, extra = version.split("-", 1)
            if "+" in extra:
                self.prerelease, self.build = extra.split("+", 1)
            else:
                self.prerelease = extra
        elif "+" in version:
            version, self.build = version.split("+", 1)
        
        # Git info
        self.commit_hash = self._get_git_commit_hash()
        self.commit_date = self._get_git_commit_date()
        
        # Build info
        self.build_date = datetime.now().isoformat()
        self.python_version = sys.version
        
        # Additional metadata
        self._additional_info: Dict[str, Any] = {}
    
    def _parse_version(self, version_str: str) -> Tuple[int, int, int]:
        """
        Parse version string into components.
        
        Args:
            version_str: Version string
            
        Returns:
            Tuple of (major, minor, patch)
            
        Raises:
            ValueError: If version string is invalid
        """
        # Remove prerelease and build metadata
        if "-" in version_str:
            version_str = version_str.split("-")[0]
        if "+" in version_str:
            version_str = version_str.split("+")[0]
        
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")
        
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            raise ValueError(f"Invalid version string: {version_str}")
    
    def _get_git_commit_hash(self) -> Optional[str]:
        """
        Get current Git commit hash.
        
        Returns:
            Commit hash or None if not available
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _get_git_commit_date(self) -> Optional[str]:
        """
        Get current Git commit date.
        
        Returns:
            Commit date or None if not available
        """
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%cd", "--date=iso"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def add_info(self, key: str, value: Any) -> None:
        """
        Add additional version information.
        
        Args:
            key: Information key
            value: Information value
        """
        self._additional_info[key] = value
    
    def to_dict(self, include_git: bool = True) -> Dict[str, Any]:
        """
        Convert version info to dictionary.
        
        Args:
            include_git: Include Git information
            
        Returns:
            Dictionary representation
        """
        data = {
            "version": self.version,
            "semantic": {
                "major": self.major,
                "minor": self.minor,
                "patch": self.patch,
            },
            "build": {
                "date": self.build_date,
                "python_version": self.python_version,
            },
        }
        
        if self.prerelease:
            data["semantic"]["prerelease"] = self.prerelease
        
        if self.build:
            data["build"]["metadata"] = self.build
        
        if include_git and self.commit_hash:
            data["git"] = {
                "commit_hash": self.commit_hash,
                "commit_date": self.commit_date,
            }
        
        if self._additional_info:
            data["additional"] = self._additional_info
        
        return data
    
    def __str__(self) -> str:
        """String representation of version."""
        parts = [f"v{self.version}"]
        
        if self.commit_hash:
            parts.append(f"git:{self.commit_hash[:8]}")
        
        if self.build_date:
            # Format date nicely
            try:
                dt = datetime.fromisoformat(self.build_date.replace('Z', '+00:00'))
                parts.append(f"built:{dt.strftime('%Y-%m-%d')}")
            except ValueError:
                parts.append(f"built:{self.build_date}")
        
        return " ".join(parts)
    
    def bump_major(self) -> str:
        """
        Bump major version.
        
        Returns:
            New version string
        """
        self.major += 1
        self.minor = 0
        self.patch = 0
        self._update_version_string()
        return self.version
    
    def bump_minor(self) -> str:
        """
        Bump minor version.
        
        Returns:
            New version string
        """
        self.minor += 1
        self.patch = 0
        self._update_version_string()
        return self.version
    
    def bump_patch(self) -> str:
        """
        Bump patch version.
        
        Returns:
            New version string
        """
        self.patch += 1
        self._update_version_string()
        return self.version
    
    def _update_version_string(self) -> None:
        """Update version string from components."""
        self.version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            self.version += f"-{self.prerelease}"
        if self.build:
            self.version += f"+{self.build}"


# Global version info instance
_version_info: Optional[VersionInfo] = None


def get_version_info() -> VersionInfo:
    """
    Get or create version info instance.
    
    Returns:
        VersionInfo instance
    """
    global _version_info
    if _version_info is None:
        _version_info = VersionInfo()
    return _version_info


def get_version() -> str:
    """
    Get version string.
    
    Returns:
        Version string
    """
    return __version__


def get_version_dict(include_git: bool = True) -> Dict[str, Any]:
    """
    Get version information as dictionary.
    
    Args:
        include_git: Include Git information
        
    Returns:
        Dictionary with version info
    """
    return get_version_info().to_dict(include_git)


def get_build_info() -> Dict[str, Any]:
    """
    Get build information.
    
    Returns:
        Dictionary with build info
    """
    version_info = get_version_info()
    return {
        "version": version_info.version,
        "build_date": version_info.build_date,
        "commit_hash": version_info.commit_hash,
        "commit_date": version_info.commit_date,
        "python_version": version_info.python_version,
    }


def check_version_compatibility(
    required_version: str,
    current_version: str = __version__
) -> bool:
    """
    Check if current version is compatible with required version.
    
    Args:
        required_version: Minimum required version
        current_version: Current version
        
    Returns:
        True if compatible
    """
    try:
        required = VersionInfo(required_version)
        current = VersionInfo(current_version)
        
        # Check major version
        if current.major != required.major:
            return False
        
        # Check minor version
        if current.minor < required.minor:
            return False
        
        # Check patch version if minor matches
        if current.minor == required.minor and current.patch < required.patch:
            return False
        
        return True
        
    except ValueError:
        return False


def format_version_banner() -> str:
    """
    Format version banner for console output.
    
    Returns:
        Formatted banner
    """
    version_info = get_version_info()
    
    banner = f"""
╔{'═' * 60}╗
║ {'My Application':^58} ║
║ {'':^58} ║
║ {'Version':>25} : {version_info.version:<30} ║
"""
    
    if version_info.commit_hash:
        banner += f"║ {'Git Commit':>25} : {version_info.commit_hash[:8]:<30} ║\n"
    
    banner += f"║ {'Python':>25} : {sys.version.split()[0]:<30} ║\n"
    banner += f"║ {'Build Date':>25} : {version_info.build_date:<30} ║\n"
    banner += f"╚{'═' * 60}╝"
    
    return banner


# Convenience function for semantic version checking
def is_version_at_least(version: str, min_version: str) -> bool:
    """
    Check if version is at least min_version.
    
    Args:
        version: Version to check
        min_version: Minimum version
        
    Returns:
        True if version >= min_version
    """
    return check_version_compatibility(min_version, version)