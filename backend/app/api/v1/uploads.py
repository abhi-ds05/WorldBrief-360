"""
File upload endpoints for incidents, profile images, and other media.
"""
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.core.security import get_current_user
from app.db.models import User, IncidentImage, ProfileImage
from app.schemas import UploadResponse, FileInfo
from app.services.utils.file_manager import (
    save_uploaded_file,
    validate_file_type,
    get_file_info,
    delete_file,
    generate_presigned_url,
)

router = APIRouter()


@router.post("/incident", response_model=UploadResponse)
async def upload_incident_image(
    incident_id: int = Form(...),
    file: UploadFile = File(...),
    is_primary: bool = Form(False),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UploadResponse:
    """
    Upload an image for an incident.
    """
    # Validate incident exists and user has permission
    from app.db.models import Incident
    incident = db.query(Incident).filter(
        Incident.id == incident_id,
        Incident.is_deleted == False,
    ).first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Check permissions
    if incident.reporter_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Not authorized to upload images for this incident"
        )
    
    # Validate file type
    if not validate_file_type(file, allowed_types=["image/jpeg", "image/png", "image/gif", "image/webp"]):
        raise HTTPException(
            status_code=400,
            detail="Only image files (JPEG, PNG, GIF, WebP) are allowed"
        )
    
    # Validate file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
        )
    
    # Save file
    try:
        file_path = await save_uploaded_file(
            file=file,
            user_id=current_user.id,
            prefix=f"incident_{incident_id}",
            subdirectory="incidents"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Create database record
    incident_image = IncidentImage(
        incident_id=incident_id,
        image_path=file_path,
        uploaded_by=current_user.id,
        is_primary=is_primary,
        description=description,
    )
    
    db.add(incident_image)
    
    # If this is primary, unset other primary images
    if is_primary:
        db.query(IncidentImage).filter(
            IncidentImage.incident_id == incident_id,
            IncidentImage.is_primary == True,
            IncidentImage.id != incident_image.id,  # Will be None until commit
        ).update({"is_primary": False})
    
    db.commit()
    db.refresh(incident_image)
    
    # Get file info
    file_info = get_file_info(file_path)
    
    return UploadResponse(
        success=True,
        message="File uploaded successfully",
        file_id=incident_image.id,
        file_path=file_path,
        file_info=file_info,
        incident_id=incident_id,
        is_primary=is_primary,
    )


@router.post("/profile", response_model=UploadResponse)
async def upload_profile_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UploadResponse:
    """
    Upload a profile image.
    """
    # Validate file type
    if not validate_file_type(file, allowed_types=["image/jpeg", "image/png", "image/gif"]):
        raise HTTPException(
            status_code=400,
            detail="Only image files (JPEG, PNG, GIF) are allowed"
        )
    
    # Validate file size (max 5MB for profile)
    max_size = 5 * 1024 * 1024  # 5MB
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
        )
    
    # Delete old profile image if exists
    old_image = db.query(ProfileImage).filter(
        ProfileImage.user_id == current_user.id,
        ProfileImage.is_active == True,
    ).first()
    
    if old_image:
        try:
            delete_file(old_image.image_path)
        except Exception as e:
            print(f"Failed to delete old profile image: {e}")
        
        old_image.is_active = False
        old_image.updated_at = datetime.utcnow()
    
    # Save new file
    try:
        file_path = await save_uploaded_file(
            file=file,
            user_id=current_user.id,
            prefix="profile",
            subdirectory="profiles"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Create profile image record
    profile_image = ProfileImage(
        user_id=current_user.id,
        image_path=file_path,
        is_active=True,
    )
    
    db.add(profile_image)
    db.commit()
    db.refresh(profile_image)
    
    # Update user profile image reference
    current_user.profile_image_id = profile_image.id
    db.commit()
    
    # Get file info
    file_info = get_file_info(file_path)
    
    return UploadResponse(
        success=True,
        message="Profile image uploaded successfully",
        file_id=profile_image.id,
        file_path=file_path,
        file_info=file_info,
        user_id=current_user.id,
    )


@router.post("/briefing", response_model=UploadResponse)
async def upload_briefing_asset(
    briefing_id: int = Form(...),
    asset_type: str = Form(..., description="Type of asset: image, audio, document"),
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UploadResponse:
    """
    Upload an asset for a briefing (images, audio, etc.).
    """
    from app.db.models import Briefing
    briefing = db.query(Briefing).filter(
        Briefing.id == briefing_id,
        Briefing.user_id == current_user.id,
    ).first()
    
    if not briefing:
        raise HTTPException(status_code=404, detail="Briefing not found")
    
    # Validate asset type
    valid_types = {
        "image": ["image/jpeg", "image/png", "image/gif", "image/webp", "image/svg+xml"],
        "audio": ["audio/mpeg", "audio/wav", "audio/ogg", "audio/webm"],
        "document": ["application/pdf", "text/plain", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
    }
    
    if asset_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid asset type. Must be one of: {', '.join(valid_types.keys())}"
        )
    
    # Validate file type
    if not validate_file_type(file, allowed_types=valid_types[asset_type]):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type for {asset_type}. Allowed types: {', '.join(valid_types[asset_type])}"
        )
    
    # Validate file size based on type
    max_sizes = {
        "image": 10 * 1024 * 1024,  # 10MB
        "audio": 50 * 1024 * 1024,  # 50MB
        "document": 20 * 1024 * 1024,  # 20MB
    }
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > max_sizes[asset_type]:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size for {asset_type} is {max_sizes[asset_type] // (1024*1024)}MB"
        )
    
    # Save file
    try:
        file_path = await save_uploaded_file(
            file=file,
            user_id=current_user.id,
            prefix=f"briefing_{briefing_id}_{asset_type}",
            subdirectory=f"briefings/{asset_type}s"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Create record in appropriate table based on asset type
    if asset_type == "image":
        from app.db.models import BriefingImage
        asset_record = BriefingImage(
            briefing_id=briefing_id,
            image_path=file_path,
            description=description,
        )
    elif asset_type == "audio":
        from app.db.models import BriefingAudio
        asset_record = BriefingAudio(
            briefing_id=briefing_id,
            audio_path=file_path,
            description=description,
        )
    else:  # document
        from app.db.models import BriefingDocument
        asset_record = BriefingDocument(
            briefing_id=briefing_id,
            document_path=file_path,
            description=description,
        )
    
    db.add(asset_record)
    db.commit()
    db.refresh(asset_record)
    
    # Get file info
    file_info = get_file_info(file_path)
    
    return UploadResponse(
        success=True,
        message=f"{asset_type.capitalize()} uploaded successfully",
        file_id=asset_record.id,
        file_path=file_path,
        file_info=file_info,
        briefing_id=briefing_id,
        asset_type=asset_type,
    )


@router.get("/{file_path:path}")
async def get_file(
    file_path: str,
    download: bool = False,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Get a file by path. Supports authentication and authorization.
    """
    # Security: Prevent directory traversal
    if ".." in file_path or file_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    # Construct full path
    full_path = Path(settings.UPLOAD_DIR) / file_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check permissions based on file type and location
    if not await check_file_permissions(file_path, current_user, db):
        raise HTTPException(status_code=403, detail="Not authorized to access this file")
    
    # Serve file
    if download:
        return FileResponse(
            path=full_path,
            filename=full_path.name,
            media_type="application/octet-stream"
        )
    else:
        # Guess media type from extension
        import mimetypes
        media_type, _ = mimetypes.guess_type(str(full_path))
        if not media_type:
            media_type = "application/octet-stream"
        
        return FileResponse(
            path=full_path,
            media_type=media_type
        )


async def check_file_permissions(
    file_path: str,
    user: Optional[User],
    db: Session
) -> bool:
    """
    Check if user has permission to access a file.
    """
    # Public files (e.g., topic images, public incident images)
    if file_path.startswith("public/"):
        return True
    
    # Profile images - only accessible by the user
    if file_path.startswith("profiles/"):
        if not user:
            return False
        
        # Extract user ID from filename (profile_{user_id}_*)
        import re
        match = re.search(r'profile_(\d+)_', file_path)
        if match:
            file_user_id = int(match.group(1))
            return user.id == file_user_id or user.role == "admin"
    
    # Incident images - check incident permissions
    if file_path.startswith("incidents/"):
        if not user:
            return False
        
        # Extract incident ID from filename
        import re
        match = re.search(r'incident_(\d+)_', file_path)
        if match:
            incident_id = int(match.group(1))
            from app.db.models import Incident, IncidentImage
            
            # Check if user is reporter, admin, or incident is public
            incident = db.query(Incident).filter(Incident.id == incident_id).first()
            if not incident:
                return False
            
            if incident.reporter_id == user.id or user.role == "admin":
                return True
            
            # Check if incident is public/verified
            if incident.verification_status == "verified":
                return True
    
    # Briefing files - check briefing ownership
    if file_path.startswith("briefings/"):
        if not user:
            return False
        
        # Extract briefing ID from filename
        import re
        match = re.search(r'briefing_(\d+)_', file_path)
        if match:
            briefing_id = int(match.group(1))
            from app.db.models import Briefing
            
            briefing = db.query(Briefing).filter(Briefing.id == briefing_id).first()
            if not briefing:
                return False
            
            if briefing.user_id == user.id or user.role == "admin" or briefing.is_public:
                return True
    
    # Default: require authentication
    return user is not None


@router.delete("/{file_id}")
async def delete_uploaded_file(
    file_id: int,
    file_type: str = Query(..., description="Type of file: incident_image, profile_image, briefing_asset"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Delete an uploaded file.
    """
    if file_type == "incident_image":
        # Get incident image
        incident_image = db.query(IncidentImage).filter(
            IncidentImage.id == file_id,
        ).first()
        
        if not incident_image:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check permissions
        from app.db.models import Incident
        incident = db.query(Incident).filter(Incident.id == incident_image.incident_id).first()
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        if incident.reporter_id != current_user.id and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Not authorized to delete this file")
        
        # Delete file from storage
        try:
            delete_file(incident_image.image_path)
        except Exception as e:
            print(f"Failed to delete file: {e}")
        
        # Delete database record
        db.delete(incident_image)
        db.commit()
        
        return {"message": "Incident image deleted successfully"}
    
    elif file_type == "profile_image":
        # Get profile image
        profile_image = db.query(ProfileImage).filter(
            ProfileImage.id == file_id,
            ProfileImage.user_id == current_user.id,
        ).first()
        
        if not profile_image:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete file from storage
        try:
            delete_file(profile_image.image_path)
        except Exception as e:
            print(f"Failed to delete file: {e}")
        
        # Delete database record
        db.delete(profile_image)
        
        # Clear user's profile image reference
        if current_user.profile_image_id == file_id:
            current_user.profile_image_id = None
        
        db.commit()
        
        return {"message": "Profile image deleted successfully"}
    
    elif file_type == "briefing_asset":
        # Determine asset type and get record
        from app.db.models import BriefingImage, BriefingAudio, BriefingDocument
        
        asset = None
        for model in [BriefingImage, BriefingAudio, BriefingDocument]:
            asset = db.query(model).filter(model.id == file_id).first()
            if asset:
                asset_type = model.__name__.lower().replace("briefing", "")
                break
        
        if not asset:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check briefing ownership
        from app.db.models import Briefing
        briefing = db.query(Briefing).filter(Briefing.id == asset.briefing_id).first()
        if not briefing:
            raise HTTPException(status_code=404, detail="Briefing not found")
        
        if briefing.user_id != current_user.id and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Not authorized to delete this file")
        
        # Get file path
        file_path = getattr(asset, f"{asset_type}_path")
        
        # Delete file from storage
        try:
            delete_file(file_path)
        except Exception as e:
            print(f"Failed to delete file: {e}")
        
        # Delete database record
        db.delete(asset)
        db.commit()
        
        return {"message": f"Briefing {asset_type} deleted successfully"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")


@router.get("/presigned/url")
async def get_presigned_url(
    file_path: str,
    expires_in: int = Query(3600, ge=60, le=86400, description="URL expiry time in seconds"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Get a presigned URL for accessing a file (for S3 storage).
    """
    # Check permissions
    if not await check_file_permissions(file_path, current_user, db):
        raise HTTPException(status_code=403, detail="Not authorized to access this file")
    
    try:
        url = generate_presigned_url(file_path, expires_in)
        return {"url": url, "expires_in": expires_in}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate presigned URL: {str(e)}"
        )


@router.get("/user/files", response_model=List[FileInfo])
async def get_user_files(
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[FileInfo]:
    """
    Get all files uploaded by the current user.
    """
    files = []
    
    # Get incident images
    if not file_type or file_type == "incident_image":
        incident_images = db.query(IncidentImage).join(
            db.models.Incident, db.models.Incident.id == IncidentImage.incident_id
        ).filter(
            db.models.Incident.reporter_id == current_user.id,
        ).offset(offset).limit(limit).all()
        
        for img in incident_images:
            file_info = get_file_info(img.image_path)
            if file_info:
                files.append(file_info)
    
    # Get profile images
    if not file_type or file_type == "profile_image":
        profile_images = db.query(ProfileImage).filter(
            ProfileImage.user_id == current_user.id,
        ).offset(offset).limit(limit).all()
        
        for img in profile_images:
            file_info = get_file_info(img.image_path)
            if file_info:
                files.append(file_info)
    
    return files


@router.post("/validate")
async def validate_file(
    file: UploadFile = File(...),
    max_size_mb: int = Form(10),
    allowed_types: List[str] = Form(["image/jpeg", "image/png", "application/pdf"]),
) -> Dict[str, Any]:
    """
    Validate a file without uploading it.
    """
    max_size = max_size_mb * 1024 * 1024
    
    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > max_size:
        return {
            "valid": False,
            "error": f"File too large. Maximum size is {max_size_mb}MB",
            "file_size": file_size,
            "max_size": max_size,
        }
    
    # Check file type
    if not validate_file_type(file, allowed_types):
        return {
            "valid": False,
            "error": f"Invalid file type. Allowed types: {', '.join(allowed_types)}",
            "file_type": file.content_type,
            "allowed_types": allowed_types,
        }
    
    # Get basic file info
    import magic
    file_bytes = await file.read(1024)
    file.file.seek(0)
    
    mime = magic.Magic(mime=True)
    detected_type = mime.from_buffer(file_bytes)
    
    return {
        "valid": True,
        "filename": file.filename,
        "content_type": file.content_type,
        "detected_type": detected_type,
        "file_size": file_size,
        "max_size": max_size,
        "allowed_types": allowed_types,
    }