"""Input validators for La Corrigeuse."""

from typing import Tuple
from fastapi import UploadFile, HTTPException

# NIST-aligned: length over complexity
MIN_PASSWORD_LENGTH = 8

# Common password blocklist (French context)
COMMON_PASSWORDS = {
    'password', 'password123', '12345678', 'qwerty',
    'azerty', 'admin', 'letmein', 'welcome',
    'azerty123', 'password1', '123456789'
}

# PDF magic bytes (first 4 bytes of valid PDF)
PDF_MAGIC_BYTES = b'%PDF'
MAX_PDF_SIZE_MB = 25


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate password against NIST-aligned policy.

    Returns:
        (is_valid, error_message)
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Le mot de passe doit contenir au moins {MIN_PASSWORD_LENGTH} caractères"

    if password.lower() in COMMON_PASSWORDS:
        return False, "Ce mot de passe est trop courant"

    return True, ""


async def validate_pdf_upload(
    file: UploadFile,
    max_size_mb: int = MAX_PDF_SIZE_MB
) -> bytes:
    """
    Validate PDF file type and size, return content.

    Args:
        file: The uploaded file
        max_size_mb: Maximum file size in megabytes

    Returns:
        File content as bytes

    Raises:
        HTTPException 413: File too large
        HTTPException 400: Invalid PDF format
    """
    # Check file size
    content = await file.read()
    if len(content) > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Fichier trop volumineux (max 25 MB)"
        )

    # Check magic bytes (PDF signature)
    if not content.startswith(PDF_MAGIC_BYTES):
        # Per CONTEXT.md: Generic error for security
        raise HTTPException(
            status_code=400,
            detail="Format de fichier invalide"
        )

    # Reset file pointer for further processing
    await file.seek(0)

    return content
