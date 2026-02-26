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
