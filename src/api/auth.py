"""Authentication API routes."""

from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
import jwt

from sqlalchemy.orm import Session
from db import get_db, User, SubscriptionTier
from config.settings import get_settings
from utils.validators import validate_password
from api.rate_limiter import limiter

# Configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24 * 7  # 7 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer(auto_error=False)

# Router
router = APIRouter(prefix="/auth", tags=["auth"])


# Pydantic models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    subscription_tier: str
    copies_used_this_month: int
    monthly_limit: int
    remaining_copies: int


# Helper functions
def hash_password(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: str) -> str:
    """Create a JWT access token."""
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token."""
    from loguru import logger
    try:
        settings = get_settings()
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError as e:
        logger.warning(f"Token expired: {e}")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e.__class__.__name__}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected token decode error: {e.__class__.__name__}: {e}")
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user."""
    from loguru import logger

    if credentials is None:
        logger.warning("Auth failed: No credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Non authentifié",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    logger.debug(f"Auth: Token received (first 20 chars): {token[:20]}...")

    payload = decode_token(token)

    if payload is None:
        logger.warning("Auth failed: Token decode returned None (expired or invalid)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if user_id is None:
        logger.warning(f"Auth failed: No 'sub' in payload. Payload keys: {list(payload.keys())}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        logger.warning(f"Auth failed: User not found in DB. user_id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur non trouvé",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.debug(f"Auth success: user_id={user_id}, email={user.email}")

    # Set Sentry user context
    from middleware.error_handler import set_user_context
    set_user_context(user_id=user.id, email=user.email, username=user.name)

    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get the current user if authenticated, otherwise return None."""
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


async def get_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get the current user if they are an admin."""
    from db import SubscriptionTier

    if current_user.subscription_tier != SubscriptionTier.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé. Privilèges administrateur requis.",
        )
    return current_user


# Routes
@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def register(
    request: Request,
    user_data: UserRegister,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    # Validate password
    is_valid, error_msg = validate_password(user_data.password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Check if email already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        # Per CONTEXT.md: Specific error acceptable for signup
        raise HTTPException(status_code=400, detail="Email déjà utilisé")

    # Create user
    user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        name=user_data.name,
        subscription_tier=SubscriptionTier.FREE,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    # Create token
    access_token = create_access_token(user.id)

    return Token(
        access_token=access_token,
        user=user.to_dict()
    )


@router.post("/login", response_model=Token)
@limiter.limit("10/minute")
async def login(
    request: Request,
    credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """Login and get an access token."""
    # Find user
    user = db.query(User).filter(User.email == credentials.email).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect"
        )

    # Verify password
    if not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect"
        )

    # Create token
    access_token = create_access_token(user.id)

    return Token(
        access_token=access_token,
        user=user.to_dict()
    )


@router.post("/logout")
async def logout():
    """
    Logout endpoint.

    Per CONTEXT.md: Client-side logout only (delete token from browser).
    Server returns 200 OK to confirm logout request received.
    """
    return {"message": "Déconnexion réussie"}


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user)
):
    """Get the current user's profile."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        subscription_tier=current_user.subscription_tier.value,
        copies_used_this_month=current_user.copies_used_this_month,
        monthly_limit=current_user.get_monthly_limit(),
        remaining_copies=current_user.remaining_copies,
    )
