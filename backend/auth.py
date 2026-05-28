"""
Authentication helpers: bcrypt password hashing and JWT issuance/verification.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_HOURS
from database import get_db
from models import AdminUser


# Tells FastAPI/Swagger where to obtain a token (used for the lock icon UX).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/login", auto_error=False)


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------
def hash_password(plain_password: str) -> str:
    """Hash a plain password with bcrypt and return the UTF-8 string."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Constant-time compare of plain password to bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def create_access_token(
    subject: str,
    role: str = "admin",
    expires_hours: Optional[int] = None,
) -> str:
    """Encode a JWT with username as 'sub' and role as 'role'."""
    expire = datetime.now(timezone.utc) + timedelta(
        hours=expires_hours or JWT_EXPIRE_HOURS
    )
    payload = {
        "sub": subject,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and verify a JWT. Raises jwt exceptions on failure."""
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------------------------
# FastAPI dependency: require a valid admin JWT
# ---------------------------------------------------------------------------
async def get_current_admin(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> AdminUser:
    """Return the AdminUser for the bearer token, or raise 401."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Your session has expired. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    try:
        payload = decode_access_token(token)
        username: Optional[str] = payload.get("sub")
        if not username:
            raise credentials_exception
    except jwt.ExpiredSignatureError:
        raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    result = await db.execute(
        select(AdminUser).where(AdminUser.username == username)
    )
    admin = result.scalar_one_or_none()
    if admin is None:
        raise credentials_exception
    return admin
