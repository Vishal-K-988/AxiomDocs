from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from jose.utils import base64url_decode
import requests
from typing import Optional
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Security scheme for JWT token
security = HTTPBearer()

# Clerk configuration
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
CLERK_ISSUER = os.getenv("CLERK_ISSUER", "https://clerk.your-domain.com")  # Replace with your Clerk domain

def get_clerk_jwks():
    """Fetch Clerk's JWKS (JSON Web Key Set)."""
    try:
        response = requests.get(f"{CLERK_ISSUER}/.well-known/jwks.json")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching Clerk JWKS: {e}")
        return None

def get_public_key(token: str) -> Optional[str]:
    """Get the public key for the token's key ID."""
    try:
        # Get the key ID from the token header
        header = jwt.get_unverified_header(token)
        key_id = header.get("kid")
        
        if not key_id:
            return None
            
        # Fetch JWKS
        jwks = get_clerk_jwks()
        if not jwks:
            return None
            
        # Find the key with matching key ID
        for key in jwks.get("keys", []):
            if key.get("kid") == key_id:
                return key
                
        return None
    except Exception as e:
        print(f"Error getting public key: {e}")
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Validate the JWT token from Clerk and return the user ID.
    This will be used as a dependency in protected routes.
    """
    try:
        # Get the token from the Authorization header
        token = credentials.credentials
        
        # Get the public key for this token
        public_key = get_public_key(token)
        if not public_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Verify the token
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
            options={"verify_aud": False}  # Clerk doesn't use audience
        )
        
        # Extract the user ID from the token
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return user_id
        
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Optional user dependency for routes that can work with or without authentication
async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """
    Similar to get_current_user but doesn't raise an exception if no token is provided.
    Returns None if no valid token is present.
    """
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        public_key = get_public_key(token)
        if not public_key:
            return None
            
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
            options={"verify_aud": False}
        )
        
        return payload.get("sub")
    except:
        return None 