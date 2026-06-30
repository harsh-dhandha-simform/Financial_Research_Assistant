import os
import hashlib
from typing import Optional

import chainlit as cl
from chainlit.data import get_data_layer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + ':' + key.hex()

def verify_password(password: str, hashed: str) -> bool:
    try:
        salt_hex, key_hex = hashed.split(':')
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return key.hex() == key_hex
    except Exception:
        return False


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Local password authentication callback for Chainlit."""
    dl = get_data_layer()
    
    # Check against Supabase / SQLAlchemy database
    if isinstance(dl, SQLAlchemyDataLayer):
        user = await dl.get_user(username)
        if user and user.metadata:
            hashed_password = user.metadata.get("password_hash")
            if hashed_password and verify_password(password, hashed_password):
                return cl.User(identifier=user.identifier, metadata=user.metadata)
            
    # Fallback to env admin (useful for first-time setup or bypassing DB)
    env_user = os.environ.get("CHAINLIT_AUTH_USER")
    env_pass = os.environ.get("CHAINLIT_AUTH_PASSWORD")
    if env_user and env_pass and username == env_user and password == env_pass:
        return cl.User(
            identifier=username,
            metadata={"role": "admin", "provider": "env"}
        )
        
    return None
