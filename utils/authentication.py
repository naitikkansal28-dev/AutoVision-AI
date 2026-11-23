"""
AutoVision AI - Authentication & Access Control System
Provides secure password hashing (PBKDF2-HMAC-SHA256 with salt),
session state management, user role verification, and login/logout handlers.
"""

import hashlib
import os
import secrets
import streamlit as st
import config
from database.connection import fetch_dataframe, execute_query


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with a secure salt.
    Returns (password_hash_hex, salt_hex).
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    hash_bytes = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        iterations=100_000
    )
    return hash_bytes.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a plain password against the stored hash and salt."""
    computed_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(computed_hash, stored_hash)


def authenticate_user(username: str, password: str):
    """
    Authenticates user against database records.
    Returns user dict if valid, None otherwise.
    """
    sql = "SELECT id, username, password_hash, salt, full_name, email, role, is_active FROM users WHERE username = %s"
    df = fetch_dataframe(sql, (username,))
    
    if df.empty:
        return None
        
    user = df.iloc[0].to_dict()
    
    if not user.get("is_active", 1):
        return None
        
    if verify_password(password, user["password_hash"], user["salt"]):
        return {
            "id": user["id"],
            "username": user["username"],
            "full_name": user["full_name"],
            "email": user["email"],
            "role": user["role"]
        }
    return None


def init_session_state():
    """Initializes authentication variables in Streamlit session_state."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "role" not in st.session_state:
        st.session_state["role"] = None


def login(username, password):
    """Handles login logic and updates session state."""
    user = authenticate_user(username, password)
    if user:
        st.session_state["authenticated"] = True
        st.session_state["user"] = user
        st.session_state["role"] = user["role"]
        return True
    return False


def logout():
    """Logs out current user and resets session state."""
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.rerun()


def check_permission(required_role: str) -> bool:
    """
    Verifies if current user has the required permission level.
    Admin has full access; Manager has operational access.
    """
    current_role = st.session_state.get("role")
    if not current_role:
        return False
    if current_role == "Admin":
        return True
    return current_role == required_role
