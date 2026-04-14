# src/auth.py

import json
import os
import hashlib

USERS_FILE = "data/users.json"
PENDING_FILE = "data/pending_requests.json"


# ---------------------------
# INITIAL SETUP
# ---------------------------

def init_files():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(USERS_FILE):
        users = {
            "admin": {
                "password": hash_password("admin123"),
                "role": "admin"
            }
        }
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

    if not os.path.exists(PENDING_FILE):
        with open(PENDING_FILE, "w") as f:
            json.dump([], f)


# ---------------------------
# HASHING
# ---------------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------------------
# LOAD / SAVE
# ---------------------------

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


def load_pending():
    with open(PENDING_FILE, "r") as f:
        return json.load(f)


def save_pending(pending):
    with open(PENDING_FILE, "w") as f:
        json.dump(pending, f, indent=4)


# ---------------------------
# LOGIN
# ---------------------------

def login(username, password):
    users = load_users()

    if username in users:
        if users[username]["password"] == hash_password(password):
            return True, users[username]["role"]

    return False, None


# ---------------------------
# SIGNUP REQUEST
# ---------------------------

def signup_request(username, password):
    users = load_users()
    pending = load_pending()

    if username in users:
        return "User already exists"

    for req in pending:
        if req["username"] == username:
            return "Request already pending"

    pending.append({
        "username": username,
        "password": hash_password(password),
        "role": "employee"
    })

    save_pending(pending)
    return "Request submitted for approval"


# ---------------------------
# ADMIN FUNCTIONS
# ---------------------------

def get_pending_requests():
    return load_pending()


def approve_user(username, role="employee"):
    users = load_users()
    pending = load_pending()

    for req in pending:
        if req["username"] == username:
            users[username] = {
                "password": req["password"],
                "role": role
            }
            pending.remove(req)
            break

    save_users(users)
    save_pending(pending)


def reject_user(username):
    pending = load_pending()
    pending = [req for req in pending if req["username"] != username]
    save_pending(pending)