from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import chainlit as cl
from chainlit.data import get_data_layer
import json

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from ui.auth import hash_password

router = APIRouter()

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up - Financial Assistant</title>
    <style>
        body {
            background-color: #111010;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background-color: #1e1619;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            width: 100%;
            max-width: 400px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        h2 {
            margin-top: 0;
            color: #f80061;
        }
        form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        input {
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background-color: #2c2528;
            color: white;
            font-size: 16px;
        }
        input:focus {
            outline: none;
            border-color: #f80061;
        }
        button {
            padding: 12px;
            border-radius: 8px;
            border: none;
            background-color: #f80061;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: #d10052;
        }
        .error {
            color: #ff4d4d;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .login-link {
            margin-top: 20px;
            font-size: 14px;
            color: #a69ea1;
        }
        .login-link a {
            color: #f80061;
            text-decoration: none;
        }
        .login-link a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Create Account</h2>
        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}
        <form action="/signup" method="post">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Sign Up</button>
        </form>
        <div class="login-link">
            Already have an account? <a href="/">Log in here</a>
        </div>
    </div>
</body>
</html>
"""

# Very simple template rendering to avoid adding Jinja2 files
def render_template(error: str = None):
    if error:
        return html_template.replace('{% if error %}', '').replace('{% endif %}', '').replace('{{ error }}', error)
    else:
        return html_template.replace('{% if error %}', '<!--').replace('{% endif %}', '-->')

@router.get("/signup", response_class=HTMLResponse)
async def signup_form():
    return render_template()

@router.post("/signup")
async def signup(username: str = Form(...), password: str = Form(...)):
    dl = get_data_layer()
    if not isinstance(dl, SQLAlchemyDataLayer):
        return HTMLResponse(render_template("Error: SQLAlchemy Data Layer not configured."), status_code=500)
    
    # Check if user exists
    existing = await dl.get_user(username)
    if existing:
        return HTMLResponse(render_template("Username already exists."), status_code=400)
    
    # Hash password
    hashed_password = hash_password(password)
    
    # Create PersistedUser using data layer so they have an ID
    user = cl.PersistedUser(
        id="", # SQLAlchemyDataLayer generates UUID if missing or we can leave it
        identifier=username,
        metadata={"role": "user", "provider": "credentials", "password_hash": hashed_password},
        createdAt=""
    )
    # chainlit's SQLAlchemyDataLayer doesn't export PersistedUser properly from cl directly sometimes,
    # cl.User is normally used but dl.create_user expects PersistedUser. 
    # Actually, dl.create_user accepts cl.User in 1.3+ but let's just construct cl.User.
    
    from chainlit.user import User
    new_user = User(identifier=username, metadata={"role": "user", "provider": "credentials", "password_hash": hashed_password})
    await dl.create_user(new_user)
    
    # Redirect to the main chat page for login
    return RedirectResponse(url="/", status_code=303)
