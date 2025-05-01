import streamlit as st
import sqlite3
import hashlib
import secrets
import smtplib
from email.mime.text import MIMEText
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Update these configuration variables
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TOKEN_EXPIRY_HOURS = 24
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501")  # Default for local dev

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  email TEXT UNIQUE, 
                  password TEXT, 
                  verified BOOLEAN DEFAULT 0)''')

    c.execute('''CREATE TABLE IF NOT EXISTS tokens
                 (token TEXT PRIMARY KEY,
                  username TEXT,
                  purpose TEXT,
                  expires DATETIME)''')
    
    conn.commit()
    return conn

conn = init_db()

# --- Helper Functions ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def generate_token(username, purpose):
    token = secrets.token_urlsafe(32)
    expires = datetime.now() + timedelta(hours=TOKEN_EXPIRY_HOURS)
    conn.cursor().execute("INSERT INTO tokens VALUES (?,?,?,?)", 
              (token, username, purpose, expires))
    conn.commit()
    return token

def send_email(to_email, subject, body):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# --- Authentication Pages ---
def show_auth_pages():
    """Main function to show auth pages"""
    query_params = st.experimental_get_query_params()
    
    # Handle verification/reset tokens
    if 'token' in query_params and 'purpose' in query_params:
        token = query_params['token'][0]
        purpose = query_params['purpose'][0]
        
        if purpose == "verify":
            verify_page(token)
        elif purpose == "reset":
            reset_password_page(token)
        return
    
    # Normal auth flow
    st.title("Lung Cancer Prediction App")
    menu = st.selectbox("Menu", ["Login", "SignUp", "Forgot Password"])
    
    if menu == "Login":
        login_page()
    elif menu == "SignUp":
        signup_page()
    else:
        forgot_password_page()

def signup_page():
    st.subheader("Create New Account")
    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')
    
    if st.button("Sign Up"):
        if not (email and username and password):
            st.warning("Please fill all fields")
        elif password != confirm_password:
            st.error("Passwords don't match")
        else:
            try:
                hashed_pwd = make_hashes(password)
                conn.cursor().execute("INSERT INTO users (username, email, password) VALUES (?,?,?)",
                          (username, email, hashed_pwd))
                conn.commit()
                
                token = generate_token(username, "verify")
                verification_link = f"{APP_BASE_URL}/?token={token}&purpose=verify"
                email_body = f"""Click this link to verify your account:
                {verification_link}
                
                Link expires in {TOKEN_EXPIRY_HOURS} hours."""
                
                if send_email(email, "Verify Your Account", email_body):
                    st.success("Account created! Check your email for verification link.")
                else:
                    st.error("Failed to send verification email")
            except sqlite3.IntegrityError as e:
                st.error("Username or email already exists")

def login_page():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Login"):
        if username and password:
            c = conn.cursor()
            c.execute("SELECT password, verified FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            
            if result and check_hashes(password, result[0]):
                if result[1]:  # If verified
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Account not verified. Check your email.")
            else:
                st.error("Invalid credentials")
        else:
            st.warning("Please enter both fields")

def forgot_password_page():
    st.subheader("Reset Password")
    email = st.text_input("Enter your registered email")
    
    if st.button("Send Reset Link"):
        if email:
            c = conn.cursor()
            c.execute("SELECT username FROM users WHERE email = ?", (email,))
            result = c.fetchone()
            
            if result:
                username = result[0]
                token = generate_token(username, "reset")
                reset_link = f"{APP_BASE_URL}/?token={token}&purpose=reset"
                email_body = f"""Click this link to reset your password:
                {reset_link}
                
                Link expires in {TOKEN_EXPIRY_HOURS} hours."""
                
                if send_email(email, "Password Reset Request", email_body):
                    st.success("Password reset link sent to your email")
                else:
                    st.error("Failed to send reset email")
            else:
                st.error("Email not found")

def verify_page(token):
    c = conn.cursor()
    c.execute("SELECT username, expires FROM tokens WHERE token = ? AND purpose = 'verify'", (token,))
    result = c.fetchone()
    
    if result and datetime.now() < datetime.strptime(result[1], '%Y-%m-%d %H:%M:%S.%f'):
        username = result[0]
        c.execute("UPDATE users SET verified = 1 WHERE username = ?", (username,))
        c.execute("DELETE FROM tokens WHERE token = ?", (token,))
        conn.commit()
        st.success("Account verified successfully! You can now login.")
    else:
        st.error("Invalid or expired token")

def reset_password_page(token):
    c = conn.cursor()
    c.execute("SELECT username, expires FROM tokens WHERE token = ? AND purpose = 'reset'", (token,))
    result = c.fetchone()
    
    if result and datetime.now() < datetime.strptime(result[1], '%Y-%m-%d %H:%M:%S.%f'):
        username = result[0]
        new_password = st.text_input("New Password", type='password')
        confirm_password = st.text_input("Confirm New Password", type='password')
        
        if st.button("Reset Password"):
            if new_password == confirm_password:
                hashed_pwd = make_hashes(new_password)
                c.execute("UPDATE users SET password = ? WHERE username = ?", (hashed_pwd, username))
                c.execute("DELETE FROM tokens WHERE token = ?", (token,))
                conn.commit()
                st.success("Password updated successfully!")
            else:
                st.error("Passwords don't match")
    else:
        st.error("Invalid or expired token")

def logout():
    if 'logged_in' in st.session_state:
        del st.session_state.logged_in
    if 'username' in st.session_state:
        del st.session_state.username

def is_logged_in():
    return st.session_state.get('logged_in', False)

def get_username():
    return st.session_state.get('username', None)