import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import io
import base64
import time
from auth import show_auth_pages, is_logged_in, get_username, logout

# --- Page Config ---
st.set_page_config(
    page_title="XVision | Advanced Medical Imaging",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# --- Theme Toggle ---
def set_theme():
    st.session_state.theme = not st.session_state.get('theme', False)
    
if 'theme' not in st.session_state:
    st.session_state.theme = False

# --- Animations CSS ---
st.markdown(f"""
<style>
:root {{
    --primary: {'#4b6cb7' if not st.session_state.theme else '#3da56a'};
    --secondary: {'#182848' if not st.session_state.theme else '#5a96c9'};
    --accent: {'#d32f2f' if not st.session_state.theme else '#e53935'};
    --bg: {'#f8f9fa' if not st.session_state.theme else '#121212'};
    --card-bg: {'#ffffff' if not st.session_state.theme else '#1e1e1e'};
    --text: {'#2c3e50' if not st.session_state.theme else '#f0f0f0'};
    --sidebar-bg: {'linear-gradient(180deg, var(--primary) 0%, #1e6b45 100%)' 
                  if not st.session_state.theme else 
                  'linear-gradient(180deg, #121212 0%, #1e1e1e 100%)'};
}}

/* Animations */
@keyframes float {{
    0% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-8px); }}
    100% {{ transform: translateY(0px); }}
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes pulse {{
    0% {{ transform: scale(1); }}
    50% {{ transform: scale(1.03); }}
    100% {{ transform: scale(1); }}
}}

.floating {{
    animation: float 4s ease-in-out infinite;
}}

.fade-in {{
    animation: fadeIn 1s ease-out forwards;
}}

.pulse {{
    animation: pulse 2s ease-in-out infinite;
}}

/* Main App Styling */
.stApp {{
    background: linear-gradient(135deg, var(--bg) 0%, {'#e4e8eb' if not st.session_state.theme else '#252525'} 100%);
    font-family: 'Inter', sans-serif;
    color: var(--text);
    transition: all 0.5s ease;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    color: {'white' if not st.session_state.theme else '#f0f0f0'} !important;
}}

/* Cards */
.card {{
    background: var(--card-bg);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
    border: none;
    transition: all 0.3s ease;
}}

.card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.15);
}}

/* Buttons */
.stButton>button {{
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s ease !important;
    font-weight: 600 !important;
}}

.stButton>button:hover {{
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 12px rgba(0,0,0,0.2) !important;
}}

/* Images */
.stImage>img {{
    border-radius: 16px;
    box-shadow: 0 12px 24px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}}

.stImage>img:hover {{
    transform: scale(1.03);
}}

/* Titles */
h1, h2, h3 {{
    color: var(--text) !important;
    font-weight: 700 !important;
}}

/* Custom Classes */
.hero {{
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    padding: 4rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
}}

.alert {{
    padding: 1.5rem;
    border-radius: 12px;
    font-weight: 500;
    margin: 1rem 0;
}}

.alert-warning {{
    background-color: {'#fff3cd' if not st.session_state.theme else '#5c4b00'};
    color: {'#856404' if not st.session_state.theme else '#ffd95c'};
}}

.alert-success {{
    background-color: {'#d4edda' if not st.session_state.theme else '#1a3a23'};
    color: {'#155724' if not st.session_state.theme else '#5ae37d'};
}}

.team-member {{
    display: flex;
    justify-content: space-between;
    padding: 1rem 0;
    border-bottom: 1px solid {'rgba(0,0,0,0.1)' if not st.session_state.theme else 'rgba(255,255,255,0.1)'};
    align-items: center;
}}

.team-member:hover {{
    transform: translateX(10px);
}}
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_cancer_model():
    try:
        model = tf.keras.models.load_model("model_xception (2).h5")
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.stop()

# [Keep all your existing image processing functions here]

def show_auth_pages():
    """Enhanced authentication pages"""
    query_params = st.query_params
    
    if 'token' in query_params and 'purpose' in query_params:
        token = query_params['token'][0]
        purpose = query_params['purpose'][0]
        
        if purpose == "verify":
            verify_page(token)
        elif purpose == "reset":
            reset_password_page(token)
        return
    
    st.markdown("""
    <div style="max-width: 500px; margin: 0 auto;">
        <h1 class="login-title floating" style="text-align: center; margin-bottom: 2rem;">🔬 XVision</h1>
    """, unsafe_allow_html=True)
    
    menu = st.selectbox("Menu", ["Login", "SignUp", "Forgot Password"], label_visibility="collapsed")
    
    if menu == "Login":
        with st.form("login_form"):
            st.markdown("<h3 style='text-align: center;'>Welcome Back</h3>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            if st.form_submit_button("Login", type="primary"):
                if username and password:
                    c = conn.cursor()
                    c.execute("SELECT password, verified FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    
                    if result and check_hashes(password, result[0]):
                        if result[1]:
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
    
    elif menu == "SignUp":
        with st.form("signup_form"):
            st.markdown("<h3 style='text-align: center;'>Create Account</h3>", unsafe_allow_html=True)
            email = st.text_input("Email", placeholder="Your email address")
            username = st.text_input("Username", placeholder="Choose a username")
            password = st.text_input("Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            if st.form_submit_button("Sign Up", type="primary"):
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
    
    else:  # Forgot Password
        with st.form("forgot_form"):
            st.markdown("<h3 style='text-align: center;'>Reset Password</h3>", unsafe_allow_html=True)
            email = st.text_input("Email", placeholder="Your registered email")
            
            if st.form_submit_button("Send Reset Link", type="primary"):
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
    
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    # --- Theme Toggle Button ---
    col1, col2 = st.columns([6,1])
    with col2:
        st.button("🌓", on_click=set_theme, help="Toggle dark/light mode")

    # --- Authentication Check ---
    if not is_logged_in():
        show_auth_pages()
        return

    # --- Animated Sidebar ---
    with st.sidebar:
        st.markdown(f"""
        <div class="fade-in">
            <h3 style='color:{"white" if not st.session_state.theme else "#f0f0f0"};'>
                Welcome back, {get_username()}!
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", type="primary"):
            logout()
            st.rerun()
        
        st.markdown("---")
        st.markdown(f"""
        <div class="fade-in" style="color:{"white" if not st.session_state.theme else "#f0f0f0"}">
            <h4>Quick Guide</h4>
            <div>1. Upload a medical scan</div>
            <div>2. Get AI analysis</div>
            <div>3. Review results</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Main Content ---
    menu_selection = st.sidebar.radio("Navigation", ["Dashboard", "Scan Analysis", "About"])

    if menu_selection == "Dashboard":
        st.markdown("""
        <div class="hero floating">
            <h1>XVision</h1>
            <h3>Advanced Medical Imaging Platform</h3>
            <div>Next-generation diagnostic tools powered by AI</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="card fade-in">
                <h3>🔄 24/7 Analysis</h3>
                <div>Instant results anytime</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card fade-in" style="animation-delay: 0.2s">
                <h3>🎯 High Accuracy</h3>
                <div>Clinically validated</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card fade-in" style="animation-delay: 0.4s">
                <h3>🔍 Detailed Visualization</h3>
                <div>Precise heatmaps</div>
            </div>
            """, unsafe_allow_html=True)

    elif menu_selection == "Scan Analysis":
        st.markdown("""
        <div class="card pulse">
            <h2>📤 Upload Medical Scan</h2>
            <div>CT, MRI, or X-Ray images</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "dcm"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            with st.spinner("🔍 Analyzing scan..."):
                image = Image.open(uploaded_file)
                img_array = preprocess_image(image)
                
                if img_array is not None:
                    prediction = predict_label(img_array)
                    
                    if prediction is not None:
                        label = "Normal" if prediction[0][0] >= 0.5 else "Abnormal"
                        confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]
                        
                        st.markdown(f"""
                        <div class="card fade-in">
                            <h2>📋 Analysis Results</h2>
                            <div style="display: flex; align-items: center; gap: 1rem; margin: 1.5rem 0;">
                                <div style="font-size: 2rem; background: {'#d32f2f20' if label == 'Abnormal' else '#2e8b5720'}; 
                                    color: {'var(--accent)' if label == 'Abnormal' else 'var(--primary)'}; 
                                    padding: 1rem; border-radius: 50%;">
                                    {'⚠️' if label == 'Abnormal' else '✅'}
                                </div>
                                <div>
                                    <h3 style="margin: 0; color: {'var(--accent)' if label == 'Abnormal' else 'var(--primary)'};">
                                        {label} Findings
                                    </h3>
                                    <div>Confidence: {confidence:.1%}</div>
                                </div>
                            </div>
                            {"<div class='alert alert-warning'>Recommendation: Consult a specialist</div>" 
                            if label == "Abnormal" else 
                            "<div class='alert alert-success'>No concerning patterns detected</div>"}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("""
                            <div class="card fade-in">
                                <h3>Original Scan</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(image, use_column_width=True)
                        
                        if label == "Abnormal":
                            with col2:
                                heatmap = make_gradcam_heatmap(img_array)
                                if heatmap is not None:
                                    heatmap_image = overlay_heatmap(image, heatmap)
                                    if heatmap_image is not None:
                                        st.markdown("""
                                        <div class="card fade-in">
                                            <h3>AI Heatmap</h3>
                                            <div>Areas of interest highlighted</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        st.image(heatmap_image, use_column_width=True)
                                        
                                        buffered = io.BytesIO()
                                        heatmap_image.save(buffered, format="PNG")
                                        st.download_button(
                                            label="📥 Download Analysis",
                                            data=buffered.getvalue(),
                                            file_name="xvision_analysis.png",
                                            mime="image/png",
                                            use_container_width=True
                                        )

    elif menu_selection == "About":
        st.markdown("""
        <div class="card floating">
            <h1>About XVision</h1>
            <div>Advanced medical imaging platform</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("""
            <div class="fade-in" style="text-align: center;">
                <div style="font-size: 6rem;">👨‍⚕️</div>
                <h3>Our Team</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card fade-in" style="animation-delay: 0.2s">
                <h3>Core Development</h3>
                <div class="team-member">
                    <span>Jami Jaswanth</span>
                    <span>AI/ML Engineer</span>
                </div>
                <div class="team-member">
                    <span>Palameti Reddy Lakshmi Manoj</span>
                    <span>Data Scientist</span>
                </div>
                <div class="team-member">
                    <span>Munagala Chandra Vamsi Reddy</span>
                    <span>Backend Developer</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card fade-in" style="animation-delay: 0.4s">
            <h3>Our Technology</h3>
            <div>XVision uses state-of-the-art deep learning models trained on thousands of annotated medical scans to assist healthcare professionals in early detection of abnormalities.</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Footer ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text); padding: 1rem;">
        <div>XVision Medical Imaging Platform</div>
        <div><small>For professional use only. Not for diagnostic use.</small></div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()