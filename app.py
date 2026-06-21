import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import json
import time
import pickle
import random
import hashlib
from datetime import datetime, date, timedelta
from pathlib import Path
from PIL import Image, ImageFilter
import warnings
warnings.filterwarnings("ignore")

try:
    from torchvision import transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Surveillance and Attendance System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Geist:wght@300;400;500;600&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Geist', ui-sans-serif, system-ui, sans-serif !important;
}

.stApp {
    background: #0e1117 !important;
    color: #c9cdd6 !important;
}
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

.main-header {
    background: linear-gradient(135deg, #13161e 0%, #161c28 100%);
    border: 1px solid #1f2d40;
    border-top: 3px solid #0ea5b0;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.4rem;
}
.main-header h1 {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e5ec;
    margin: 0;
    letter-spacing: 0.01em;
}
.main-header p {
    color: #5a6480;
    font-size: 0.8rem;
    margin: 0.25rem 0 0;
}

.metric-card {
    background: linear-gradient(135deg, #13161e 0%, #161c28 100%);
    border: 1px solid #1f2d40;
    border-radius: 10px;
    padding: 1rem 1.2rem 0.9rem;
    margin-bottom: 0.5rem;
}
.metric-card .value {
    font-family: 'Geist Mono', monospace !important;
    font-size: 1.9rem;
    font-weight: 500;
    color: #e2e5ec;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-card .label {
    color: #5a6480;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}
.metric-card .delta {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.72rem;
    color: #22c55e;
    margin-top: 0.2rem;
}
.metric-card .delta.neg { color: #ef4444; }

section[data-testid="stSidebar"] {
    background: #0a0d14 !important;
    border-right: 1px solid #1a2030 !important;
}
section[data-testid="stSidebar"] > div {
    background: #0a0d14 !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #8892a4 !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"] label {
    color: #8892a4 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 0 !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
    color: #c9cdd6 !important;
}

.sidebar-logo {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.75rem;
    color: #0ea5b0 !important;
    padding: 0.7rem 0 0.9rem;
    border-bottom: 1px solid #1a2030;
    margin-bottom: 1.2rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 600;
}

.user-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #13161e;
    border: 1px solid #1f2d40;
    border-radius: 20px;
    padding: 6px 12px;
    margin-bottom: 1rem;
    font-size: 0.78rem;
    color: #8892a4 !important;
    font-weight: 400;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #13161e 0%, #161c28 100%);
    border: 1px solid #1f2d40;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="metric-container"] label {
    color: #5a6480 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e5ec !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 1.6rem !important;
}

.stButton > button {
    background: #161c28 !important;
    color: #c9cdd6 !important;
    border: 1px solid #2a3550 !important;
    border-radius: 8px !important;
    font-size: 0.83rem !important;
    font-weight: 400 !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    border-color: #0ea5b0 !important;
    color: #e2e5ec !important;
    background: #162028 !important;
}
.stButton > button[kind="primary"] {
    background: #0a4a50 !important;
    border-color: #0ea5b0 !important;
    color: #67e8f0 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #0d5c63 !important;
    color: #a5f3fc !important;
}

.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: #13161e !important;
    border: 1px solid #1f2d40 !important;
    border-radius: 8px !important;
    color: #c9cdd6 !important;
    font-size: 0.85rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #0ea5b0 !important;
    box-shadow: 0 0 0 2px rgba(14,165,176,0.15) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label,
.stSlider label, .stFileUploader label, .stRadio label,
.stToggle label {
    color: #8892a4 !important;
    font-size: 0.82rem !important;
}

.stDataFrame, [data-testid="stDataFrame"] {
    border: 1px solid #1f2d40 !important;
    border-radius: 10px !important;
}
[data-testid="stDataFrame"] th {
    background: #13161e !important;
    color: #5a6480 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    border-bottom: 1px solid #1f2d40 !important;
}
[data-testid="stDataFrame"] td {
    color: #c9cdd6 !important;
    font-size: 0.84rem !important;
    border-bottom: 1px solid #181c26 !important;
}

div[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left-width: 3px !important;
}
.stAlert [data-baseweb="notification"] {
    background: #13161e !important;
    border-color: #1f2d40 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1f2d40 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #5a6480 !important;
    font-size: 0.82rem !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.5rem 1.1rem !important;
}
.stTabs [aria-selected="true"] {
    color: #e2e5ec !important;
    border-bottom-color: #0ea5b0 !important;
}

[data-testid="stForm"] {
    background: #13161e;
    border: 1px solid #1f2d40;
    border-radius: 12px;
    padding: 1.2rem;
}

.stProgress > div > div {
    background: #0ea5b0 !important;
}

.info-box {
    background: #13161e;
    border: 1px solid #1f2d40;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: #8892a4;
    font-size: 0.84rem;
    line-height: 1.6;
}

.status-live {
    display: inline-flex; align-items: center; gap: 7px;
    background: #0a1f14; border: 1px solid #14532d;
    border-radius: 20px; padding: 3px 12px; color: #4ade80;
    font-size: 0.75rem; font-weight: 500; letter-spacing: 0.05em;
    text-transform: uppercase;
}
.status-dot {
    width: 6px; height: 6px; background: #22c55e;
    border-radius: 50%; animation: blink 2s infinite;
}
.status-intruder {
    display: inline-flex; align-items: center; gap: 7px;
    background: #1a0a0a; border: 1px solid #7f1d1d;
    border-radius: 20px; padding: 3px 12px; color: #f87171;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase;
}
.status-intruder-dot {
    width: 6px; height: 6px; background: #ef4444;
    border-radius: 50%; animation: blink 0.7s infinite;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.15; } }

.intruder-alert {
    background: #130a0a;
    border: 1px solid #7f1d1d;
    border-left: 3px solid #ef4444;
    border-radius: 10px;
    padding: 0.9rem 1.1rem; margin: 0.5rem 0;
}
.intruder-alert-title {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.72rem;
    color: #ef4444; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 0.4rem;
}
.intruder-log-row {
    background: #0d0f14; border: 1px solid #2a1a1a;
    border-radius: 8px; padding: 0.45rem 0.75rem; margin: 0.2rem 0;
    display: flex; align-items: center; gap: 12px;
    font-size: 0.8rem; font-family: 'Geist Mono', monospace !important;
    color: #8892a4;
}

.agent-bubble-user {
    background: #111c2a;
    border: 1px solid #1e3040;
    border-radius: 12px 12px 4px 12px;
    padding: 0.8rem 1rem; margin: 0.4rem 0 0.4rem 2.5rem;
    color: #7dd3da; font-size: 0.85rem;
}
.agent-bubble-ai {
    background: #13161e;
    border: 1px solid #1f2d40;
    border-radius: 12px 12px 12px 4px;
    padding: 0.8rem 1rem; margin: 0.4rem 2.5rem 0.4rem 0;
    color: #c9cdd6; font-size: 0.85rem;
}
.agent-bubble-ai .agent-name {
    font-size: 0.68rem; color: #0ea5b0; font-weight: 600;
    letter-spacing: 0.1em; margin-bottom: 0.3rem; text-transform: uppercase;
    font-family: 'Geist Mono', monospace !important;
}
.agent-bubble-user .user-name {
    font-size: 0.68rem; color: #7dd3da; font-weight: 600;
    letter-spacing: 0.1em; margin-bottom: 0.3rem; text-transform: uppercase;
    text-align: right; font-family: 'Geist Mono', monospace !important;
}

.anomaly-card {
    background: #130a0a; border: 1px solid #3f1515;
    border-left: 3px solid #ef4444;
    border-radius: 10px; padding: 0.85rem 1rem; margin: 0.35rem 0;
}
.anomaly-card.warning {
    background: #130f08; border-color: #3f2d10; border-left-color: #f59e0b;
}
.anomaly-card.info {
    background: #090e18; border-color: #1a2540; border-left-color: #0ea5b0;
}
.anomaly-severity {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 0.2rem;
    font-family: 'Geist Mono', monospace !important;
}
.severity-high { color: #ef4444; }
.severity-med  { color: #f59e0b; }
.severity-low  { color: #22c55e; }

.report-section {
    background: #13161e; border: 1px solid #1f2d40;
    border-radius: 10px; padding: 1.1rem; margin: 0.5rem 0;
}
.report-section h4 { color: #0ea5b0; margin: 0 0 0.5rem; font-size: 0.88rem; font-weight: 500; }

.pipeline-step {
    display: flex; align-items: center; gap: 10px;
    padding: 0.5rem 0.8rem; border-radius: 8px;
    margin: 0.2rem 0; background: #0d0f14; border: 1px solid #1a2030;
    font-size: 0.8rem; color: #5a6480;
    font-family: 'Geist Mono', monospace !important;
}
.pipeline-step.done  { border-left: 2px solid #22c55e; color: #22c55e; }
.pipeline-step.active { border-left: 2px solid #0ea5b0; color: #67e8f0; }
.pipeline-step.pending { color: #2d3348; }

.login-container {
    max-width: 400px;
    margin: 70px auto 0;
    padding: 2.2rem 2.2rem 1.8rem;
    background: #13161e;
    border: 1px solid #1f2d40;
    border-top: 2px solid #0ea5b0;
    border-radius: 12px;
}
.login-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e5ec;
    margin-bottom: 0.3rem;
    letter-spacing: 0.01em;
    line-height: 1.4;
    white-space: nowrap;
}
.login-subtitle {
    font-size: 0.78rem;
    color: #5a6480;
    margin-bottom: 1.8rem;
}

[data-testid="stExpander"] {
    background: #13161e !important;
    border: 1px solid #1f2d40 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #8892a4 !important;
    font-size: 0.84rem !important;
}

[data-baseweb="select"] > div {
    background: #13161e !important;
    border-color: #1f2d40 !important;
    border-radius: 8px !important;
    color: #c9cdd6 !important;
}
[data-baseweb="menu"] {
    background: #13161e !important;
    border: 1px solid #1f2d40 !important;
    border-radius: 8px !important;
}
[data-baseweb="option"]:hover {
    background: #1a2030 !important;
}

[data-testid="stToggle"] {
    color: #8892a4 !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #0ea5b0 !important;
}

hr { border-color: #1f2d40 !important; }

h2, h3 { color: #c9cdd6 !important; font-weight: 500 !important; font-size: 0.95rem !important; }

[data-testid="collapsedControl"] { display: none !important; }
button[kind="headerNoPadding"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] button svg { display: none !important; }
section[data-testid="stSidebar"] button[aria-label*="arrow"],
section[data-testid="stSidebar"] button[aria-label*="collapse"],
section[data-testid="stSidebar"] button[aria-label*="sidebar"] {
    display: none !important;
}
[data-testid="stSidebar"] > div:first-child > div > button {
    display: none !important;
}
button[data-testid="baseButton-headerNoPadding"] { display: none !important; }

[data-baseweb="radio"] svg { display: none !important; }
[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p svg { display: none !important; }
section[data-testid="stSidebar"] [data-baseweb="radio"] div[role="radio"]::before {
    content: "·";
    color: #3a4055;
    font-size: 1.2rem;
    margin-right: 6px;
    line-height: 1;
}
section[data-testid="stSidebar"] [data-baseweb="radio"] div[aria-checked="true"]::before {
    content: "✓";
    color: #0ea5b0;
    font-size: 0.85rem;
}

/* Session config card */
.session-config-card {
    background: #0d1520;
    border: 1px solid #1a3040;
    border-left: 3px solid #0ea5b0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.session-config-card .sc-title {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.7rem;
    color: #0ea5b0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.session-config-card .sc-row {
    font-size: 0.82rem;
    color: #c9cdd6;
    margin: 0.15rem 0;
}
.session-config-card .sc-label {
    color: #5a6480;
    font-size: 0.75rem;
}

/* Duplicate face warning */
.face-dupe-warning {
    background: #130f08;
    border: 1px solid #92400e;
    border-left: 3px solid #f59e0b;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
}
.face-dupe-title {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.72rem;
    color: #f59e0b;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Paths & Constants ────────────────────────────────────────────────────────
DATA_DIR          = Path("attendance_data")
FACES_DIR         = DATA_DIR / "faces"
MODELS_DIR        = DATA_DIR / "models"
ATTENDANCE_FILE   = DATA_DIR / "attendance.csv"
USERS_FILE        = DATA_DIR / "users.json"
EMBEDDINGS_FILE   = DATA_DIR / "embeddings.pkl"
INTRUDERS_FILE    = DATA_DIR / "intruders.json"
INTRUDERS_CSV     = DATA_DIR / "intruders.csv"
CREDENTIALS_FILE  = DATA_DIR / "credentials.json"

for _d in [DATA_DIR, FACES_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

COSINE_THRESHOLD      = 0.45
FRAME_SCALE           = 0.5
FRAME_SKIP            = 6
IMG_SIZE_AUG          = 224
IMAGES_PER_PERSON     = 400
WEBCAM_FRAMES         = 100

SESSION_DURATION_MINS = 75
SESSION_DURATION_SECS = SESSION_DURATION_MINS * 60

# Duplicate face detection threshold — if a new registrant's embedding is
# closer than this to any existing person's mean embedding, warn the user.
DUPLICATE_FACE_THRESHOLD = 0.60

# ─── Groq API Configuration ───────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ─── Semester / Section / Subject Options ────────────────────────────────────
SEMESTER_OPTIONS = [
    "Semester 1", "Semester 2", "Semester 3", "Semester 4",
    "Semester 5", "Semester 6", "Semester 7", "Semester 8",
]
SECTION_OPTIONS  = ["A", "B", "C", "D", "E", "F"]
SUBJECT_OPTIONS  = [
    "Mathematics", "Physics", "Chemistry", "Computer Science",
    "Data Structures", "Algorithms", "Operating Systems",
    "Database Systems", "Networks", "Software Engineering",
    "Artificial Intelligence", "Machine Learning",
    "Digital Logic Design", "Electronics", "English",
    "Islamic Studies", "Pakistan Studies", "Other",
]

# ─── Authentication Functions ──────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def init_credentials():
    if not CREDENTIALS_FILE.exists():
        default_creds = {
            "admin": {
                "password": hash_password("admin123"),
                "role": "admin",
                "name": "System Administrator",
                "created_at": str(date.today())
            }
        }
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump(default_creds, f, indent=2)

def load_credentials() -> dict:
    init_credentials()
    with open(CREDENTIALS_FILE, "r") as f:
        return json.load(f)

def save_credentials(creds: dict):
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(creds, f, indent=2)

def authenticate_user(username: str, password: str) -> dict:
    creds = load_credentials()
    if username in creds:
        if creds[username]["password"] == hash_password(password):
            return {
                "username": username,
                "role": creds[username]["role"],
                "name": creds[username].get("name", username)
            }
    return None

def register_teacher(username: str, password: str, name: str) -> bool:
    creds = load_credentials()
    if username in creds:
        return False
    creds[username] = {
        "password": hash_password(password),
        "role": "teacher",
        "name": name,
        "created_at": str(date.today())
    }
    save_credentials(creds)
    return True

def is_admin() -> bool:
    return st.session_state.get("user_role") == "admin"

def is_teacher() -> bool:
    return st.session_state.get("user_role") == "teacher"

def is_authenticated() -> bool:
    return st.session_state.get("authenticated", False)

def logout():
    for key in ["authenticated", "username", "user_role", "user_name"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def get_sidebar_navigation():
    all_items = {
        "Dashboard": "Dashboard",
        "Live Attendance": "Live Attendance",
        "Register User": "Register User",
        "Records & Analytics": "Records & Analytics",
        "Intruder Detection Agent": "Intruder Detection Agent",
        "AI Report Assistant": "AI Report Assistant",
        "Batch Import": "Batch Import",
        "Settings": "Settings",
        "Manage Teachers": "Manage Teachers"
    }
    if is_teacher():
        restricted = ["Batch Import", "Settings", "Manage Teachers"]
        return {k: v for k, v in all_items.items() if k not in restricted}
    return all_items

# ─── Admin Functions for Teacher Management ───────────────────────────────────
def get_all_teachers() -> list:
    creds = load_credentials()
    teachers = []
    for username, info in creds.items():
        if info.get("role") == "teacher":
            teachers.append({
                "username": username,
                "name": info.get("name", username),
                "created_at": info.get("created_at", "N/A")
            })
    return teachers

def delete_teacher(username: str) -> bool:
    creds = load_credentials()
    if username in creds and creds[username].get("role") == "teacher":
        del creds[username]
        save_credentials(creds)
        return True
    return False

# ─── Augmentation Pipeline ────────────────────────────────────────────────────
def _build_augmentation_pipeline():
    if not HAS_TORCHVISION:
        return None, None
    base = T.Compose([
        T.Resize((IMG_SIZE_AUG, IMG_SIZE_AUG)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        T.RandomPerspective(distortion_scale=0.2, p=0.4),
        T.RandomGrayscale(p=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    ])
    extra = T.Compose([
        T.RandomAutocontrast(p=0.3),
        T.RandomEqualize(p=0.2),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    ])
    return base, extra

def _augment_pil(pil_img, base_aug, extra_aug):
    img = base_aug(pil_img)
    if random.random() < 0.5:
        img = extra_aug(img)
    if random.random() < 0.2:
        img = img.filter(ImageFilter.SMOOTH)
    return img

def augment_source_images(source_pil_list, target_count=IMAGES_PER_PERSON, seed=42):
    random.seed(seed)
    base_aug, extra_aug = _build_augmentation_pipeline()

    if base_aug is None:
        result = [img.resize((IMG_SIZE_AUG, IMG_SIZE_AUG)) for img in source_pil_list]
        while len(result) < target_count:
            result.append(random.choice(source_pil_list).resize((IMG_SIZE_AUG, IMG_SIZE_AUG)))
        return result[:target_count]

    result = []
    for img in source_pil_list:
        result.append(img.resize((IMG_SIZE_AUG, IMG_SIZE_AUG)))
    for img in source_pil_list:
        if len(result) >= target_count:
            break
        result.append(_augment_pil(img, base_aug, extra_aug))
    while len(result) < target_count:
        src = random.choice(source_pil_list)
        result.append(_augment_pil(src, base_aug, extra_aug))
    return result[:target_count]

def bgr_to_pil(bgr_frame):
    return Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ─── Cached Model Loader ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_insightface():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(320, 320))
        return app
    except Exception:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(320, 320))
            return app
        except Exception:
            return None

# ─── Data Helpers ─────────────────────────────────────────────────────────────
def load_users():
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def load_embeddings():
    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_embeddings(emb: dict):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(emb, f)

def load_attendance():
    if ATTENDANCE_FILE.exists():
        df = pd.read_csv(ATTENDANCE_FILE)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["time"] = pd.to_datetime(
            df["time"].replace("", pd.NaT), format="%H:%M:%S", errors="coerce"
        ).dt.time
        # Ensure new columns exist for backward compat
        for col in ["semester", "section", "subject"]:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=["id", "name", "date", "time", "status",
                                  "session_id", "semester", "section", "subject"])

def mark_attendance(uid: str, name: str, session_id: str, session_marked_ids: set,
                    semester: str = "", section: str = "", subject: str = ""):
    if uid in session_marked_ids:
        return False, "Already marked this session"

    df = load_attendance()
    local_now = datetime.now().astimezone()
    new_row = pd.DataFrame([{
        "id":         uid,
        "name":       name,
        "date":       str(date.today()),
        "time":       local_now.strftime("%H:%M:%S"),
        "status":     "Present",
        "session_id": session_id,
        "semester":   semester,
        "section":    section,
        "subject":    subject,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
    return True, f"Marked attendance for {name}"

def mark_absent_for_unseen(users: dict, session_marked_ids: set, session_id: str,
                            semester: str = "", section: str = "", subject: str = ""):
    df = load_attendance()
    new_rows = []
    for lbl, info in users.items():
        uid  = info.get("id", lbl)
        name = info.get("name", lbl)
        if uid not in session_marked_ids:
            new_rows.append({
                "id":         uid,
                "name":       name,
                "date":       str(date.today()),
                "time":       "",
                "status":     "Absent",
                "session_id": session_id,
                "semester":   semester,
                "section":    section,
                "subject":    subject,
            })
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

def load_intruder_log():
    if INTRUDERS_FILE.exists():
        with open(INTRUDERS_FILE) as f:
            return json.load(f)
    return []

def save_intruder_log(log: list):
    with open(INTRUDERS_FILE, "w") as f:
        json.dump(log, f, indent=2)

# ─── Intruder CSV Helpers ─────────────────────────────────────────────────────
_INTRUDER_CSV_COLS = [
    "intruder_id", "date", "time", "timestamp",
    "detection_score", "face_quality", "status",
]

def _ensure_intruder_csv_header(): 
    if not INTRUDERS_CSV.exists():
        pd.DataFrame(columns=_INTRUDER_CSV_COLS).to_csv(INTRUDERS_CSV, index=False)

def load_intruder_csv() -> pd.DataFrame:
    _ensure_intruder_csv_header()
    df = pd.read_csv(INTRUDERS_CSV)
    for col in _INTRUDER_CSV_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[_INTRUDER_CSV_COLS]

def append_intruder_csv(event: dict): #adding new intruder to existing csv
    _ensure_intruder_csv_header()
    row = pd.DataFrame([{
        "intruder_id":     event["id"],
        "date":            event["date"],
        "time":            event["time"],
        "timestamp":       event["timestamp"],
        "detection_score": event.get("detection_score", ""),
        "face_quality":    event.get("face_quality", ""),
        "status":          event.get("status", "UNRESOLVED"),
    }])
    row.to_csv(INTRUDERS_CSV, mode="a", header=False, index=False)

def log_intruder_event( #log that displays in screen
    confidence_score: float,
    face_quality: float = 0.0,
    detection_score: float = 0.0,
) -> dict:
    log = load_intruder_log()
    now_dt = datetime.now().astimezone()

    event = {
        "id":               f"INTR-{len(log)+1:04d}",
        "timestamp":        now_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "date":             str(date.today()),
        "time":             now_dt.strftime("%H:%M:%S"),
        "confidence":       round(confidence_score, 4),
        "detection_score":  round(detection_score or confidence_score, 4),
        "face_quality":     round(face_quality, 4),
        "status":           "UNRESOLVED",
        "ai_alert":         None,
    }

    log.append(event)
    save_intruder_log(log)
    append_intruder_csv(event)

    return event

# ══════════════════════════════════════════════════════════════════════════════
# ARCFACE RECOGNITION ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_embedding_matrix(emb_db: dict): #live attandance embedding working
    if not emb_db:
        return [], None
    labels, vecs = [], []
    for label, vectors in emb_db.items():
        arr      = np.array(vectors, dtype=np.float32)
        mean_vec = arr.mean(axis=0)
        norm     = np.linalg.norm(mean_vec)
        if norm > 1e-8:
            mean_vec /= norm
        labels.append(label)
        vecs.append(mean_vec)
    matrix = np.stack(vecs, axis=1).astype(np.float32)
    return labels, matrix

def get_best_match_fast(embedding, labels, matrix):
    emb  = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb /= norm
    scores = emb @ matrix
    idx    = int(np.argmax(scores))
    return labels[idx], float(scores[idx])

def extract_embedding(face_app, image_bgr):
    if face_app is None:
        return []
    rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb)
    return [
        (f.bbox.astype(int),
         np.asarray(f.normed_embedding, dtype=np.float32),
         float(f.det_score))
        for f in faces
    ]

def draw_face_box(frame, bbox, label, score, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    txt = f"{label}" if label else "Unknown"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, txt, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)
    return frame

# ─── Auto-label for same-name, different-face registrations ──────────────────
def make_unique_label(base_label: str, emb_db: dict) -> str:
    """
    If base_label already exists in emb_db, return 'base_label 2',
    'base_label 3', etc. until a free slot is found.
    """
    if base_label not in emb_db:
        return base_label
    counter = 2
    while f"{base_label} {counter}" in emb_db:
        counter += 1
    return f"{base_label} {counter}"

# ─── Duplicate face check ─────────────────────────────────────────────────────
def check_duplicate_face(new_embeddings: list, emb_db: dict, threshold: float = DUPLICATE_FACE_THRESHOLD):
    """
    Given a list of new embeddings for a registrant, compute the mean embedding
    and compare it against every existing person's mean embedding.
    Returns (is_duplicate: bool, matched_label: str, score: float).
    """
    if not emb_db or not new_embeddings:
        return False, None, 0.0

    new_arr = np.array(new_embeddings, dtype=np.float32)
    new_mean = new_arr.mean(axis=0)
    norm = np.linalg.norm(new_mean)
    if norm > 1e-8:
        new_mean /= norm

    best_label = None
    best_score = -1.0

    for label, vectors in emb_db.items():
        arr = np.array(vectors, dtype=np.float32)
        mean_vec = arr.mean(axis=0)
        n = np.linalg.norm(mean_vec)
        if n > 1e-8:
            mean_vec /= n
        score = float(new_mean @ mean_vec)
        if score > best_score:
            best_score = score
            best_label = label

    is_dup = best_score >= threshold
    return is_dup, best_label, best_score

# ─── Transfer Learning from CSV (training)──────────────────────────────────────────────
def finetune_from_csv(csv_file, face_app, progress_bar, status_text):
    try:
        df = pd.read_csv(csv_file)
        if "path" not in df.columns or "label" not in df.columns:
            return False, "CSV must have 'path' and 'label' columns."
        emb_db  = load_embeddings()
        users   = load_users()
        total   = len(df)
        success, failed = 0, 0
        for i, row in df.iterrows():
            img_path = str(row["path"]).strip()
            label    = str(row["label"]).strip()
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Processing {i+1}/{total}: {label}")
            if not os.path.exists(img_path):
                failed += 1
                continue
            img = cv2.imread(img_path)
            if img is None:
                failed += 1
                continue
            results = extract_embedding(face_app, img)
            if not results:
                failed += 1
                continue
            _, emb, _ = results[0]
            emb_db.setdefault(label, []).append(emb.tolist())
            person_dir = FACES_DIR / label
            person_dir.mkdir(parents=True, exist_ok=True)
            existing_count = len(list(person_dir.glob("*.jpg")))
            dst = person_dir / f"{label}_{existing_count + 1:03d}.jpg"
            cv2.imwrite(str(dst), img)
            if label not in users:
                users[label] = {
                    "id": f"USR{len(users)+1:04d}",
                    "name": label, "dept": "",
                    "registered": str(date.today())
                }
            success += 1
        save_embeddings(emb_db)
        save_users(users)
        if "emb_matrix_v2" in st.session_state:
            del st.session_state["emb_matrix_v2"]
        return True, f"Done: {success} processed, {failed} failed."
    except Exception as e:
        return False, f"Error: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# GROQ API INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def call_groq_api(messages: list, system: str = "", temperature: float = 0.7) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        groq_messages = []
        if system:
            groq_messages.append({"role": "system", "content": system})
        for msg in messages:
            groq_messages.append({"role": msg["role"], "content": msg["content"]})
        chat_completion = client.chat.completions.create(
            messages=groq_messages,
            model=GROQ_MODEL,
            temperature=temperature,
            max_tokens=2048,
            top_p=0.95,
        )
        return chat_completion.choices[0].message.content.strip()
    except ImportError:
        return "[Error: groq library not installed. Run: pip install groq]"
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg.lower() or "invalid" in error_msg.lower():
            return "[Error: Invalid or missing Groq API key. Please set a valid GROQ_API_KEY]"
        elif "rate" in error_msg.lower():
            return "[Error: Rate limit exceeded. Please try again in a moment.]"
        else:
            return f"[Groq API Error: {error_msg}]"

def test_groq_connection() -> bool:
    try:
        st.info("Testing Groq API connection with Llama 3...")
        result = call_groq_api(
            [{"role": "user", "content": "Say 'API is working!' in exactly those words."}],
            temperature=0.0
        )
        if "Error" in result and "API is working" not in result:
            st.error(f"Groq API test failed: {result}")
            return False
        else:
            st.success(f"Groq API is working. Model: {GROQ_MODEL}")
            return True
    except Exception as e:
        st.error(f"Groq API test failed: {e}")
        return False

# ─── Shared Data Context Builder ──────────────────────────────────────────────
def build_data_context(att_df: pd.DataFrame, users: dict) -> str:
    if len(att_df) == 0:
        return "No attendance data available."
    full_df   = load_attendance()
    all_dates = sorted(full_df["date"].unique()) if len(full_df) > 0 else []
    n_days    = len(all_dates)
    date_min  = att_df["date"].min() if len(att_df) > 0 else "N/A"
    date_max  = att_df["date"].max() if len(att_df) > 0 else "N/A"
    lines = [
        f"Report period: {date_min} to {date_max}",
        f"Total sessions/days recorded: {n_days} (these are the only days attendance was taken)",
        f"Total registered students: {len(users)}",
        "",
        "Per-student summary:"
    ]
    for lbl, info in users.items():
        name = info.get("name", lbl)
        udata = att_df[att_df["name"] == name] if len(att_df) > 0 else pd.DataFrame()
        present_dates = set(udata[udata["status"] == "Present"]["date"].tolist()) if len(udata) > 0 else set()
        days_present  = len(present_dates)
        days_absent   = n_days - days_present
        rate          = days_present / max(1, n_days) * 100
        streak        = 0
        for d in sorted(all_dates, reverse=True):
            if d not in present_dates:
                streak += 1
            else:
                break
        lines.append(
            f"  {name}: {days_present}/{n_days} sessions present ({rate:.0f}%), "
            f"{days_absent} absent, current absence streak: {streak} session(s)"
        )
    return "\n".join(lines)

def compute_attendance_stats(att_df: pd.DataFrame, users: dict) -> dict:
    stats = {}
    if len(att_df) == 0 or not users:
        return stats
    full_df   = load_attendance()
    all_dates = sorted(full_df["date"].unique()) if len(full_df) > 0 else sorted(att_df["date"].unique())
    n_days = len(all_dates)
    for lbl, info in users.items():
        name = info.get("name", lbl)
        udata = att_df[att_df["name"] == name]
        present_dates = set(udata[udata["status"] == "Present"]["date"].tolist()) if len(udata) > 0 else set()
        absent_dates  = set(udata[udata["status"] == "Absent"]["date"].tolist())  if len(udata) > 0 else set()
        days_present  = len(present_dates)
        days_absent   = len(absent_dates)
        rate = days_present / max(1, n_days)
        streak = 0
        max_streak = 0
        for d in sorted(all_dates, reverse=True):
            if d not in present_dates:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        stats[name] = {
            "days_present":       days_present,
            "days_absent":        days_absent,
            "total_days":         n_days,
            "rate":               rate,
            "max_absence_streak": max_streak,
            "current_streak":     streak,
        }
    return stats

# ══════════════════════════════════════════════════════════════════════════════
# INTRUDER ALERT AGENT
# ══════════════════════════════════════════════════════════════════════════════

def generate_intruder_ai_alert(event: dict, total_intruders_today: int) -> str:
    prompt = (
        f"You are a university security AI agent. An unknown person (not in the student/staff registry) "
        f"was detected by the face recognition system.\n\n"
        f"Incident ID: {event['id']}\n"
        f"Timestamp: {event['timestamp']}\n"
        f"Detection Confidence Score: {event['confidence']}\n"
        f"Total unknown detections today: {total_intruders_today}\n\n"
        f"Generate a concise security alert (3-5 sentences) that:\n"
        f"1. States the incident clearly and urgently\n"
        f"2. Gives a recommended immediate action for security staff\n"
        f"3. Rates the threat level (LOW / MEDIUM / HIGH) based on frequency and time\n"
        f"4. Suggests a follow-up protocol\n"
        f"Keep it professional, actionable, and brief."
    )
    return call_groq_api(
        [{"role": "user", "content": prompt}],
        system="You are a university campus security AI. Generate concise, professional, actionable security alerts.",
        temperature=0.5
    )

def run_intruder_analysis_chat(question: str, intruder_log: list, chat_history: list) -> str:
    log_summary = f"Total unknown face detections: {len(intruder_log)}\n"
    if intruder_log:
        today_str = str(date.today())
        today_count = sum(1 for e in intruder_log if e.get("date") == today_str)
        log_summary += f"Detections today: {today_count}\n"
        log_summary += "\nRecent incidents:\n"
        for e in intruder_log[-10:]:
            log_summary += f"  [{e['id']}] {e['timestamp']}\n"

    system_prompt = (
        "You are a professional security assistant for a university surveillance system. "
        "Your job is to summarize and explain unknown face detection events in clear, plain English. "
        "RULES: "
        "1. Never mention technical terms like 'confidence score', 'UNRESOLVED', 'embedding', or any internal system fields. "
        "2. Refer to detections simply as 'unknown individuals' or 'unrecognised faces'. "
        "3. Assess risk based on frequency and timing. "
        "4. Be concise and professional. Write in full sentences, not bullet dumps. "
        "5. Only use the data provided. Do not invent details.\n\n"
        f"DETECTION LOG:\n{log_summary}"
    )

    messages = []
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    return call_groq_api(messages, system=system_prompt, temperature=0.1)

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def show_login_page():
    st.markdown("""
    <div style="height: 80px;"></div>
    <div style="max-width: 380px; margin: 0 auto;">
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #e2e5ec; letter-spacing: -0.01em; line-height: 1.3;">
                Smart Surveillance and<br>Attendance System
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username  = st.text_input("Username",  placeholder="Enter username")
            password  = st.text_input("Password",  type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.username  = user["username"]
                        st.session_state.user_role = user["role"]
                        st.session_state.user_name = user["name"]
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        st.markdown("""
        <div style="text-align: center; margin-top: 1rem;">
            <span style="font-size: 0.75rem; color: #3a4055;">
                Contact your administrator for access
            </span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MANAGE TEACHERS PAGE (Admin Only)
# ══════════════════════════════════════════════════════════════════════════════

def show_manage_teachers():
    st.markdown("""
    <div class="main-header">
        <h1>Manage Teachers</h1>
        <p>Add, view, or remove teacher accounts</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Registered Teachers")
        teachers = get_all_teachers()
        if teachers:
            df_teachers = pd.DataFrame(teachers)
            st.dataframe(df_teachers, use_container_width=True, hide_index=True)
        else:
            st.info("No teachers registered yet.")

    with col2:
        st.subheader("Add New Teacher")
        with st.form("add_teacher_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_name     = st.text_input("Full Name")
            submitted    = st.form_submit_button("Add Teacher", use_container_width=True)

            if submitted:
                if not new_username or not new_password or not new_name:
                    st.error("Please fill all fields")
                elif len(new_password) < 4:
                    st.error("Password must be at least 4 characters")
                else:
                    if register_teacher(new_username, new_password, new_name):
                        st.success(f"Teacher {new_name} added successfully.")
                        st.rerun()
                    else:
                        st.error("Username already exists")

        st.markdown("---")
        st.subheader("Remove Teacher")
        teachers_list = get_all_teachers()
        if teachers_list:
            teacher_options = {f"{t['name']} ({t['username']})": t['username'] for t in teachers_list}
            selected = st.selectbox("Select teacher to remove", options=list(teacher_options.keys()))
            if st.button("Remove Selected Teacher", use_container_width=True, type="secondary"):
                username_to_remove = teacher_options[selected]
                if delete_teacher(username_to_remove):
                    st.success("Teacher removed successfully.")
                    st.rerun()
                else:
                    st.error("Failed to remove teacher")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP SHELL
# ══════════════════════════════════════════════════════════════════════════════

def show_main_app():
    with st.sidebar:
        role_name = "Administrator" if is_admin() else "Teacher"
        st.markdown(f"""
        <div style="text-align: left; margin-bottom: 1rem;">
            <div class="user-badge">
                {st.session_state.get('user_name', 'User')} &middot; {role_name}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-logo">Smart Surveillance and Attendance System</div>',
                    unsafe_allow_html=True)

        nav_items = get_sidebar_navigation()
        page = st.radio("Navigation", list(nav_items.keys()), label_visibility="collapsed")

        st.markdown("---")
        if st.button("Sign Out", use_container_width=True):
            logout()

    if page == "Dashboard":
        show_dashboard()
    elif page == "Live Attendance":
        show_live_attendance()
    elif page == "Register User":
        show_register_user()
    elif page == "Records & Analytics":
        show_reports_analytics()
    elif page == "Intruder Detection Agent":
        show_intruder_agent()
    elif page == "AI Report Assistant":
        show_smart_report_agent()
    elif page == "Batch Import":
        if is_admin():
            show_finetune_model()
        else:
            st.error("Access Denied: Only administrators can access Batch Import.")
    elif page == "Settings":
        if is_admin():
            show_settings()
        else:
            st.error("Access Denied: Only administrators can access Settings.")
    elif page == "Manage Teachers":
        if is_admin():
            show_manage_teachers()
        else:
            st.error("Access Denied: Only administrators can manage teachers.")

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def show_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>Dashboard</h1>
        <p>Attendance overview and system status</p>
    </div>""", unsafe_allow_html=True)

    att_df    = load_attendance()
    users     = load_users()
    today     = date.today()
    intruders = load_intruder_log()
    today_intr = sum(1 for e in intruders if e.get("date") == str(today))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="value">{len(users)}</div>'
                    f'<div class="label">Registered Users</div></div>', unsafe_allow_html=True)
    with c2:
        present_today = att_df[(att_df["date"] == today) & (att_df["status"] == "Present")]["name"].nunique() \
                        if len(att_df) > 0 else 0
        st.markdown(f'<div class="metric-card"><div class="value">{present_today}</div>'
                    f'<div class="label">Present Today</div></div>', unsafe_allow_html=True)
    with c3:
        absent = max(0, len(users) - present_today)
        st.markdown(f'<div class="metric-card"><div class="value">{absent}</div>'
                    f'<div class="label">Absent Today</div></div>', unsafe_allow_html=True)
    with c4:
        rate_val = int(present_today / max(1, len(users)) * 100)
        st.markdown(f'<div class="metric-card"><div class="value">{rate_val}%</div>'
                    f'<div class="label">Attendance Rate — Today</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Today's Attendance")
        today_att = att_df[(att_df["date"] == today) & (att_df["status"] == "Present")] \
                    if len(att_df) > 0 else pd.DataFrame()
        if len(today_att) > 0:
            show = today_att[["name", "time", "status"]].copy()
            show.columns = ["Name", "Time", "Status"]
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.info("No attendance marked today.")

    with col_r:
        st.subheader("Registered Users")
        if users:
            user_rows = []
            for lbl, info in users.items():
                user_rows.append({
                    "Name":       info.get("name", lbl),
                    "ID":         info.get("id", lbl),
                    "Registered": info.get("registered", "—"),
                })
            users_df = pd.DataFrame(user_rows)
            st.dataframe(users_df, use_container_width=True, hide_index=True)
        else:
            st.info("No users registered yet.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Daily Attendance Rate — Last 14 Days")
    if len(att_df) > 0 and len(users) > 0:
        try:
            import plotly.graph_objects as go
            n_users    = max(1, len(users))
            window     = 14
            week_start = today - timedelta(days=window - 1)
            date_range = [week_start + timedelta(days=i) for i in range(window)]
            window_df  = att_df[att_df["date"] >= week_start]
            present_by_day = (
                window_df[window_df["status"] == "Present"]
                .groupby("date")["name"].nunique()
                .reindex(date_range, fill_value=0)
            )
            rates      = (present_by_day / n_users * 100).round(1)
            x_labels   = [d.strftime("%d %b") for d in date_range]
            bar_colors = ["#3b82f6" if d == today else "#1e3a5c" for d in date_range]
            fig = go.Figure(go.Bar(
                x=x_labels, y=rates.values,
                marker_color=bar_colors,
                marker_line_width=0,
                text=[f"{v:.0f}%" if v > 0 else "" for v in rates.values],
                textposition="outside",
                textfont=dict(color="#5a6480", size=10),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#5a6480", family="Geist, sans-serif", size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                height=230,
                yaxis=dict(
                    gridcolor="#1a1e2a", ticksuffix="%",
                    range=[0, 110], showline=False, zeroline=False,
                    tickfont=dict(size=10)
                ),
                xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
                bargap=0.35,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install plotly for charts: pip install plotly")
    else:
        st.info("No attendance data recorded yet.")

# ══════════════════════════════════════════════════════════════════════════════
# LIVE ATTENDANCE  (with session config: semester / section / subject)
# ══════════════════════════════════════════════════════════════════════════════

def show_live_attendance():
    st.markdown("""
    <div class="main-header">
        <h1>Live Face Attendance</h1>
        <p>Configure your session, then start the camera. Each person is marked once per session.</p>
    </div>""", unsafe_allow_html=True)

    emb_db = load_embeddings()
    users  = load_users()

    if not emb_db:
        st.warning("No face embeddings found. Register users or run fine-tuning first.")

    if emb_db:
        cache_key  = "emb_matrix_v2"
        emb_db_sig = hash(frozenset(emb_db.keys()))
        if cache_key not in st.session_state or st.session_state.get("emb_db_sig") != emb_db_sig:
            emb_labels, emb_matrix = build_embedding_matrix(emb_db)
            st.session_state[cache_key]    = (emb_labels, emb_matrix)
            st.session_state["emb_db_sig"] = emb_db_sig
        else:
            emb_labels, emb_matrix = st.session_state[cache_key]
    else:
        emb_labels, emb_matrix = [], None

    # ── Session Configuration ──────────────────────────────────────────────
    st.markdown("#### Session Configuration")
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        sess_semester = st.selectbox("Semester", SEMESTER_OPTIONS, key="live_semester")
    with cfg2:
        sess_section  = st.selectbox("Section",  SECTION_OPTIONS,  key="live_section")
    with cfg3:
        sess_subject  = st.selectbox("Subject",  SUBJECT_OPTIONS,  key="live_subject")

    # Show active session card
    st.markdown(f"""
    <div class="session-config-card">
        <div class="sc-title">Active Session</div>
        <div class="sc-row"><span class="sc-label">Semester:</span> {sess_semester}</div>
        <div class="sc-row"><span class="sc-label">Section:</span> {sess_section}</div>
        <div class="sc-row"><span class="sc-label">Subject:</span> {sess_subject}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="status-live"><div class="status-dot"></div> Camera Feed</div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run              = st.toggle("Start Camera", value=False)
        intruder_logging = st.toggle("Enable Intruder Detection & Logging", value=True)
        frame_placeholder = st.empty()

    with col2:
        st.subheader("Session Log")
        session_log    = st.empty()
        marked_today   = st.empty()
        session_timer  = st.empty()
        intruder_panel = st.empty()

    if not run:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Camera is off. Toggle above to start.",
                    (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 1)
        frame_placeholder.image(blank, channels="BGR", width=640)

    if run:
        face_app = load_insightface()
        if face_app is None:
            st.error("InsightFace not available. Run: pip install insightface onnxruntime")
            st.stop()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam. Check camera permissions.")
            st.stop()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        session_start_ts = time.time()
        session_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_marked_ids: set = set()
        session_records   = []
        session_intruders = []
        last_marked_ts    = {}
        last_intruder_log = 0
        INTRUDER_COOLDOWN = 30
        MARK_COOLDOWN     = 5
        frame_count       = 0
        last_results      = []
        session_known_labels: set = set()
        WARMUP_SECS = 10

        try:
            while run:
                elapsed_secs   = time.time() - session_start_ts
                remaining_secs = max(0, SESSION_DURATION_SECS - elapsed_secs)
                elapsed_mins   = int(elapsed_secs // 60)
                elapsed_s_part = int(elapsed_secs % 60)
                remain_mins    = int(remaining_secs // 60)
                remain_s_part  = int(remaining_secs % 60)

                warmup_left = max(0, WARMUP_SECS - elapsed_secs)
                if warmup_left > 0:
                    session_timer.markdown(
                        f"⏳ **Warming up… {int(warmup_left)}s** — please face the camera."
                    )
                else:
                    session_timer.markdown(
                        f"**Session:** `{elapsed_mins:02d}:{elapsed_s_part:02d}` elapsed &nbsp;|&nbsp; "
                        f"`{remain_mins:02d}:{remain_s_part:02d}` remaining"
                    )

                if elapsed_secs >= SESSION_DURATION_SECS:
                    session_timer.markdown(
                        f"**Session ended** — {SESSION_DURATION_MINS}-minute limit reached."
                    )
                    break

                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera.")
                    break

                frame_count  += 1
                display_frame = frame.copy()

                if frame_count % FRAME_SKIP == 0 and emb_matrix is not None:
                    small       = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
                    raw_results = extract_embedding(face_app, small)
                    last_results = []

                    for bbox_small, emb, det_score in raw_results:
                        if det_score < 0.75:
                            continue

                        bbox = (bbox_small / FRAME_SCALE).astype(int)
                        label, score = get_best_match_fast(emb, emb_labels, emb_matrix)

                        if label and score >= COSINE_THRESHOLD:
                            user_info    = users.get(label, {})
                            uid          = user_info.get("id", label)
                            display_name = user_info.get("name", label)
                            color        = (74, 222, 128)

                            session_known_labels.add(label)
                            session_known_labels.add(uid)

                            now_ts = time.time()
                            if uid not in session_marked_ids and \
                               (uid not in last_marked_ts or (now_ts - last_marked_ts[uid]) > MARK_COOLDOWN):
                                last_marked_ts[uid] = now_ts
                                ok, msg = mark_attendance(
                                    uid, display_name, session_id, session_marked_ids,
                                    semester=sess_semester,
                                    section=sess_section,
                                    subject=sess_subject,
                                )
                                if ok:
                                    session_marked_ids.add(uid)
                                    session_records.append({
                                        "Name":   display_name,
                                        "Time":   datetime.now().astimezone().strftime("%H:%M:%S"),
                                        "Status": "Marked"
                                    })
                                else:
                                    session_records.append({
                                        "Name":   display_name,
                                        "Time":   datetime.now().astimezone().strftime("%H:%M:%S"),
                                        "Status": f"Note: {msg}"
                                    })

                        else:
                            display_name = "UNKNOWN"
                            score        = score or 0.0
                            color        = (59, 59, 220)

                            now_ts     = time.time()
                            in_warmup  = (now_ts - session_start_ts) < WARMUP_SECS
                            is_known_person = (label in session_known_labels) or (
                                label is not None and
                                users.get(label, {}).get('id') in session_known_labels
                            )

                            if (intruder_logging
                                    and not in_warmup
                                    and not is_known_person
                                    and (now_ts - last_intruder_log) > INTRUDER_COOLDOWN):
                                last_intruder_log = now_ts
                                event = log_intruder_event(
                                    confidence_score=float(score),
                                    face_quality=float(det_score),
                                    detection_score=float(score),
                                )
                                session_intruders.append({
                                    "ID":     event["id"],
                                    "Time":   event["time"],
                                    "Status": "UNKNOWN"
                                })

                        last_results.append((bbox, display_name, score, color))

                for bbox, display_name, score, color in last_results:
                    display_frame = draw_face_box(display_frame, bbox, display_name, score, color)

                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", width=640)

                if session_records:
                    session_log.dataframe(
                        pd.DataFrame(session_records[-10:]),
                        use_container_width=True, hide_index=True
                    )

                marked_today.markdown(
                    f"**Marked this session:** {len(session_marked_ids)} person(s)"
                )

                if session_intruders:
                    intr_html = '<div style="margin-top:0.8rem;">'
                    intr_html += '<div style="color:#be123c;font-weight:600;font-size:0.75rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.4rem;">UNKNOWN FACE LOG (SESSION)</div>'
                    for ev in session_intruders[-5:]:
                        intr_html += (
                            f'<div class="intruder-log-row">'
                            f'<span style="color:#be123c;font-weight:600;">{ev["ID"]}</span>'
                            f'<span style="color:#6b7280">{ev["Time"]}</span>'
                            f'</div>'
                        )
                    intr_html += "</div>"
                    intruder_panel.markdown(intr_html, unsafe_allow_html=True)

                time.sleep(0.02)

        finally:
            cap.release()
            mark_absent_for_unseen(
                users, session_marked_ids, session_id,
                semester=sess_semester, section=sess_section, subject=sess_subject,
            )
            session_timer.markdown(
                f"**Session complete** — {len(session_marked_ids)} present, "
                f"{len(users) - len(session_marked_ids)} absent recorded."
            )

# ══════════════════════════════════════════════════════════════════════════════
# REGISTER USER  (persistent state across tab changes, duplicate face check)
# ══════════════════════════════════════════════════════════════════════════════

def show_register_user():
    st.markdown("""
    <div class="main-header">
        <h1>Register New User</h1>
        <p>Register a new user and enroll their face</p>
    </div>""", unsafe_allow_html=True)

    if not HAS_TORCHVISION:
        st.warning(
            "`torchvision` not found — augmentation will be skipped (basic resize only). "
            "Install with: `pip install torchvision`"
        )

    # ── Persistent registration state so tab navigation doesn't reset form ──
    if "reg_name"   not in st.session_state: st.session_state.reg_name   = ""
    if "reg_id"     not in st.session_state: st.session_state.reg_id     = ""
    if "reg_dept"   not in st.session_state: st.session_state.reg_dept   = ""
    if "reg_method" not in st.session_state: st.session_state.reg_method = "Upload Photo(s)"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("User Details")
        reg_name = st.text_input(
            "Full Name *", placeholder="e.g. John Doe",
            value=st.session_state.reg_name, key="_reg_name_input"
        )
        reg_id = st.text_input(
            "Employee / Student ID", placeholder="Auto-generated if blank",
            value=st.session_state.reg_id, key="_reg_id_input"
        )
        reg_dept = st.text_input(
            "Department / Class", placeholder="e.g. Engineering",
            value=st.session_state.reg_dept, key="_reg_dept_input"
        )

        # Persist values back to session state on every render
        st.session_state.reg_name = reg_name
        st.session_state.reg_id   = reg_id
        st.session_state.reg_dept = reg_dept

        st.subheader("Capture Method")
        capture_method = st.radio(
            "Choose method",
            ["Upload Photo(s)", "Live Webcam Capture"],
            index=0 if st.session_state.reg_method == "Upload Photo(s)" else 1,
            key="_reg_method_radio"
        )
        st.session_state.reg_method = capture_method

    with col2:
        st.subheader("Preview / Progress")

        if capture_method == "Upload Photo(s)":
            uploaded_imgs = st.file_uploader(
                "Upload 1–20 frontal face images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )

            if uploaded_imgs:
                previews = []
                for uf in uploaded_imgs[:3]:
                    arr = np.frombuffer(uf.read(), np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        previews.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    uf.seek(0)
                if previews:
                    st.image(previews, width=180,
                             caption=[f"Photo {i+1}" for i in range(len(previews))])

            if st.button("Register User", type="primary", use_container_width=True):
                if not reg_name.strip():
                    st.error("Name is required.")
                elif not uploaded_imgs:
                    st.error("Please upload at least one image.")
                else:
                        face_app = load_insightface()
                        if face_app is None:
                            st.error("InsightFace not loaded.")
                        else:
                            label      = reg_name.strip()
                            uid        = reg_id.strip() if reg_id.strip() \
                                         else f"USR{len(load_users())+1:04d}"
                            person_dir = FACES_DIR / label
                            person_dir.mkdir(parents=True, exist_ok=True)

                            st.markdown("Loading photos…")
                            prog1 = st.progress(0)
                            source_pil = []
                            for i, uf in enumerate(uploaded_imgs):
                                uf.seek(0)
                                arr = np.frombuffer(uf.read(), np.uint8)
                                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if bgr is not None:
                                    source_pil.append(bgr_to_pil(bgr))
                                prog1.progress((i + 1) / len(uploaded_imgs))

                            if not source_pil:
                                st.error("Could not load any images.")
                            else:
                                st.markdown("Augmenting images…")
                                prog2   = st.progress(0)
                                status2 = st.empty()
                                status2.text("Running augmentation...")

                                augmented_pil = augment_source_images(source_pil, target_count=IMAGES_PER_PERSON)
                                prog2.progress(1.0)

                                existing = len(list(person_dir.glob("*.jpg")))
                                for idx, aug_img in enumerate(augmented_pil):
                                    suffix = "orig" if idx < len(source_pil) else "aug"
                                    fname  = person_dir / f"{label}_{existing + idx + 1:04d}_{suffix}.jpg"
                                    aug_img.save(str(fname), quality=95)

                                status2.text(f"Augmentation complete. {len(augmented_pil)} images saved.")

                                st.markdown("Extracting embeddings…")
                                prog3   = st.progress(0)
                                status3 = st.empty()

                                emb_db   = load_embeddings()
                                users_db = load_users()
                                new_embs = []
                                failed   = 0

                                for i, aug_img in enumerate(augmented_pil):
                                    bgr     = pil_to_bgr(aug_img)
                                    results = extract_embedding(face_app, bgr)
                                    if results:
                                        _, emb, det_score = results[0]
                                        if det_score > 0.4:
                                            new_embs.append(emb.tolist())
                                        else:
                                            failed += 1
                                    else:
                                        failed += 1

                                    if (i + 1) % 10 == 0 or (i + 1) == len(augmented_pil):
                                        prog3.progress((i + 1) / len(augmented_pil))
                                        status3.text(
                                            f"Extracted {len(new_embs)} embeddings "
                                            f"({failed} skipped low-confidence)..."
                                        )

                                if new_embs:
                                    # ── Duplicate face & name check ───────────
                                    is_dup, matched_label, dup_score = check_duplicate_face(
                                        new_embs, emb_db
                                    )
                                    name_exists = label in emb_db

                                    if is_dup and matched_label == label:
                                        # Same face, same name → person already registered
                                        existing_id = users_db.get(label, {}).get("id", "N/A")
                                        st.warning(
                                            f"**{label}** is already registered in the system "
                                            f"(ID: `{existing_id}`, similarity {dup_score:.2f}). "
                                            "No changes were made."
                                        )
                                    elif is_dup and matched_label != label:
                                        # Same face, different name → duplicate person
                                        st.markdown(f"""
                                        <div class="face-dupe-warning">
                                            <div class="face-dupe-title">⚠ Duplicate Face Detected</div>
                                            This face closely matches an already-registered person:
                                            <strong>{matched_label}</strong> (similarity {dup_score:.2f}).
                                            Registration has been blocked to prevent duplicate entries.
                                            If this is a different person, please use a clearer, more
                                            distinct photo or contact the administrator.
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif not is_dup and name_exists:
                                        # Different face, same name → auto-assign a unique label
                                        label = make_unique_label(label, emb_db)
                                        uid   = reg_id.strip() if reg_id.strip() \
                                                else f"USR{len(users_db)+1:04d}"
                                        person_dir = FACES_DIR / label
                                        person_dir.mkdir(parents=True, exist_ok=True)
                                        emb_db.setdefault(label, []).extend(new_embs)
                                        save_embeddings(emb_db)
                                        users_db[label] = {
                                            "id": uid, "name": label,
                                            "dept": reg_dept.strip(),
                                            "registered": str(date.today())
                                        }
                                        save_users(users_db)
                                        if "emb_matrix_v2" in st.session_state:
                                            del st.session_state["emb_matrix_v2"]
                                        st.session_state.reg_name = ""
                                        st.session_state.reg_id   = ""
                                        st.session_state.reg_dept = ""
                                        st.success(
                                            f"A different person with the same name already exists. "
                                            f"Registered automatically as **{label}**.\n\n"
                                            f"- {len(source_pil)} source photo(s)\n"
                                            f"- {len(augmented_pil)} augmented images saved\n"
                                            f"- **{len(new_embs)} embeddings** extracted "
                                            f"({failed} skipped)\n"
                                            f"- ID: `{uid}`"
                                        )
                                    else:
                                        # New face, new name → register normally
                                        emb_db.setdefault(label, []).extend(new_embs)
                                        save_embeddings(emb_db)
                                        users_db[label] = {
                                            "id": uid, "name": label,
                                            "dept": reg_dept.strip(),
                                            "registered": str(date.today())
                                        }
                                        save_users(users_db)
                                        if "emb_matrix_v2" in st.session_state:
                                            del st.session_state["emb_matrix_v2"]
                                        # Clear persistent form state after success
                                        st.session_state.reg_name = ""
                                        st.session_state.reg_id   = ""
                                        st.session_state.reg_dept = ""
                                        st.success(
                                            f"**{label}** registered successfully.\n\n"
                                            f"- {len(source_pil)} source photo(s)\n"
                                            f"- {len(augmented_pil)} augmented images saved\n"
                                            f"- **{len(new_embs)} embeddings** extracted "
                                            f"({failed} skipped)\n"
                                            f"- ID: `{uid}`"
                                        )
                                else:
                                    st.error(
                                        "No valid face embeddings extracted. "
                                        "Try clearer, well-lit frontal photos."
                                    )

        else:  # Live Webcam Capture
            webcam_frames_target = st.slider(
                "Frames to capture from webcam",
                min_value=20, max_value=150, value=WEBCAM_FRAMES, step=10,
                key="reg_webcam_frames"
            )

            capture_run         = st.toggle("Start Capture Session", key="reg_capture_toggle")
            capture_placeholder = st.empty()
            stage_label         = st.empty()

            if not capture_run:
                blank = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(blank, "Toggle above to start capture",
                            (55, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
                capture_placeholder.image(blank, channels="BGR", width=480)

            if capture_run:
                if not reg_name.strip():
                    st.warning("Please enter a name first, then toggle.")
                else:
                    face_app = load_insightface()
                    if face_app is None:
                        st.error("InsightFace not loaded.")
                    else:
                        label      = reg_name.strip()
                        person_dir = FACES_DIR / label
                        person_dir.mkdir(parents=True, exist_ok=True)

                        stage_label.markdown("Capturing from webcam…")
                        prog1   = st.progress(0)
                        status1 = st.empty()

                        cap = cv2.VideoCapture(0)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)

                        source_frames_pil = []
                        total_read        = 0
                        MAX_ATTEMPTS      = webcam_frames_target * 8

                        REG_WARMUP_SECS = 10
                        warmup_start    = time.time()
                        while True:
                            warmup_elapsed = time.time() - warmup_start
                            warmup_left    = REG_WARMUP_SECS - warmup_elapsed
                            if warmup_left <= 0:
                                break
                            ret_w, frame_w = cap.read()
                            if ret_w:
                                disp_w = frame_w.copy()
                            else:
                                disp_w = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(
                                disp_w,
                                f'Get ready... {int(warmup_left) + 1}s',
                                (160, 220), cv2.FONT_HERSHEY_SIMPLEX,
                                1.4, (0, 200, 255), 3
                            )
                            cv2.putText(
                                disp_w,
                                'Position your face clearly in frame',
                                (80, 270), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (200, 200, 200), 2
                            )
                            rgb_w = cv2.cvtColor(disp_w, cv2.COLOR_BGR2RGB)
                            capture_placeholder.image(rgb_w, channels='RGB', width=480)
                            status1.text(f'Starting in {int(warmup_left) + 1}s - adjust your face...')
                            time.sleep(0.05)

                        stage_label.markdown('Capturing from webcam…')

                        while len(source_frames_pil) < webcam_frames_target \
                              and total_read < MAX_ATTEMPTS:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            total_read += 1
                            display    = frame.copy()
                            results    = extract_embedding(face_app, frame)
                            face_found = False

                            for bbox_raw, emb, det_score in results:
                                if det_score > 0.6:
                                    bbox = bbox_raw.astype(int)
                                    cv2.rectangle(display,
                                                  (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                                  (74, 222, 128), 2)
                                    cv2.putText(
                                        display,
                                        f"Captured: {len(source_frames_pil)}/{webcam_frames_target}",
                                        (bbox[0], bbox[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (74, 222, 128), 1
                                    )
                                    if not face_found:
                                        source_frames_pil.append(bgr_to_pil(frame))
                                        face_found = True

                            cv2.putText(
                                display,
                                f"Good frames: {len(source_frames_pil)}/{webcam_frames_target}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (74, 222, 128) if face_found else (100, 100, 100), 2
                            )

                            rgb_disp = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                            capture_placeholder.image(rgb_disp, channels="RGB", width=480)
                            prog1.progress(min(len(source_frames_pil) / webcam_frames_target, 1.0))
                            status1.text(
                                f"Captured {len(source_frames_pil)}/{webcam_frames_target} "
                                f"good frames (scanned {total_read} total)..."
                            )
                            time.sleep(0.05)

                        cap.release()

                        blank = np.zeros((360, 480, 3), dtype=np.uint8)
                        cv2.putText(
                            blank,
                            f"Captured {len(source_frames_pil)} frames — processing...",
                            (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (74, 222, 128), 2
                        )
                        capture_placeholder.image(blank, channels="BGR", width=480)

                        if len(source_frames_pil) < 5:
                            st.error(
                                f"Only {len(source_frames_pil)} frames captured. "
                                "Ensure your face is clearly visible and well-lit."
                            )
                        else:
                            actual_captured = len(source_frames_pil)
                            status1.text(f"Captured {actual_captured} frames")

                            stage_label.markdown("Augmenting frames…")
                            prog2   = st.progress(0)
                            status2 = st.empty()
                            status2.text("Running augmentation...")
                            augmented_pil = augment_source_images(
                                source_frames_pil, target_count=IMAGES_PER_PERSON
                            )
                            prog2.progress(1.0)

                            for idx, aug_img in enumerate(augmented_pil):
                                suffix = "orig" if idx < actual_captured else "aug"
                                fname  = person_dir / f"{label}_{idx + 1:04d}_{suffix}.jpg"
                                aug_img.save(str(fname), quality=95)
                            status2.text(f"Augmentation complete. {len(augmented_pil)} images saved.")

                            stage_label.markdown("Extracting embeddings…")
                            prog3   = st.progress(0)
                            status3 = st.empty()

                            emb_db   = load_embeddings()
                            users_db = load_users()
                            new_embs = []
                            failed   = 0

                            for i, aug_img in enumerate(augmented_pil):
                                bgr     = pil_to_bgr(aug_img)
                                results = extract_embedding(face_app, bgr)
                                if results:
                                    _, emb, det_score = results[0]
                                    if det_score > 0.4:
                                        new_embs.append(emb.tolist())
                                    else:
                                        failed += 1
                                else:
                                    failed += 1

                                if (i + 1) % 10 == 0 or (i + 1) == len(augmented_pil):
                                    prog3.progress((i + 1) / len(augmented_pil))
                                    status3.text(
                                        f"Extracted {len(new_embs)} embeddings "
                                        f"({failed} skipped)..."
                                    )

                            if new_embs:
                                # ── Duplicate face & name check ───────────
                                is_dup, matched_label, dup_score = check_duplicate_face(
                                    new_embs, emb_db
                                )
                                name_exists = label in emb_db

                                if is_dup and matched_label == label:
                                    # Same face, same name → already registered
                                    existing_id = users_db.get(label, {}).get("id", "N/A")
                                    stage_label.markdown("")
                                    st.warning(
                                        f"**{label}** is already registered in the system "
                                        f"(ID: `{existing_id}`, similarity {dup_score:.2f}). "
                                        "No changes were made."
                                    )
                                elif is_dup and matched_label != label:
                                    # Same face, different name → duplicate person
                                    stage_label.markdown("")
                                    st.markdown(f"""
                                    <div class="face-dupe-warning">
                                        <div class="face-dupe-title">⚠ Duplicate Face Detected</div>
                                        This face closely matches an already-registered person:
                                        <strong>{matched_label}</strong> (similarity {dup_score:.2f}).
                                        Registration has been blocked to prevent duplicate entries.
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif not is_dup and name_exists:
                                    # Different face, same name → auto-assign a unique label
                                    label = make_unique_label(label, emb_db)
                                    uid   = reg_id.strip() if reg_id.strip() \
                                            else f"USR{len(users_db)+1:04d}"
                                    person_dir = FACES_DIR / label
                                    person_dir.mkdir(parents=True, exist_ok=True)
                                    emb_db.setdefault(label, []).extend(new_embs)
                                    save_embeddings(emb_db)
                                    users_db[label] = {
                                        "id": uid, "name": label,
                                        "dept": reg_dept.strip(),
                                        "registered": str(date.today())
                                    }
                                    save_users(users_db)
                                    if "emb_matrix_v2" in st.session_state:
                                        del st.session_state["emb_matrix_v2"]
                                    st.session_state.reg_name = ""
                                    st.session_state.reg_id   = ""
                                    st.session_state.reg_dept = ""
                                    stage_label.markdown("Registration complete.")
                                    st.success(
                                        f"A different person with the same name already exists. "
                                        f"Registered automatically as **{label}**.\n\n"
                                        f"- {actual_captured} frames captured from webcam\n"
                                        f"- {len(augmented_pil)} augmented images saved\n"
                                        f"- **{len(new_embs)} embeddings** extracted "
                                        f"({failed} skipped)\n"
                                        f"- ID: `{uid}`"
                                    )
                                else:
                                    # New face, new name → register normally
                                    uid = reg_id.strip() if reg_id.strip() \
                                          else f"USR{len(users_db)+1:04d}"
                                    emb_db.setdefault(label, []).extend(new_embs)
                                    save_embeddings(emb_db)
                                    users_db[label] = {
                                        "id": uid, "name": label,
                                        "dept": reg_dept.strip(),
                                        "registered": str(date.today())
                                    }
                                    save_users(users_db)
                                    if "emb_matrix_v2" in st.session_state:
                                        del st.session_state["emb_matrix_v2"]
                                    # Clear persistent form state
                                    st.session_state.reg_name = ""
                                    st.session_state.reg_id   = ""
                                    st.session_state.reg_dept = ""

                                    stage_label.markdown("Registration complete.")
                                    st.success(
                                        f"**{label}** registered successfully.\n\n"
                                        f"- {actual_captured} frames captured from webcam\n"
                                        f"- {len(augmented_pil)} augmented images saved\n"
                                        f"- **{len(new_embs)} embeddings** extracted "
                                        f"({failed} skipped)\n"
                                        f"- ID: `{uid}`"
                                    )
                            else:
                                st.error(
                                    "No valid embeddings extracted. "
                                    "Try better lighting and a clearer view of your face."
                                )

    st.markdown("---")
    st.subheader("Delete Student")
    st.caption("Removes the student's record, embeddings, and face photos from the system.")

    del_users = load_users()
    if not del_users:
        st.info("No registered students to delete.")
    else:
        del_options = {
            f"{info['name']} (ID: {info['id']})": key
            for key, info in del_users.items()
        }
        del_selection = st.selectbox(
            "Select student to delete",
            options=list(del_options.keys()),
            index=None,
            placeholder="Choose a student..."
        )

        if del_selection:
            del_label = del_options[del_selection]
            del_info  = del_users[del_label]
            st.warning(
                f"You are about to delete **{del_info['name']}** "
                f"(ID: `{del_info['id']}`, Registered: {del_info.get('registered', 'N/A')}). "
                "This will remove their embeddings and face photos. Attendance records will be kept."
            )
            if st.button("Confirm Delete Student", type="primary", use_container_width=True):
                import shutil
                del del_users[del_label]
                save_users(del_users)
                emb_db_del = load_embeddings()
                if del_label in emb_db_del:
                    del emb_db_del[del_label]
                    save_embeddings(emb_db_del)
                person_dir_del = FACES_DIR / del_label
                if person_dir_del.exists():
                    shutil.rmtree(str(person_dir_del))
                if "emb_matrix_v2" in st.session_state:
                    del st.session_state["emb_matrix_v2"]
                st.success(f"**{del_info['name']}** has been deleted.")
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RECORDS & ANALYTICS  (section / semester / subject filters + longer time ranges)
# ══════════════════════════════════════════════════════════════════════════════

def show_reports_analytics():
    st.markdown("""
    <div class="main-header">
        <h1>Records & Analytics</h1>
        <p>Attendance records, filters, and trends</p>
    </div>""", unsafe_allow_html=True)

    try:
        import plotly.graph_objects as go
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False
        st.warning("Install plotly for charts: pip install plotly")

    att_df = load_attendance()
    users  = load_users()

    if len(att_df) == 0:
        st.info("No attendance data yet. Start marking attendance first!")
        st.stop()

    # ── Date-range quick-select ────────────────────────────────────────────
    if "rep_start" not in st.session_state:
        st.session_state["rep_start"] = date.today() - timedelta(days=30)
    if "rep_end" not in st.session_state:
        st.session_state["rep_end"] = date.today()
    if "rep_active_btn" not in st.session_state:
        st.session_state["rep_active_btn"] = None

    active_btn = st.session_state["rep_active_btn"]

    pb1, pb2, pb3, pb4, pb5, _ = st.columns([1, 1, 1, 1, 1, 2])
    if pb1.button("Today",         key="rbtn_today", type="primary" if active_btn == "rbtn_today" else "secondary"):
        st.session_state["rep_start"]      = date.today()
        st.session_state["rep_end"]        = date.today()
        st.session_state["rep_active_btn"] = "rbtn_today"
        st.rerun()
    if pb2.button("This Week",     key="rbtn_week",  type="primary" if active_btn == "rbtn_week"  else "secondary"):
        st.session_state["rep_start"]      = date.today() - timedelta(days=date.today().weekday())
        st.session_state["rep_end"]        = date.today()
        st.session_state["rep_active_btn"] = "rbtn_week"
        st.rerun()
    if pb3.button("This Month",    key="rbtn_month", type="primary" if active_btn == "rbtn_month" else "secondary"):
        st.session_state["rep_start"]      = date.today().replace(day=1)
        st.session_state["rep_end"]        = date.today()
        st.session_state["rep_active_btn"] = "rbtn_month"
        st.rerun()
    if pb4.button("Last 3 Months", key="rbtn_3m",    type="primary" if active_btn == "rbtn_3m"    else "secondary"):
        st.session_state["rep_start"]      = date.today() - timedelta(days=90)
        st.session_state["rep_end"]        = date.today()
        st.session_state["rep_active_btn"] = "rbtn_3m"
        st.rerun()
    if pb5.button("All Time",      key="rbtn_all",   type="primary" if active_btn == "rbtn_all"   else "secondary"):
        st.session_state["rep_start"]      = att_df["date"].min() if len(att_df) > 0 else date.today()
        st.session_state["rep_end"]        = date.today()
        st.session_state["rep_active_btn"] = "rbtn_all"
        st.rerun()

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.date_input("From", key="rep_start")
    with col_f2:
        st.date_input("To", key="rep_end")

    start_date = st.session_state["rep_start"]
    end_date   = st.session_state["rep_end"]

    # ── Section / Semester / Subject filters ─────────────────────────────
    st.markdown("#### Filters")
    fc1, fc2, fc3 = st.columns(3)

    # Build dynamic options from data (+ "All" default)
    def _col_opts(col):
        if col in att_df.columns:
            vals = sorted(att_df[col].dropna().unique().tolist())
            vals = [v for v in vals if str(v).strip()]
            return ["All"] + vals
        return ["All"]

    sem_opts  = _col_opts("semester")
    sec_opts  = _col_opts("section")
    subj_opts = _col_opts("subject")

    with fc1:
        filt_semester = st.selectbox("Semester", sem_opts,  key="rep_filt_semester")
    with fc2:
        filt_section  = st.selectbox("Section",  sec_opts,  key="rep_filt_section")
    with fc3:
        filt_subject  = st.selectbox("Subject",  subj_opts, key="rep_filt_subject")

    # ── Name and Time filters ─────────────────────────────────────────────
    fc4, fc5 = st.columns(2)

    # Name filter: dropdown of known names from data
    def _name_opts():
        if "name" in att_df.columns:
            vals = sorted(att_df["name"].dropna().unique().tolist())
            vals = [v for v in vals if str(v).strip()]
            return ["All"] + vals
        return ["All"]

    name_opts = _name_opts()
    with fc4:
        filt_name = st.selectbox("Student Name", name_opts, key="rep_filt_name")
    with fc5:
        filt_time_range = st.selectbox(
            "Time of Day",
            ["All", "Morning (00:00–11:59)", "Afternoon (12:00–17:59)", "Evening (18:00–23:59)"],
            key="rep_filt_time"
        )

    # ── Apply filters ─────────────────────────────────────────────────────
    att_df["date_str"] = att_df["date"].apply(str)
    start_str, end_str = str(start_date), str(end_date)
    filtered = att_df[(att_df["date_str"] >= start_str) & (att_df["date_str"] <= end_str)]

    if filt_semester != "All" and "semester" in filtered.columns:
        filtered = filtered[filtered["semester"] == filt_semester]
    if filt_section  != "All" and "section"  in filtered.columns:
        filtered = filtered[filtered["section"]  == filt_section]
    if filt_subject  != "All" and "subject"  in filtered.columns:
        filtered = filtered[filtered["subject"]  == filt_subject]
    if filt_name != "All" and "name" in filtered.columns:
        filtered = filtered[filtered["name"] == filt_name]
    if filt_time_range != "All" and "time" in filtered.columns:
        def _in_time_range(t):
            if pd.isna(t) or t == "" or t is None:
                return False
            try:
                if hasattr(t, "hour"):
                    h = t.hour
                else:
                    h = int(str(t).split(":")[0])
                if filt_time_range.startswith("Morning"):
                    return 0 <= h <= 11
                elif filt_time_range.startswith("Afternoon"):
                    return 12 <= h <= 17
                elif filt_time_range.startswith("Evening"):
                    return 18 <= h <= 23
            except Exception:
                return False
            return True
        filtered = filtered[filtered["time"].apply(_in_time_range)]

    present_filtered = filtered[filtered["status"] == "Present"]
    total_records    = len(present_filtered)
    unique_days      = present_filtered["date"].nunique() if len(present_filtered) > 0 else 0
    unique_people    = present_filtered["name"].nunique() if len(present_filtered) > 0 else 0
    avg_daily        = total_records / max(1, unique_days)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Present Records", total_records)
    m2.metric("Days Covered",    unique_days)
    m3.metric("Unique People",   unique_people)
    m4.metric("Avg Daily",       f"{avg_daily:.1f}")

    if HAS_PLOTLY and len(present_filtered) > 0:
        st.markdown("---")
        r1c1, r1c2 = st.columns(2)

        # ── Choose time granularity based on date range ──────────────────
        date_span = (end_date - start_date).days

        with r1c1:
            if date_span <= 31:
                st.subheader("Daily Attendance Count")
                grp = present_filtered.groupby("date").size().reset_index(name="count")
                x_vals = [str(d) for d in grp["date"]]
                mode   = "lines+markers"
                x_title = "Date"
            elif date_span <= 120:
                st.subheader("Weekly Attendance Count")
                tmp = present_filtered.copy()
                tmp["week"] = pd.to_datetime(tmp["date"].apply(str)).dt.to_period("W").apply(lambda r: str(r.start_time.date()))
                grp = tmp.groupby("week").size().reset_index(name="count")
                x_vals = grp["week"].tolist()
                mode   = "lines+markers"
                x_title = "Week"
            else:
                st.subheader("Monthly Attendance Count")
                tmp = present_filtered.copy()
                tmp["month"] = pd.to_datetime(tmp["date"].apply(str)).dt.to_period("M").apply(str)
                grp = tmp.groupby("month").size().reset_index(name="count")
                x_vals = grp["month"].tolist()
                mode   = "lines+markers"
                x_title = "Month"

            fig1 = go.Figure(go.Scatter(
                x=x_vals, y=grp["count"],
                mode=mode,
                line=dict(color="#4fc3f7", width=2),
                marker=dict(color="#4fc3f7", size=7),
                fill="tozeroy", fillcolor="rgba(79,195,247,0.1)"
            ))
            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8892a4", height=300,
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis=dict(gridcolor="#1a1e2a", title=x_title),
                yaxis=dict(gridcolor="#1a1e2a"),
            )
            st.plotly_chart(fig1, use_container_width=True)

        with r1c2:
            if unique_days > 0:
                st.subheader("Attendance Rate by Person")
                # Use only names that appear in the filtered dataset
                filtered_names = sorted(present_filtered["name"].dropna().unique().tolist())
                rates = []
                for name in filtered_names:
                    pdays = present_filtered[present_filtered["name"] == name]["date"].nunique()
                    rates.append({
                        "Name":     name,
                        "Rate (%)": round(pdays / unique_days * 100, 1)
                    })
                if rates:
                    rd         = pd.DataFrame(rates).sort_values("Rate (%)", ascending=False)
                    bar_colors = [
                        "#4ade80" if r >= 75 else "#fbbf24" if r >= 50 else "#f87171"
                        for r in rd["Rate (%)"]
                    ]
                    fig4 = go.Figure(go.Bar(
                        x=rd["Name"], y=rd["Rate (%)"],
                        marker_color=bar_colors
                    ))
                    fig4.add_hline(y=75, line_dash="dash", line_color="#4ade80",
                                   annotation_text="75% threshold")
                    fig4.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#8892a4", height=300,
                        margin=dict(l=10, r=10, t=10, b=30),
                        yaxis=dict(range=[0, 100], gridcolor="#1e2640"),
                        xaxis=dict(gridcolor="#1a1e2a"),
                    )
                    st.plotly_chart(fig4, use_container_width=True)

        # ── Heatmap: student × date ───────────────────────────────────────
        if len(present_filtered) > 0 and users:
            st.markdown("---")
            st.subheader("Attendance Heatmap (Student × Date)")

            pivot = (
                filtered
                .assign(present=lambda d: (d["status"] == "Present").astype(int))
                .groupby(["name", "date"])["present"].max()
                .unstack(fill_value=0)
            )
            if not pivot.empty:
                # Cap columns to last 30 dates for readability
                cols_to_show = sorted(pivot.columns)[-30:]
                pivot        = pivot[cols_to_show]
                fig_heat = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=[str(c) for c in pivot.columns],
                    y=pivot.index.tolist(),
                    colorscale=[[0, "#1a1e2a"], [1, "#0ea5b0"]],
                    showscale=False,
                    xgap=2, ygap=2,
                ))
                fig_heat.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8892a4",
                    height=max(250, len(pivot) * 28),
                    margin=dict(l=10, r=10, t=10, b=30),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                    yaxis=dict(tickfont=dict(size=10)),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.subheader("Attendance Records")
    st.dataframe(
        filtered.sort_values(["date", "time"], ascending=[False, False]),
        use_container_width=True, hide_index=True
    )

    st.subheader("Download Report")

    if users and len(filtered) > 0:
        all_dates_range = pd.date_range(start=start_date, end=end_date).date
        report_rows = []

        # Determine which student names are relevant given the active filters.
        # If a name filter is set, only that student is included.
        # Otherwise, include only students who appear at least once in the
        # filtered dataset (i.e. they belong to the selected semester/section/subject).
        if filt_name != "All":
            relevant_names = {filt_name}
        else:
            relevant_names = set(filtered["name"].dropna().unique())

        # Build a lookup from name → user info for the relevant students only
        relevant_users = {
            lbl: info for lbl, info in users.items()
            if info.get("name", lbl) in relevant_names
        }

        for d in all_dates_range:
            day_df = filtered[filtered["date"] == d]
            present_map = {
                row["name"]: row
                for _, row in day_df[day_df["status"] == "Present"].iterrows()
            }
            for lbl, info in relevant_users.items():
                name = info.get("name", lbl)
                uid  = info.get("id", lbl)
                if name in present_map:
                    row = present_map[name]
                    report_rows.append({
                        "id":       uid,
                        "name":     name,
                        "date":     str(d),
                        "time":     str(row["time"]) if pd.notna(row["time"]) else "",
                        "status":   "Present",
                        "semester": row.get("semester", ""),
                        "section":  row.get("section", ""),
                        "subject":  row.get("subject", ""),
                    })
                else:
                    report_rows.append({
                        "id":       uid,
                        "name":     name,
                        "date":     str(d),
                        "time":     "",
                        "status":   "Absent",
                        "semester": filt_semester if filt_semester != "All" else "",
                        "section":  filt_section  if filt_section  != "All" else "",
                        "subject":  filt_subject  if filt_subject  != "All" else "",
                    })

        combined_df = pd.DataFrame(
            report_rows,
            columns=["id", "name", "date", "time", "status", "semester", "section", "subject"]
        )
        combined_df = combined_df.sort_values(["date", "name"]).reset_index(drop=True)

        st.download_button(
            label="Download Full Report (CSV)",
            data=combined_df.to_csv(index=False).encode("utf-8"),
            file_name=f"attendance_full_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            f"Combined report: {len(combined_df)} rows · "
            f"{len(combined_df[combined_df['status']=='Present'])} present, "
            f"{len(combined_df[combined_df['status']=='Absent'])} absent · "
            f"Period: {start_date} → {end_date}"
        )
    else:
        st.download_button(
            label="Download CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"attendance_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# INTRUDER DETECTION AGENT
# ══════════════════════════════════════════════════════════════════════════════

def show_intruder_agent():
    st.markdown("""
    <div class="main-header">
        <h1>Intruder Detection</h1>
        <p>Unknown face log and security chat</p>
    </div>""", unsafe_allow_html=True)

    intruder_log = load_intruder_log()
    today_str    = str(date.today())
    today_events = [e for e in intruder_log if e.get("date") == today_str]

    if today_events:
        st.markdown(f"""
        <div class="intruder-alert">
            <div class="intruder-alert-title">{len(today_events)} UNKNOWN INDIVIDUAL(S) DETECTED TODAY</div>
        </div>""", unsafe_allow_html=True)

    tab_log, tab_chat = st.tabs(["Incident Log", "Security Chat"])

    with tab_log:
        intr_df = load_intruder_csv()

        if len(intr_df) == 0:
            st.info("No intruder records yet. Run Live Attendance to detect intruders.")
        else:
            today_count = len(intr_df[intr_df["date"] == today_str]) if "date" in intr_df.columns else 0
            total_count = len(intr_df)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="value">{today_count}</div><div class="label">Detections Today</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="value">{total_count}</div><div class="label">All-Time Total</div></div>', unsafe_allow_html=True)

            date_filter = st.selectbox("Filter by Date", ["All time", "Today", "Last 7 days"])
            if date_filter == "Today":
                filtered_df = intr_df[intr_df["date"] == today_str]
            elif date_filter == "Last 7 days":
                cutoff = str(date.today() - timedelta(days=7))
                filtered_df = intr_df[intr_df["date"] >= cutoff]
            else:
                filtered_df = intr_df

            display_cols = [c for c in filtered_df.columns
                            if c not in ("detection_score", "face_quality", "status", "location")]
            st.dataframe(
                filtered_df[display_cols].sort_values("timestamp", ascending=False).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")
            st.download_button(
                "Download as CSV",
                data=intr_df.to_csv(index=False).encode("utf-8"),
                file_name=f"intruders_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_chat:
        if "intruder_chat" not in st.session_state:
            st.session_state.intruder_chat = [
                {
                    "role": "assistant",
                    "content": (
                        "Ask me about the intruder log. For example:<br>"
                        "- <i>How many unknown detections this week?</i><br>"
                        "- <i>What's the overall risk level?</i>"
                    )
                }
            ]

        for msg in st.session_state.intruder_chat:
            if msg["role"] == "assistant":
                st.markdown(f"""
                <div class="agent-bubble-ai">
                    <div class="agent-name">Security Agent</div>
                    {msg['content']}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="agent-bubble-user">
                    <div class="user-name">You</div>
                    {msg['content']}
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        quick_sec = None
        if st.button("Weekly summary", use_container_width=True):
            quick_sec = "Summarize all intruder incidents from the last 7 days and assess the risk level based only on the log data."

        sec_input = st.text_input("Ask the security agent...", key="intruder_agent_input",
                                  placeholder="e.g. What time of day do most intruders appear?")
        final_sec = quick_sec or (sec_input.strip() if sec_input.strip() else None)

        last_sec_sent = st.session_state.get("intruder_last_sent", None)

        if final_sec and final_sec != last_sec_sent:
            st.session_state.intruder_last_sent = final_sec
            st.session_state.intruder_chat.append({"role": "user", "content": final_sec})

            with st.spinner("Processing..."):
                response = run_intruder_analysis_chat(
                    final_sec, intruder_log, st.session_state.intruder_chat
                )

            st.session_state.intruder_chat.append({"role": "assistant", "content": response})
            st.rerun()

        if st.button("Clear Chat"):
            st.session_state.intruder_chat = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# AI REPORT ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

def show_smart_report_agent():
    st.markdown("""
    <div class="main-header">
        <h1>AI Report Assistant</h1>
        <p>Generate reports and query attendance data</p>
    </div>""", unsafe_allow_html=True)

    att_df = load_attendance()
    users  = load_users()

    col_cfg, col_chat = st.columns([1, 1])

    with col_cfg:
        st.subheader("Generate Report")

        report_type   = st.selectbox("Report Type", ["Summary", "Individual Student Report"])
        report_period = st.selectbox("Time Period", ["Last 7 days", "Last 30 days", "All time"])
        focus_person  = st.text_input("Focus on specific student (optional)")
        custom_notes  = st.text_area("Additional context",
                                     placeholder="e.g. class cancelled Monday, ignore that gap",
                                     height=70)
        generate_btn  = st.button("Generate Report", type="primary", use_container_width=True)

        if generate_btn:
            if len(att_df) == 0:
                st.warning("No attendance data to report on.")
            else:
                days_map  = {"Last 7 days": 7, "Last 30 days": 30, "All time": 9999}
                window    = days_map.get(report_period, 30)
                cutoff_dt = date.today() - timedelta(days=window)
                rep_df    = att_df[att_df["date"] >= cutoff_dt] if window < 9999 else att_df

                if focus_person.strip():
                    rep_df = rep_df[rep_df["name"].str.contains(focus_person.strip(), case=False, na=False)]

                pipeline_ph = st.empty()
                pipeline_steps = [
                    ("1.", "Loading data"),
                    ("2.", "Computing statistics"),
                    ("3.", "Generating report"),
                    ("4.", "Done"),
                ]
                ph_html = '<div style="margin:0.5rem 0;">'
                for icon, label in pipeline_steps:
                    ph_html += f'<div class="pipeline-step active">{icon} {label}...</div>'
                    pipeline_ph.markdown(ph_html + "</div>", unsafe_allow_html=True)
                    time.sleep(0.3)

                data_ctx = build_data_context(rep_df, users)

                include_flags = [
                    "actionable recommendations",
                    "risk flags for students below 75% attendance",
                    "statistical summary"
                ]
                include_str  = ', '.join(include_flags)
                focus_str    = f'FOCUS ONLY ON: {focus_person}' if focus_person.strip() else ''
                notes_str    = f'EXTRA CONTEXT: {custom_notes}' if custom_notes.strip() else ''
                data_block   = (
                    f"ATTENDANCE DATA (source of truth — use only these numbers):\n{data_ctx}\n\n"
                    "IMPORTANT:\n"
                    "- 'sessions recorded' = the only days attendance was taken.\n"
                    "- Attendance % = sessions present / total sessions recorded. Max is 100%.\n"
                    "- Only use the numbers above. Do not invent, estimate, or speculate.\n"
                    "- Use clear ## section headers."
                )

                if "Summary" in report_type:
                    prompt = (
                        f"You are writing an Attendance Summary for university leadership.\n\n"
                        f"INCLUDE: {include_str}\n{focus_str}\n{notes_str}\n\n{data_block}\n\n"
                        "Structure the report as:\n"
                        "## Overview\n## Top Performers\n## At-Risk Students\n"
                        "## Class-Wide Statistics\n## Recommendations\n## Risk Flags\n\n"
                        "Be concise, factual, and written for a dean or department head."
                    )
                else:
                    student_name = focus_person.strip() if focus_person.strip() else "each student individually"
                    prompt = (
                        f"You are writing an Individual Student Attendance Report.\n\n"
                        f"INCLUDE: {include_str}\nSTUDENT FOCUS: {student_name}\n{notes_str}\n\n{data_block}\n\n"
                        "For each student:\n"
                        "## Student Profile\n## Attendance Breakdown\n## Status\n"
                        "## Recommendations\n## Risk Flag (only if below 75%)\n\n"
                        "Be precise with every number."
                    )

                with st.spinner("Generating..."):
                    report_text = call_groq_api(
                        [{"role": "user", "content": prompt}],
                        system=(
                            "You are an expert university attendance analytics agent. "
                            "STRICT RULES: "
                            "1. Only use numbers and facts from the ATTENDANCE DATA provided. "
                            "2. Attendance rate = (sessions present) / (total sessions recorded) * 100. Never exceed 100%. "
                            "3. Structure with clear ## headers. Be concise and factual. "
                            "4. Never mention lateness or tardiness — attendance is binary: Present or Absent."
                        ),
                        temperature=0.3
                    )

                pipeline_ph.empty()

                st.markdown("---")
                st.markdown(
                    f"**{report_type}** · *{report_period}* · "
                    f"Generated {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}"
                )
                for line in report_text.split("\n"):
                    if line.startswith("## "):
                        st.subheader(line[3:])
                    elif line.startswith("### "):
                        st.markdown(f"**{line[4:]}**")
                    else:
                        st.markdown(line)

                if "report_history" not in st.session_state:
                    st.session_state.report_history = []
                st.session_state.report_history.append({
                    "type":      report_type,
                    "period":    report_period,
                    "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "content":   report_text,
                })

                st.markdown("---")
                st.download_button(
                    "Download Report as TXT",
                    data=report_text.encode("utf-8"),
                    file_name=f"report_{report_type.replace(' ', '_')}_{date.today()}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        history = st.session_state.get("report_history", [])
        if history:
            st.markdown("---")
            st.subheader("Previous Reports")
            for i, rep in enumerate(reversed(history)):
                idx = len(history) - 1 - i
                st.markdown(f"""
                <div class="report-section">
                    <div style="font-size:0.78rem;color:#5a6480;margin-bottom:0.5rem;">{rep['type']} — {rep['generated']}</div>
                    <div style="font-size:0.84rem;color:#c9cdd6;">{rep['content'].replace(chr(10), '<br>')}</div>
                </div>""", unsafe_allow_html=True)
                st.download_button(
                    "Download",
                    data=rep["content"].encode("utf-8"),
                    file_name=f"report_{rep['type'].replace(' ','_')}_{rep['generated'][:10]}.txt",
                    mime="text/plain",
                    key=f"dl_hist_{idx}"
                )
            if st.button("Clear Report History"):
                st.session_state.report_history = []
                st.rerun()

    with col_chat:
        st.subheader("Chat")

        if "report_chat" not in st.session_state:
            st.session_state.report_chat = [
                {
                    "role": "assistant",
                    "content": (
                        "Ask me anything about attendance. For example:<br>"
                        "- <i>Who should I be concerned about?</i><br>"
                        "- <i>Which students have perfect attendance?</i>"
                    )
                }
            ]

        for msg in st.session_state.report_chat:
            if msg["role"] == "assistant":
                st.markdown(f"""
                <div class="agent-bubble-ai">
                    <div class="agent-name">Report Agent</div>
                    {msg['content']}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="agent-bubble-user">
                    <div class="user-name">You</div>
                    {msg['content']}
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        rquick_q = None
        if st.button("Weekly attendance summary", use_container_width=True):
            rquick_q = "Write a concise weekly attendance summary for a team standup."
            st.session_state.report_last_sent = None

        report_user_input = st.text_input("Ask anything about attendance...", key="report_agent_input")
        final_r_input     = rquick_q or (report_user_input.strip() if report_user_input.strip() else None)

        att_mtime    = str(ATTENDANCE_FILE.stat().st_mtime) if ATTENDANCE_FILE.exists() else "0"
        cache_key    = f"{final_r_input}||{att_mtime}" if final_r_input else None
        report_last_sent = st.session_state.get("report_last_sent", None)

        if final_r_input and cache_key != report_last_sent:
            st.session_state.report_last_sent = cache_key
            st.session_state.report_chat.append({"role": "user", "content": final_r_input})

            fresh_att_df = load_attendance()
            fresh_users  = load_users()
            data_ctx     = build_data_context(fresh_att_df, fresh_users)

            recent_history = st.session_state.report_chat[-8:]
            messages = []
            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append({"role": "user",      "content": msg["content"]})
                else:
                    messages.append({"role": "assistant", "content": msg["content"]})

            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i]["content"] = (
                        f"[CURRENT ATTENDANCE DATA — use only these numbers]\n{data_ctx}\n\n"
                        f"USER REQUEST: {messages[i]['content']}"
                    )
                    break

            with st.spinner("Processing..."):
                response = call_groq_api(messages, system=(
                    "You are an AI Report Assistant for university attendance analytics. "
                    "STRICT RULES: "
                    "1. Only state facts from the ATTENDANCE DATA provided. Never invent numbers. "
                    "2. Attendance = unique days present / total class days. Never report more than 100%. "
                    "3. Attendance is binary: PRESENT or ABSENT per day. "
                    "4. If asked about one student, only discuss that student. "
                    "5. Be concise, factual, and professional."
                ), temperature=0.3)

            st.session_state.report_chat.append({"role": "assistant", "content": response})
            st.rerun()

        if st.button("Clear Chat"):
            st.session_state.report_chat = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# BATCH IMPORT
# ══════════════════════════════════════════════════════════════════════════════

def show_finetune_model():
    st.markdown("""
    <div class="main-header">
        <h1>Batch Import</h1>
        <p>Add new faces to the recognition model from a dataset</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Upload Dataset CSV")
        csv_file = st.file_uploader("Upload CSV file", type=["csv"])

        if csv_file:
            preview_df = pd.read_csv(csv_file)
            st.dataframe(preview_df.head(5), use_container_width=True)
            n_labels = preview_df["label"].nunique() if "label" in preview_df.columns else "N/A"
            st.caption(f"Total rows: {len(preview_df)} | Unique labels: {n_labels}")
            csv_file.seek(0)

        if st.button("Start Fine-Tuning", type="primary",
                     use_container_width=True, disabled=(csv_file is None)):
            face_app = load_insightface()
            if face_app is None:
                st.error("InsightFace not loaded. Run: pip install insightface onnxruntime")
            else:
                prog = st.progress(0)
                stat = st.empty()
                ok, msg = finetune_from_csv(csv_file, face_app, prog, stat)
                st.success(msg) if ok else st.error(msg)

    with col2:
        st.subheader("Current Model Status")
        emb_db = load_embeddings()
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{len(emb_db)}</div>
            <div class="label">Known Identities</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def show_settings():
    st.markdown("""
    <div class="main-header">
        <h1>Settings</h1>
        <p>System configuration and data management</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("API Connection")
        if st.button("Test Groq API Connection", use_container_width=True):
            test_groq_connection()

    with col2:
        st.subheader("Data Management")

        st.markdown("**Clear Attendance Records**")
        st.caption("Deletes all attendance logs. Users and face data are kept.")
        if st.button("Clear Attendance Records", use_container_width=True, key="del_att"):
            if ATTENDANCE_FILE.exists():
                ATTENDANCE_FILE.unlink()
            st.success("Attendance records cleared.")

        st.markdown("---")
        st.markdown("**Clear Intruder Logs**")
        st.caption("Deletes all intruder detection logs and CSV exports.")
        if st.button("Clear Intruder Logs", use_container_width=True, key="del_intr"):
            for fp in [INTRUDERS_FILE, INTRUDERS_CSV]:
                if fp.exists():
                    fp.unlink()
            st.success("Intruder logs cleared.")

        st.markdown("---")
        st.markdown("**Clear All Users & Embeddings**")
        st.caption("Removes all registered users, embeddings, and saved photos.")
        if st.button("Clear Users & Embeddings", use_container_width=True, key="del_users"):
            for fp in [USERS_FILE, EMBEDDINGS_FILE]:
                if fp.exists():
                    fp.unlink()
            if FACES_DIR.exists():
                import shutil
                shutil.rmtree(str(FACES_DIR))
                FACES_DIR.mkdir(parents=True, exist_ok=True)
            if "emb_matrix_v2" in st.session_state:
                del st.session_state["emb_matrix_v2"]
            st.success("Users and embeddings cleared.")

        st.markdown("---")
        st.markdown("**Reset Everything**")
        st.caption("Deletes all of the above. Login credentials are NOT affected.")
        confirm_all = st.checkbox("I understand this cannot be undone", key="confirm_reset_all")
        if st.button("Reset ALL Data", type="secondary", use_container_width=True,
                     key="del_all", disabled=not confirm_all):
            for fp in [ATTENDANCE_FILE, USERS_FILE, EMBEDDINGS_FILE, INTRUDERS_FILE, INTRUDERS_CSV]:
                if fp.exists():
                    fp.unlink()
            if FACES_DIR.exists():
                import shutil
                shutil.rmtree(str(FACES_DIR))
                FACES_DIR.mkdir(parents=True, exist_ok=True)
            if "emb_matrix_v2" in st.session_state:
                del st.session_state["emb_matrix_v2"]
            st.success("All data cleared.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    init_credentials()

    if not is_authenticated():
        show_login_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()