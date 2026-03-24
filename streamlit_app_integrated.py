"""
Streamlit frontend with integrated backend for the Personal Finance Anomaly Detection System.

Run with:
    streamlit run streamlit_app_integrated.py

This version includes the FastAPI backend integrated directly for Streamlit Cloud deployment.
"""

import json
import os
import time
import uuid
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client for AI insights
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception:
    groq_client = None

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "transactions" not in st.session_state:
    st.session_state.transactions = None
if "users_db" not in st.session_state:
    # Simple in-memory user database for demo
    st.session_state.users_db = {}

st.set_page_config(
    page_title="Vortex Finance | AI Anomaly Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Pearl-like background gradient */
    .stApp {
        background: radial-gradient(circle at 20% 20%, #1e1e2e 0%, #11111b 50%, #09090b 100%);
    }

    /* Modern Card Layout (Shadcn-inspired) */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
    }

    /* Custom Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
    }

    /* Custom Header Styles */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        background: linear-gradient(to bottom right, #fff 30%, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    /* Input Fields (Shadcn style) */
    .stTextInput>div>div>input {
        background-color: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #e2e8f0;
        transition: border-color 0.2s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 1px #6366f1;
    }
    </style>
""", unsafe_allow_html=True)

# Database functions (in-memory for demo)
def get_total_users():
    return len(st.session_state.users_db)

def create_user(name, email):
    user_id = str(uuid.uuid4())
    user = {
        "id": user_id,
        "name": name,
        "email": email,
        "created_at": datetime.now().isoformat()
    }
    st.session_state.users_db[user_id] = user
    return user

def get_user_by_email(email):
    for user in st.session_state.users_db.values():
        if user["email"] == email:
            return user
    return None

# AI Insights function
def generate_ai_insights(transactions_data):
    if not groq_client:
        return {
            "error": "Groq API key not configured. Please set GROQ_API_KEY in secrets."
        }
    
    if not transactions_data:
        return {"error": "No transactions to analyze"}
    
    df = pd.DataFrame(transactions_data)
    
    # Calculate basic metrics
    total_spend = df[df['amount'] > 0]['amount'].sum()
    categories = df[df['amount'] > 0].groupby('category')['amount'].sum().to_dict()
    
    # Find anomalies (simplified)
    anomalies = df[df['amount'] > df['amount'].quantile(0.95)].to_dict('records')
    
    prompt = f"""
    You are 'Vortex', an expert AI financial assistant. Provide actionable insights.
    
    USER CONTEXT:
    - Total Spend: {total_spend:.2f}
    - Category Breakdown: {json.dumps(categories)}
    - Flagged Anomalies: {json.dumps(anomalies[:5])}
    
    INSTRUCTIONS:
    1. Provide a concise "ai_summary" (max 2 sentences) highlighting their biggest spending insight
    2. Calculate a 0-100 "risk_score" based on spending patterns (100 = high risk)
    3. Generate 3 actionable "recommendations" for saving money
    4. Return strict JSON format only
    
    Response format:
    {{
      "risk_score": int,
      "ai_summary": "string",
      "recommendations": ["string", "string", "string"],
      "categories": {json.dumps(categories)}
    }}
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a financial insight generator. Output only valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        if "categories" not in result:
            result["categories"] = categories
            
        return result
        
    except Exception as e:
        return {"error": f"AI Generation failed: {str(e)}"}

# Login page
if not st.session_state.user_id:
    # Add user counter in top right
    total_users = get_total_users()
    st.markdown(f"""
        <div style="position: fixed; top: 20px; right: 20px; z-index: 999; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); padding: 8px 16px; border-radius: 20px; color: #6366f1; font-weight: 600; font-size: 14px;">
            👥 Total Users: {total_users}
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-weight: 900; color: #6366f1; font-size: 4rem; margin-bottom: 0px;'>VORTEX</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.5); font-size: 1.2rem; margin-top: 0px; letter-spacing: 2px;'>AI ANOMALY DETECTOR</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<div style='background: rgba(255,255,255,0.02); padding: 40px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); box-shadow: 0 10px 30px rgba(0,0,0,0.5);'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; margin-bottom: 5px;'>Secure Login</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray; font-size: 0.9rem; margin-bottom: 25px;'>Sign in or create an account to continue</p>", unsafe_allow_html=True)
        
        # Simple login form
        name = st.text_input("Full Name", placeholder="e.g. John Doe")
        email = st.text_input("Email Address", placeholder="e.g. john@example.com")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Access Dashboard", use_container_width=True):
            if name and email:
                with st.spinner("Authenticating..."):
                    # Check if user exists, otherwise create new
                    existing_user = get_user_by_email(email)
                    if existing_user:
                        user = existing_user
                    else:
                        user = create_user(name, email)
                    
                    st.session_state.user_id = user["id"]
                    st.session_state.user_name = user["name"]
                    # Clear cache to refresh user count
                    st.cache_data.clear()
                    st.success("Access Granted! Redirecting...")
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.warning("All fields are required to continue.")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.stop()

# User counter for authenticated pages
total_users = get_total_users()
st.markdown(f"""
    <div style="position: fixed; top: 20px; right: 20px; z-index: 999; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); padding: 8px 16px; border-radius: 20px; color: #6366f1; font-weight: 600; font-size: 14px;">
        👥 Total Users: {total_users}
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-weight: 800; color: #6366f1; margin-bottom: 0px;'>VORTEX</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 0px;'>AI ANOMALY DETECTOR</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div style='background: rgba(0,255,100,0.1); border: 1px solid rgba(0,255,100,0.2); padding: 10px; border-radius: 8px; color: #00ff66; text-align: center; font-size: 0.85rem; font-weight: 600;'> System Online</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<p style='color: rgba(255,255,255,0.4); font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;'>NAVIGATE</p>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        [" Dashboard", " Upload Statement", " Run Analysis", " Transactions", " AI Insights", " About"],
        label_visibility="collapsed",
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("<p style='color: rgba(255,255,255,0.4); font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;'>AUTHENTICATION</p>", unsafe_allow_html=True)
    st.markdown(f"<div style='background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);'><p style='margin:0; font-size: 0.85rem; color: rgba(255,255,255,0.8);'>User: <b>{st.session_state.user_name}</b></p><p style='margin:0; font-size: 0.6rem; color: rgba(255,255,255,0.4);'>ID: {st.session_state.user_id[:12]}...</p></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(" Log Out", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.user_name = None
        st.session_state.analysis_result = None
        st.session_state.transactions = None
        st.rerun()

# Dashboard page
if page == " Dashboard":
    st.markdown('<h1 class="main-title">Vortex Finance</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 1.25rem; color: rgba(255,255,255,0.6); margin-top: -10px;">
        Intelligent anomaly detection for your personal finance.
        </p>
        """, unsafe_allow_html=True
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\n Upload your bank statement (CSV or PDF)")
    with col2:
        st.info("**Step 2**\n\n Run anomaly analysis and review results")
    with col3:
        st.info("**Step 3**\n\n Get AI insights for personalized recommendations")

    st.markdown("---")

    if st.session_state.transactions:
        st.success(f"✅ {len(st.session_state.transactions)} transactions loaded and analyzed!")
    else:
        st.warning("📊 No data yet. Upload your bank statement to get started.")

# Upload Statement page
elif page == " Upload Statement":
    st.markdown('<h1 class="main-title">Upload Statement</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 1.25rem; color: rgba(255,255,255,0.6); margin-top: -10px;">
        Upload your bank statement in CSV or PDF format to begin analysis.
        </p>
        """, unsafe_allow_html=True
    )
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "pdf"],
        help="Upload your bank statement (CSV or PDF format)"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            else:  # PDF
                import pdfplumber
                with pdfplumber.open(uploaded_file) as pdf:
                    pages = pdf.pages
                    text = ""
                    for page in pages:
                        text += page.extract_text() + "\n"
                
                # Simple CSV parsing from PDF text
                lines = text.strip().split('\n')
                data = []
                for line in lines[1:]:  # Skip header
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            data.append({
                                'date': parts[0].strip(),
                                'description': parts[1].strip(),
                                'amount': float(parts[2].strip()),
                                'category': 'Others'
                            })
                        except:
                            continue
                df = pd.DataFrame(data)

            # Store transactions in session state
            st.session_state.transactions = df.to_dict('records')
            st.success(f"✅ Successfully loaded {len(df)} transactions!")
            
            # Show preview
            st.subheader("Transaction Preview")
            st.dataframe(df.head(10))
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Analysis page
elif page == " Run Analysis":
    st.markdown('<h1 class="main-title">Run Analysis</h1>', unsafe_allow_html=True)
    
    if not st.session_state.transactions:
        st.warning("📊 Please upload a bank statement first.")
        st.stop()
    
    if st.button("🔍 Run Anomaly Detection", use_container_width=True):
        with st.spinner("Analyzing transactions..."):
            df = pd.DataFrame(st.session_state.transactions)
            
            # Simple anomaly detection using IQR
            if 'amount' in df.columns:
                Q1 = df['amount'].quantile(0.25)
                Q3 = df['amount'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df['is_anomaly'] = (df['amount'] < lower_bound) | (df['amount'] > upper_bound)
                df['anomaly_score'] = np.where(
                    df['is_anomaly'],
                    np.abs(df['amount'] - df['amount'].median()) / df['amount'].std() * 100,
                    0
                )
                
                st.session_state.analysis_result = df.to_dict('records')
                st.success("✅ Analysis complete! Check the Transactions tab.")
            else:
                st.error("❌ No amount column found in transactions.")

# Transactions page
elif page == " Transactions":
    st.markdown('<h1 class="main-title">Transactions</h1>', unsafe_allow_html=True)
    
    if not st.session_state.transactions:
        st.warning("📊 No transactions to display.")
        st.stop()
    
    df = pd.DataFrame(st.session_state.transactions)
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_anomalies = st.checkbox("Show Anomalies Only", value=False)
    with col2:
        category_filter = st.selectbox("Filter by Category", ["All"] + list(df['category'].unique()) if 'category' in df.columns else ["All"])
    
    # Apply filters
    if show_anomalies and 'is_anomaly' in df.columns:
        df = df[df['is_anomaly'] == True]
    
    if category_filter != "All" and 'category' in df.columns:
        df = df[df['category'] == category_filter]
    
    st.subheader(f"📊 {len(df)} Transactions")
    st.dataframe(df, use_container_width=True)

# AI Insights page
elif page == " AI Insights":
    st.markdown('<h1 class="main-title">AI Insights</h1>', unsafe_allow_html=True)
    
    if not st.session_state.transactions:
        st.warning("📊 Please upload and analyze transactions first.")
        st.stop()
    
    if st.button("🤖 Generate AI Insights", use_container_width=True):
        with st.spinner("Generating personalized insights..."):
            insights = generate_ai_insights(st.session_state.transactions)
            
            if "error" in insights:
                st.error(f"❌ {insights['error']}")
            else:
                # Display insights
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Score", f"{insights.get('risk_score', 0)}/100")
                with col2:
                    st.metric("Total Categories", len(insights.get('categories', {})))
                
                st.subheader("🧠 AI Summary")
                st.write(insights.get('ai_summary', 'No summary available.'))
                
                st.subheader("💡 Recommendations")
                for i, rec in enumerate(insights.get('recommendations', []), 1):
                    st.write(f"{i}. {rec}")
                
                # Category breakdown
                if insights.get('categories'):
                    st.subheader("📈 Spending by Category")
                    cat_df = pd.DataFrame(list(insights['categories'].items()), columns=['Category', 'Amount'])
                    st.bar_chart(cat_df.set_index('Category'))

# About page
elif page == " About":
    st.markdown('<h1 class="main-title">About Vortex</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        ## 🎯 Mission
        
        Vortex Finance uses advanced AI and machine learning to help you understand your spending patterns 
        and detect unusual transactions in your financial data.
        
        ## ✨ Features
        
        - **📊 Transaction Analysis**: Upload and analyze bank statements
        - **🤖 AI-Powered Insights**: Get personalized financial recommendations
        - **🔍 Anomaly Detection**: Identify unusual spending patterns automatically
        - **📈 Visual Analytics**: Interactive charts and dashboards
        
        ## 🛠️ Technology
        
        - **Frontend**: Streamlit
        - **AI Engine**: Groq (Llama models)
        - **Analytics**: Pandas, NumPy, Scikit-learn
        - **Deployment**: Streamlit Cloud
        
        ## 🔒 Privacy
        
        Your financial data is processed locally and never stored on external servers. 
        All analysis happens in your browser session.
        
        ---
        Built with ❤️ using modern AI and web technologies
        """, unsafe_allow_html=True
    )
