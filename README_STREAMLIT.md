# Vortex Finance - AI Anomaly Detector

Intelligent anomaly detection for personal finance transactions powered by machine learning.

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vortex-finance-anomaly-detector.streamlit.app)

## ✨ Features

- **📊 Transaction Analysis**: Upload bank statements (CSV/PDF) for comprehensive analysis
- **🤖 AI-Powered Detection**: Machine learning models identify spending anomalies
- **📈 Visual Insights**: Interactive charts and risk scoring
- **👥 User Management**: Simple login system with user tracking
- **🔒 Secure**: Built with FastAPI backend and Streamlit frontend

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI + SQLAlchemy
- **ML**: Scikit-learn
- **Database**: SQLite (production-ready for PostgreSQL)
- **Deployment**: Streamlit Cloud

## 🚀 Streamlit Cloud Deployment

### Step 1: Connect GitHub to Streamlit Cloud

1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Click **"New app"**
3. Connect your **GitHub account**
4. Select repository: **`Imposon/q1`**
5. Select branch: **`master`**
6. Main file path: **`streamlit_app_integrated.py`**

### Step 2: Configure Environment Variables

In Streamlit Cloud dashboard, add these secrets:

| Variable | Value | Required |
|----------|--------|----------|
| `GROQ_API_KEY` | Your Groq API key | ✅ |

### Step 3: Deploy

Click **"Deploy!"** and wait for the build to complete.

## 📋 Local Development

### Prerequisites
- Python 3.11+
- Groq API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Imposon/q1.git
   cd q1
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the integrated app**
   ```bash
   streamlit run streamlit_app_integrated.py --server.port 8502
   ```

5. **Open** http://localhost:8502

## 📊 Usage

1. **Sign up/Login** with your name and email
2. **Upload** your bank statement (CSV or PDF format)
3. **Run Analysis** to detect anomalies
4. **View Results** with detailed insights and visualizations
5. **Get AI Insights** for personalized recommendations

## 🔧 Environment Variables

```bash
GROQ_API_KEY=your_groq_api_key_here
```

## 📁 Project Structure

```
q1/
├── streamlit_app_integrated.py   # Integrated Streamlit app (no backend needed)
├── streamlit_app.py           # Original version with separate backend
├── requirements.txt             # Python dependencies
├── packages.txt              # System dependencies
└── .streamlit/
    └── config.toml             # Streamlit configuration
```

## 🌐 Deployment URLs

- **GitHub**: https://github.com/Imposon/q1
- **Streamlit Cloud**: https://vortex-finance-anomaly-detector.streamlit.app

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting guide
