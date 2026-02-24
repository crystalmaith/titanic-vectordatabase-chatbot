"""
FastAPI wrapper for Titanic Explorer to enable Vercel deployment
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import subprocess
import threading
import time
import requests

app = FastAPI(title="Titanic Explorer API")

@app.get("/")
async def root():
    """Redirect to Streamlit app on Streamlit Cloud"""
    return RedirectResponse(
        url="https://share.streamlit.io/crystalmaith/titanic-vectordatabase-chatbot/main/app_deploy.py"
    )

@app.get("/app")
async def streamlit_app():
    """Serve the Streamlit app through an iframe"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Titanic Explorer - Coast & Calm</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                margin: 0; 
                padding: 0; 
                font-family: 'Jost', sans-serif;
                background: #FFFFFF;
            }
            .header {
                background: #5D768B;
                color: #FFFFFF;
                padding: 20px;
                text-align: center;
                font-family: 'Cormorant Garamond', serif;
                font-size: 2rem;
                font-weight: 600;
            }
            .container {
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }
            .info {
                background: #EEE4DA;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                border-left: 3px solid #5D768B;
            }
            iframe { 
                width: 100%; 
                height: 800px; 
                border: none; 
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .loading {
                text-align: center;
                padding: 40px;
                color: #5D768B;
                font-size: 1.2rem;
            }
        </style>
    </head>
    <body>
        <div class="header">
            ⚓ Titanic Explorer - Coast & Calm
        </div>
        <div class="container">
            <div class="info">
                <strong>🌊 Welcome to Titanic Explorer!</strong><br>
                This interactive data visualization app lets you explore the Titanic dataset with beautiful charts and intelligent Q&A.
            </div>
            <div class="loading">
                Loading Titanic Explorer...
            </div>
            <iframe 
                src="https://share.streamlit.io/crystalmaith/titanic-vectordatabase-chatbot/main/app_deploy.py"
                onload="document.querySelector('.loading').style.display='none'">
            </iframe>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "app": "titanic-explorer", "type": "FastAPI wrapper for Streamlit"}

@app.get("/api/info")
async def info():
    """API information"""
    return {
        "name": "Titanic Explorer",
        "description": "Interactive Titanic dataset explorer with Coast & Calm theme",
        "features": [
            "Smart name search",
            "Dataset statistics", 
            "Beautiful visualizations",
            "Movie character handling",
            "Chat interface"
        ],
        "streamlit_url": "https://share.streamlit.io/crystalmaith/titanic-vectordatabase-chatbot/main/app_deploy.py"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
