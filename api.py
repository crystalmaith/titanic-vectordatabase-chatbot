"""
FastAPI wrapper for Titanic Explorer to enable Vercel deployment
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import subprocess
import threading
import time
import requests

app = FastAPI()

@app.get("/")
async def root():
    """Serve the Streamlit app through an iframe"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Titanic Explorer - Coast & Calm</title>
        <style>
            body { margin: 0; padding: 0; }
            iframe { 
                width: 100vw; 
                height: 100vh; 
                border: none; 
                position: absolute;
                top: 0;
                left: 0;
            }
        </style>
    </head>
    <body>
        <iframe src="https://share.streamlit.io/crystalmaith/titanic-vectordatabase-chatbot/main/app_deploy.py"></iframe>
    </body>
    </html>
    """)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "app": "titanic-explorer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
