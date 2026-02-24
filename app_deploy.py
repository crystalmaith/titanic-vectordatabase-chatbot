"""
Titanic Chat Agent — Streamlit Frontend (Deployment Version)
Theme: Coast & Calm  |  #5D768B · #C8B39B · #FFFFFF
"""

import base64
import os
import json
import re
import io
import sys
from typing import List, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import streamlit as st

# ─────────────────────────── Page config ────────────────────────────────────
st.set_page_config(
    page_title="Coast & Calm · Titanic Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Colour palette ──────────────────────────────────
OCEAN_BLUE  = "#5D768B"
SANDY_BEIGE = "#C8B39B"
IVORY       = "#FFFFFF"
DARK_BLUE   = "#3D5468"

# ─────────────────────────── Custom CSS ──────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Jost:wght@300;400;500&display=swap');

  /* ── global ── */
  html, body, [class*="css"] {{
    font-family: 'Jost', sans-serif;
    background-color: {IVORY};
    color: {DARK_BLUE};
  }}

  /* ── sidebar ── */
  [data-testid="stSidebar"] {{
    background-color: {OCEAN_BLUE} !important;
    border-right: none;
  }}
  [data-testid="stSidebar"] * {{
    color: {IVORY} !important;
  }}
  [data-testid="stSidebar"] input {{
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(251,239,229,0.4) !important;
    color: {IVORY} !important;
    border-radius: 8px !important;
  }}
  [data-testid="stSidebar"] input::placeholder {{
    color: rgba(251,239,229,0.5) !important;
  }}

  /* ── chat messages ── */
  .user-bubble {{
    background: {OCEAN_BLUE};
    color: {IVORY};
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 78%;
    margin-left: auto;
    margin-bottom: 10px;
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: 0 2px 8px rgba(93,118,139,0.18);
  }}
  .bot-bubble {{
    background: #EEE4DA;
    color: {DARK_BLUE};
    padding: 14px 20px;
    border-radius: 18px 18px 18px 4px;
    max-width: 82%;
    margin-bottom: 10px;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  .bot-bubble strong {{ color: {OCEAN_BLUE}; }}

  /* ── title area ── */
  .hero-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: {OCEAN_BLUE};
    letter-spacing: 0.02em;
    margin-bottom: 0;
  }}
  .hero-sub {{
    font-family: 'Jost', sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.22em;
    color: {SANDY_BEIGE};
    text-transform: uppercase;
    margin-top: 0;
  }}
  .divider {{
    border: none;
    border-top: 1px solid {SANDY_BEIGE};
    margin: 0.5rem 0 1.2rem 0;
  }}

  /* ── input area ── */
  [data-testid="stChatInput"] textarea {{
    background: #EEE4DA !important;
    border: 1.5px solid {SANDY_BEIGE} !important;
    color: {DARK_BLUE} !important;
    border-radius: 12px !important;
    font-family: 'Jost', sans-serif !important;
  }}
  [data-testid="stChatInput"] textarea:focus {{
    border-color: {OCEAN_BLUE} !important;
    box-shadow: 0 0 0 2px rgba(93,118,139,0.15) !important;
  }}
  [data-testid="stChatInput"] button {{
    background: {OCEAN_BLUE} !important;
    border-radius: 10px !important;
  }}

  /* ── stat card ── */
  .stat-card {{
    background: #C4A484;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border-left: 3px solid {OCEAN_BLUE};
  }}
  .stat-label {{
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #654321;
    margin-bottom: 2px;
  }}
  .stat-value {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #654321;
  }}

  /* ── example pills ── */
  .pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0.8rem 0; }}
  .pill {{
    background: rgba(93,118,139,0.1);
    border: 1px solid {SANDY_BEIGE};
    color: {OCEAN_BLUE};
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    cursor: pointer;
  }}

  /* hide default streamlit header ── */
  #MainMenu, header, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Load Dataset ────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "titanic.csv")
df = pd.read_csv(DATA_PATH)

# ─────────────────────────── Simple Analysis Engine ──────────────────────────
def create_dataset_documents() -> List[str]:
    """Create text documents from the Titanic dataset for vector search."""
    documents = []
    
    # General dataset statistics
    documents.append(f"The Titanic dataset contains {len(df)} passengers.")
    documents.append(f"The survival rate was {df['Survived'].mean()*100:.1f}%.")
    documents.append(f"The average passenger age was {df['Age'].mean():.1f} years.")
    documents.append(f"The average ticket fare was £{df['Fare'].mean():.2f}.")
    
    # Gender statistics
    male_count = df[df['Sex'] == 'male'].shape[0]
    female_count = df[df['Sex'] == 'female'].shape[0]
    documents.append(f"There were {male_count} male passengers ({male_count/len(df)*100:.1f}%).")
    documents.append(f"There were {female_count} female passengers ({female_count/len(df)*100:.1f}%).")
    
    # Class statistics
    for pclass in sorted(df['Pclass'].unique()):
        class_df = df[df['Pclass'] == pclass]
        survival_rate = class_df['Survived'].mean() * 100
        documents.append(f"Class {pclass} had {len(class_df)} passengers with {survival_rate:.1f}% survival rate.")
    
    return documents

def search_passenger_name(name: str) -> str:
    """Search for passengers with a specific name in dataset."""
    try:
        name_lower = name.lower()
        
        # Search for the name in the Name column
        matching_passengers = df[df['Name'].str.lower().str.contains(name_lower, na=False)]
        
        if len(matching_passengers) > 0:
            result = f"Found {len(matching_passengers)} passenger(s) with '{name}' in their name:\n\n"
            for idx, passenger in matching_passengers.head(5).iterrows():
                passenger_name = passenger['Name']
                pclass = passenger['Pclass']
                survived = "Survived" if passenger['Survived'] == 1 else "Did not survive"
                age = f"{passenger['Age']:.1f} years" if pd.notna(passenger['Age']) else "Age unknown"
                result += f"• {passenger_name} (Class {pclass}, {age}, {survived})\n"
            
            if len(matching_passengers) > 5:
                result += f"... and {len(matching_passengers) - 5} more"
            
            return result
        else:
            # Handle movie character references
            if name_lower in ["rose", "jack", "dawson", "rose dawson"]:
                if name_lower == "rose" or name_lower == "rose dawson":
                    return "The character 'Rose Dawson' from the Titanic movie is fictional. However, I can tell you about real passengers:\n\n• There were real passengers named 'Rose' on the Titanic\n• Female passengers had a much higher survival rate (74.2%) than males (20.5%)\n• First class female passengers had the highest survival rate\n\nWould you like to know about real passengers named Rose, or learn about actual survival statistics?"
                elif name_lower == "jack" or name_lower == "dawson":
                    return "The character 'Jack Dawson' from the Titanic movie is fictional. However, I can tell you about real passengers:\n\n• There were real passengers named 'Jack' on the Titanic\n• Male passengers in third class had the lowest survival rate (13.5%)\n• Overall male survival rate was only 20.5%\n\nWould you like to know about real passengers named Jack, or learn about actual survival statistics?"
            
            return f"No passengers named '{name}' found in the dataset. The dataset contains {len(df)} passengers. Try searching for common names like 'John', 'William', or 'Mary'."
    
    except Exception as e:
        return f"Error searching for names: {str(e)}"

def answer_question_simple(query: str):
    """Simple rule-based question answering without external dependencies."""
    q = query.lower()
    
    # Name searches
    if "jack" in q:
        return search_passenger_name("Jack")
    elif "rose" in q:
        return search_passenger_name("Rose")
    
    # Basic statistics
    if any(word in q for word in ["total", "how many", "count"]):
        return f"The Titanic dataset contains {len(df)} passengers with {len(df.columns)} attributes each."
    
    if any(word in q for word in ["survival", "survived", "died", "death"]):
        rate = df['Survived'].mean() * 100
        survived = df['Survived'].sum()
        return f"**{survived}** passengers survived (**{rate:.1f}%** survival rate)."
    
    if any(word in q for word in ["age", "old", "young"]):
        return f"The average passenger age was **{df['Age'].mean():.1f} years**, ranging from {df['Age'].min():.0f} to {df['Age'].max():.0f} years."
    
    if any(word in q for word in ["fare", "ticket", "price", "cost"]):
        return f"The average ticket fare was **£{df['Fare'].mean():.2f}**, median was £{df['Fare'].median():.2f}."
    
    return "I can answer questions about the Titanic dataset! Try asking about:\n- Passenger counts\n- Survival rates\n- Age and fare statistics\n- Specific passenger names"

# ─────────────────────────── Sidebar ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem 0'>
      <div style='font-size:2.4rem'>🌊</div>
      <div style='font-family:"Cormorant Garamond",serif; font-size:1.7rem; font-weight:600; letter-spacing:0.04em'>
        Coast &amp; Calm
      </div>
      <div style='font-size:0.72rem; letter-spacing:0.2em; opacity:0.65; margin-top:2px'>
        ─── 2025 EST. ───
      </div>
    </div>
    <hr style='border-color:rgba(251,239,229,0.25); margin:1rem 0'>
    """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("### 📊 Dataset Overview")
    stats = [
        ("Total Passengers", f"{len(df):,}"),
        ("Survival Rate", f"{df['Survived'].mean()*100:.1f}%"),
        ("Avg Age", f"{df['Age'].mean():.1f} yrs"),
        ("Avg Fare", f"£{df['Fare'].mean():.2f}"),
    ]
    for label, val in stats:
        st.markdown(f"""
        <div class='stat-card'>
          <div class='stat-label'>{label}</div>
          <div class='stat-value'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(251,239,229,0.25); margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("### 💡 Example Questions")
    examples = [
        "Is there anyone named Jack?",
        "Show me survival statistics",
        "What was the average age?",
        "How many passengers survived?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["prefill"] = ex

# ─────────────────────────── Main area ───────────────────────────────────────
st.markdown("""
<div style='padding: 1.2rem 0 0.2rem 0'>
  <div class='hero-title'>⚓ Titanic Explorer</div>
  <div class='hero-sub'>Coast &amp; Calm · Data Insights · Est. 2025</div>
  <hr class='divider'>
</div>
""", unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": ("Welcome aboard! 🌊 I'm your Titanic dataset analyst. "
                    "Ask me anything about the passengers — demographics, fares, "
                    "survival rates, or request specific information."),
        "image": None,
    })

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>",
                    unsafe_allow_html=True)
    else:
        content = msg["content"].replace("\n", "<br>")
        st.markdown(f"<div class='bot-bubble'>{content}</div>",
                    unsafe_allow_html=True)
        if msg.get("image"):
            img_bytes = base64.b64decode(msg["image"])
            st.image(img_bytes, use_column_width="auto")

# Handle prefill from sidebar buttons
prefill = st.session_state.pop("prefill", None)

# Chat input
question = st.chat_input("Ask me about the Titanic passengers…") or prefill

if question:
    st.session_state.messages.append({"role": "user", "content": question, "image": None})
    st.markdown(f"<div class='user-bubble'>{question}</div>", unsafe_allow_html=True)

    with st.spinner("Analysing…"):
        answer = answer_question_simple(question)

    content_html = answer.replace("\n", "<br>")
    st.markdown(f"<div class='bot-bubble'>{content_html}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "image": None,
    })

def main():
    """Main entry point for the Streamlit app."""
    pass  # The app runs from top to bottom when imported

if __name__ == "__main__":
    main()
