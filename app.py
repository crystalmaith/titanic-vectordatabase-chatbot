"""
Titanic Chat Agent — Streamlit Frontend
Theme: Coast & Calm  |  #5D768B · #C8B39B · #FBEFE5
"""

import base64
import os
import json
import re
import io
import sys
import pickle
from typing import List, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import streamlit as st
import openai
import faiss
from sentence_transformers import SentenceTransformer

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
IVORY       = "#FBEFE5"
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

# ─────────────────────────── Vector Database Setup ─────────────────────────────
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
    
    # Embarkation statistics
    embark_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
    for port in df['Embarked'].dropna().unique():
        port_df = df[df['Embarked'] == port]
        port_name = embark_map.get(port, port)
        documents.append(f"{port_name} had {len(port_df)} passengers embark.")
    
    # Age distribution insights
    age_groups = [
        (0, 12, "children"),
        (13, 18, "teenagers"), 
        (19, 35, "young adults"),
        (36, 60, "adults"),
        (61, 100, "seniors")
    ]
    
    for min_age, max_age, group_name in age_groups:
        group_df = df[(df['Age'] >= min_age) & (df['Age'] <= max_age)]
        if len(group_df) > 0:
            survival_rate = group_df['Survived'].mean() * 100
            documents.append(f"{group_name.capitalize()} (ages {min_age}-{max_age}) had {survival_rate:.1f}% survival rate.")
    
    # Family size statistics
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    for size in [1, 2, 3, 4, 5]:
        size_df = df[df['FamilySize'] == size]
        if len(size_df) > 0:
            survival_rate = size_df['Survived'].mean() * 100
            documents.append(f"Passengers with family size {size} had {survival_rate:.1f}% survival rate.")
    
    return documents

def setup_vector_database():
    """Set up the vector database with dataset documents."""
    if 'vector_db' not in st.session_state:
        with st.spinner("Setting up vector database..."):
            # Load sentence transformer model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create documents
            documents = create_dataset_documents()
            
            # Create embeddings
            embeddings = model.encode(documents)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype('float32'))
            
            # Store in session state
            st.session_state.vector_db = {
                'index': index,
                'documents': documents,
                'model': model
            }

def search_vector_database(query: str, top_k: int = 3) -> List[str]:
    """Search the vector database for relevant documents."""
    if 'vector_db' not in st.session_state:
        setup_vector_database()
    
    vector_db = st.session_state.vector_db
    query_embedding = vector_db['model'].encode([query])
    
    # Search
    distances, indices = vector_db['index'].search(query_embedding.astype('float32'), top_k)
    
    # Return relevant documents
    relevant_docs = [vector_db['documents'][i] for i in indices[0]]
    return relevant_docs

def answer_with_openai(query: str, api_key: str) -> str:
    """Answer question using OpenAI GPT with vector database context."""
    try:
        # Set up OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Get relevant documents
        relevant_docs = search_vector_database(query)
        context = "\n".join(relevant_docs)
        
        # Create prompt
        prompt = f"""You are a helpful assistant specializing in the Titanic dataset. 
Use the following information to answer the user's question about the Titanic dataset.

Context:
{context}

Question: {query}

Provide a clear, concise answer based on the context provided. If the context doesn't contain enough information to answer the question, say so politely."""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in Titanic dataset analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error using OpenAI: {str(e)}"

def search_passenger_name(name: str) -> str:
    """Search for passengers with a specific name in the dataset."""
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
            
            # Try partial matches and similar names
            name_parts = name_lower.split()
            for part in name_parts:
                if len(part) > 2:  # Skip very short parts
                    partial_matches = df[df['Name'].str.lower().str.contains(part, na=False)]
                    if len(partial_matches) > 0:
                        return f"No passengers named exactly '{name}', but found {len(partial_matches)} with '{part}' in their name. Try asking about a more specific name or check the passenger list."
            
            # Suggest common names
            common_names = ["john", "william", "james", "george", "thomas", "charles", "mary", "elizabeth", "sarah", "rose"]
            available_names = []
            for common_name in common_names:
                matches = df[df['Name'].str.lower().str.contains(common_name, na=False)]
                if len(matches) > 0:
                    available_names.append(f"{common_name.capitalize()} ({len(matches)} passengers)")
            
            if available_names:
                return f"No passengers named '{name}' found. However, the dataset has passengers with these common names:\n\n" + "\n".join(f"• {name}" for name in available_names[:5]) + f"\n\nTry asking about one of these names!"
            
            return f"No passengers named '{name}' found in the dataset. The dataset contains 418 passengers. Try searching for common names like 'John', 'Mary', or 'William'."
    
    except Exception as e:
        return f"Error searching for names: {str(e)}"

def answer_with_local_model(query: str) -> str:
    """Answer question using a local model with vector database context (no API key required)."""
    try:
        # Get relevant documents
        relevant_docs = search_vector_database(query)
        context = "\n".join(relevant_docs)
        
        query_lower = query.lower()
        
        # First, check for name searches with better pattern matching
        name_patterns = [
            r"named\s+(\w+)",
            r"called\s+(\w+)", 
            r"name\s+is\s+(\w+)",
            r"who\s+is\s+(\w+)",
            r"was\s+there\s+anyone\s+named\s+(\w+)",
            r"is\s+there\s+anyone\s+named\s+(\w+)",
            r"anyone\s+named\s+(\w+)"
        ]
        
        import re
        for pattern in name_patterns:
            match = re.search(pattern, query_lower)
            if match:
                potential_name = match.group(1).strip('?.!,"')
                if len(potential_name) > 1:
                    return search_passenger_name(potential_name)
        
        # Check for direct name mentions and movie characters
        movie_characters = ["rose dawson", "jack dawson", "rose", "jack", "dawson"]
        common_names = ["jack", "john", "mary", "william", "james", "george", "thomas", "charles", "elizabeth", "sarah", "rose"]
        
        # Check for movie character references first
        for character in movie_characters:
            if character in query_lower:
                return search_passenger_name(character)
        
        # Then check for common names
        for name in common_names:
            if name in query_lower:
                return search_passenger_name(name.capitalize())
        
        # Check if the query is just asking about a name
        words = query_lower.split()
        if len(words) <= 4 and any(word in common_names + movie_characters for word in words):
            for word in words:
                if word in common_names + movie_characters:
                    return search_passenger_name(word.capitalize())
        
        # Check for visualization requests and delegate to the original function
        viz_keywords = ["histogram", "chart", "graph", "plot", "show", "visual", "distribution", "pie", "bar"]
        if any(word in query_lower for word in viz_keywords):
            # Use the original answer_question function for visualizations
            # It returns a tuple (text, image), so we need to handle this differently
            return "VISUALIZATION_REQUEST"  # Special marker to trigger visualization in main logic
        
        # First, try to find exact matches in the context
        if any(word in context.lower() for word in query_lower.split() if len(word) > 2):
            # Extract relevant information from context
            lines = context.split('\n')
            relevant_lines = []
            
            for line in lines:
                line_lower = line.lower()
                # Check if line contains relevant keywords
                if any(word in line_lower for word in query_lower.split() if len(word) > 2):
                    relevant_lines.append(line)
            
            if relevant_lines:
                return "Based on the Titanic dataset:\n" + "\n".join(relevant_lines[:3])
        
        # Handle specific survival questions
        if "survival rate" in query_lower or "survived" in query_lower:
            if "male" in query_lower or "men" in query_lower:
                male_count = df[df['Sex'] == 'male'].shape[0]
                male_survived = df[df['Sex'] == 'male']['Survived'].sum()
                male_rate = (male_survived / male_count) * 100
                return f"Male passengers: {male_count} total, {male_survived} survived ({male_rate:.1f}% survival rate)"
            
            elif "female" in query_lower or "women" in query_lower:
                female_count = df[df['Sex'] == 'female'].shape[0]
                female_survived = df[df['Sex'] == 'female']['Survived'].sum()
                female_rate = (female_survived / female_count) * 100
                return f"Female passengers: {female_count} total, {female_survived} survived ({female_rate:.1f}% survival rate)"
            
            elif "class" in query_lower or "pclass" in query_lower:
                if "first" in query_lower or "1st" in query_lower:
                    class1 = df[df['Pclass'] == 1]
                    rate = (class1['Survived'].sum() / len(class1)) * 100
                    return f"First class passengers: {len(class1)} total, {class1['Survived'].sum()} survived ({rate:.1f}% survival rate)"
                elif "second" in query_lower or "2nd" in query_lower:
                    class2 = df[df['Pclass'] == 2]
                    rate = (class2['Survived'].sum() / len(class2)) * 100
                    return f"Second class passengers: {len(class2)} total, {class2['Survived'].sum()} survived ({rate:.1f}% survival rate)"
                elif "third" in query_lower or "3rd" in query_lower:
                    class3 = df[df['Pclass'] == 3]
                    rate = (class3['Survived'].sum() / len(class3)) * 100
                    return f"Third class passengers: {len(class3)} total, {class3['Survived'].sum()} survived ({rate:.1f}% survival rate)"
            
            else:
                total_survived = df['Survived'].sum()
                total_passengers = len(df)
                survival_rate = (total_survived / total_passengers) * 100
                return f"Overall survival: {total_survived} out of {total_passengers} passengers survived ({survival_rate:.1f}% survival rate)"
        
        # Handle age questions
        if "age" in query_lower:
            avg_age = df['Age'].mean()
            min_age = df['Age'].min()
            max_age = df['Age'].max()
            return f"Passenger ages: Average {avg_age:.1f} years, ranging from {min_age:.0f} to {max_age:.0f} years old"
        
        # Handle fare questions
        if "fare" in query_lower or "ticket" in query_lower or "price" in query_lower:
            avg_fare = df['Fare'].mean()
            median_fare = df['Fare'].median()
            return f"Ticket fares: Average £{avg_fare:.2f}, Median £{median_fare:.2f}"
        
        # Handle class questions
        if "class" in query_lower or "pclass" in query_lower:
            class1 = len(df[df['Pclass'] == 1])
            class2 = len(df[df['Pclass'] == 2])
            class3 = len(df[df['Pclass'] == 3])
            return f"Passenger classes: First class: {class1}, Second class: {class2}, Third class: {class3}"
        
        # Handle gender questions
        if "male" in query_lower or "female" in query_lower or "gender" in query_lower or "sex" in query_lower:
            male_count = df[df['Sex'] == 'male'].shape[0]
            female_count = df[df['Sex'] == 'female'].shape[0]
            male_pct = (male_count / len(df)) * 100
            female_pct = (female_count / len(df)) * 100
            return f"Gender distribution: {male_count} male passengers ({male_pct:.1f}%), {female_count} female passengers ({female_pct:.1f}%)"
        
        # Handle embarkation questions
        if "embark" in query_lower or "port" in query_lower:
            embark_counts = df['Embarked'].value_counts()
            port_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
            result = "Embarkation ports:\n"
            for port, count in embark_counts.items():
                port_name = port_map.get(port, port)
                result += f"• {port_name}: {count} passengers\n"
            return result.strip()
        
        # Default response with specific examples
        return "I can answer specific questions about the Titanic dataset. Try asking:\n\n• Was there anyone named Jack/John/Mary?\n• What was the survival rate for males/females?\n• How many passengers were in each class?\n• What was the average age of passengers?\n• How much did tickets cost?\n• Where did passengers embark from?\n• Show me a histogram of ages\n• Create a bar chart of survival rates\n• Show a pie chart of passenger classes"
    
    except Exception as e:
        return f"Error: {str(e)}"

# ─────────────────────────── Palette helpers ─────────────────────────────────
PALETTE = [OCEAN_BLUE, SANDY_BEIGE, "#8FB3C9", "#D9C5AF", DARK_BLUE]

def _apply_theme(fig):
    fig.patch.set_facecolor(IVORY)
    for ax in fig.axes:
        ax.set_facecolor("#EEE4DA")
        ax.tick_params(colors=OCEAN_BLUE, labelsize=9)
        ax.xaxis.label.set_color(OCEAN_BLUE)
        ax.yaxis.label.set_color(OCEAN_BLUE)
        ax.title.set_color(OCEAN_BLUE)
        for spine in ax.spines.values():
            spine.set_edgecolor(SANDY_BEIGE)

# ─────────────────────────── Inline Analysis Engine ──────────────────────────
# A lightweight rule-based engine so the app can work without the backend running.

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def answer_question(question: str):
    """Return (text_answer, image_b64_or_None) by parsing the question locally."""
    q = question.lower()
    image = None

    # ── Histogram ──
    for col in ["age", "fare", "sibsp", "parch", "pclass"]:
        if col in q and any(w in q for w in ["histogram", "distribution", "spread", "hist"]):
            fig, ax = plt.subplots(figsize=(7, 4))
            data = df[col.capitalize() if col != "sibsp" and col != "parch" else col.upper()
                      if False else col.title()]
            # handle column name casing
            actual = next((c for c in df.columns if c.lower() == col), col)
            ax.hist(df[actual].dropna(), bins=25, color=OCEAN_BLUE, edgecolor=IVORY, lw=0.6)
            ax.set_xlabel(actual); ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {actual}", fontweight="bold", fontsize=13)
            _apply_theme(fig)
            image = _fig_to_b64(fig)
            text = f"Here's the histogram for **{actual}**."
            if actual == "Age":
                text += f" The average age is **{df['Age'].mean():.1f} years**, ranging from {df['Age'].min():.0f} to {df['Age'].max():.0f}."
            elif actual == "Fare":
                text += f" Average fare: **£{df['Fare'].mean():.2f}**, median: £{df['Fare'].median():.2f}."
            return text, image

    # ── Survival by group ──
    for col in ["sex", "pclass", "embarked", "class", "gender"]:
        real = "Sex" if col in ("sex","gender") else "Pclass" if col in ("pclass","class") else "Embarked"
        if col in q and any(w in q for w in ["survival", "survive", "survived"]):
            grp = df.groupby(real)["Survived"].mean() * 100
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(grp.index.astype(str), grp.values,
                          color=PALETTE[:len(grp)], edgecolor=IVORY, lw=0.6)
            ax.set_xlabel(real); ax.set_ylabel("Survival Rate (%)")
            ax.set_title(f"Survival Rate by {real}", fontweight="bold", fontsize=13)
            ax.set_ylim(0, 100)
            for bar, val in zip(bars, grp.values):
                ax.text(bar.get_x()+bar.get_width()/2, val+1, f"{val:.1f}%",
                        ha="center", color=OCEAN_BLUE, fontsize=9)
            _apply_theme(fig)
            image = _fig_to_b64(fig)
            lines = [f"- **{k}**: {v:.1f}%" for k, v in grp.items()]
            return f"Survival rates by **{real}**:\n" + "\n".join(lines), image

    # ── Bar / pie for categorical ──
    for col in ["sex", "gender", "embarked", "pclass", "class"]:
        real = "Sex" if col in ("sex","gender") else "Pclass" if col in ("pclass","class") else "Embarked"
        if col in q:
            counts = df[real].value_counts()
            if any(w in q for w in ["pie", "percent", "proportion", "%"]):
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(counts.values, labels=counts.index.astype(str), colors=PALETTE[:len(counts)],
                       autopct="%1.1f%%", startangle=140,
                       textprops={"color": OCEAN_BLUE, "fontsize": 11},
                       wedgeprops={"edgecolor": IVORY, "lw": 1.5})
                ax.set_title(f"{real} Distribution", fontsize=13, fontweight="bold", color=OCEAN_BLUE)
                fig.patch.set_facecolor(IVORY)
                image = _fig_to_b64(fig)
            else:
                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.bar(counts.index.astype(str), counts.values,
                              color=PALETTE[:len(counts)], edgecolor=IVORY, lw=0.6)
                ax.set_xlabel(real); ax.set_ylabel("Count")
                ax.set_title(f"Passengers by {real}", fontweight="bold", fontsize=13)
                for bar in bars:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                            str(int(bar.get_height())), ha="center", color=OCEAN_BLUE, fontsize=9)
                _apply_theme(fig)
                image = _fig_to_b64(fig)

            pcts = (counts / counts.sum() * 100).round(1)
            lines = [f"- **{k}**: {v} ({pcts[k]}%)" for k, v in counts.items()]
            text = f"**{real}** breakdown:\n" + "\n".join(lines)
            return text, image

    # ── Specific stat questions ──
    if any(w in q for w in ["male", "female", "gender", "sex"]):
        vc = df["Sex"].value_counts()
        tot = len(df)
        return (f"The Titanic dataset has **{vc.get('male',0)}** male passengers "
                f"(**{vc.get('male',0)/tot*100:.1f}%**) and **{vc.get('female',0)}** female "
                f"(**{vc.get('female',0)/tot*100:.1f}%**)."), None

    if any(w in q for w in ["age", "old", "young"]):
        return (f"The average passenger age was **{df['Age'].mean():.1f} years**. "
                f"The youngest was {df['Age'].min():.0f} and the oldest was {df['Age'].max():.0f}."), None

    if any(w in q for w in ["fare", "ticket", "price", "cost"]):
        return (f"The average ticket fare was **£{df['Fare'].mean():.2f}**. "
                f"The median was £{df['Fare'].median():.2f}, ranging from £{df['Fare'].min():.2f} "
                f"to £{df['Fare'].max():.2f}."), None

    if any(w in q for w in ["embark", "port", "southampton", "cherbourg", "queenstown"]):
        ec = df["Embarked"].value_counts()
        port_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
        lines = [f"- **{port_map.get(k, k)}**: {v} passengers" for k, v in ec.items()]
        return "Passengers by port of embarkation:\n" + "\n".join(lines), None

    if any(w in q for w in ["surviv", "died", "death", "alive"]):
        rate = df["Survived"].mean() * 100
        surv = df["Survived"].sum()
        died = len(df) - surv
        return (f"**{surv}** passengers survived (**{rate:.1f}%**) and "
                f"**{died}** did not survive (**{100-rate:.1f}%**)."), None

    if any(w in q for w in ["total", "how many", "count", "passengers"]):
        return f"The dataset contains **{len(df)} passengers** with {len(df.columns)} attributes each.", None

    if any(w in q for w in ["class", "pclass"]):
        vc = df["Pclass"].value_counts().sort_index()
        lines = [f"- **Class {k}**: {v} passengers" for k, v in vc.items()]
        return "Passenger class distribution:\n" + "\n".join(lines), None

    # ── Fallback ──
    return (
        "I can answer questions about the Titanic dataset! Try asking about:\n"
        "- Passenger demographics (age, gender, class)\n"
        "- Ticket fares\n"
        "- Survival rates\n"
        "- Embarkation ports\n"
        "- Or ask for charts: 'show a histogram of ages'"), None


def answer_with_api(question: str, api_key: str, backend_url: str):
    """Call FastAPI backend (if running)."""
    try:
        resp = requests.post(
            f"{backend_url}/chat",
            json={"question": question, "api_key": api_key},
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("answer", ""), data.get("image_b64")
    except Exception:
        pass
    return None, None


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

    st.markdown("### 🔑 API Settings")
    use_backend = st.toggle("Use FastAPI Backend (LangChain)", value=False,
                            help="Enable to use the full LangChain agent via the FastAPI backend")
    api_key = ""
    backend_url = "http://localhost:8000"
    if use_backend:
        api_key = st.text_input("Anthropic API Key", type="password",
                                placeholder="sk-ant-...")
        backend_url = st.text_input("Backend URL", value="http://localhost:8000")

    # OpenAI settings for vector database
    st.markdown("### 🤖 AI Model Settings")
    
    # Model selection
    model_option = st.selectbox(
        "Choose AI Model:",
        ["Local Model (Free)", "OpenAI GPT-3.5 (API Key Required)"],
        help="Local model works without API keys but has limited capabilities. OpenAI provides more natural responses."
    )
    
    use_openai = model_option == "OpenAI GPT-3.5 (API Key Required)"
    openai_key = ""
    
    if use_openai:
        openai_key = st.text_input("OpenAI API Key", type="password",
                                   placeholder="sk-...")
        st.info("💡 Don't have an OpenAI API key? Use the Local Model option above for free access!")
    else:
        st.success("✅ Using Local Model - No API key required!")

    st.markdown("<hr style='border-color:rgba(251,239,229,0.25); margin:1rem 0'>", unsafe_allow_html=True)

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
        "What % of passengers were male?",
        "Show histogram of passenger ages",
        "Average ticket fare?",
        "Passengers from each port?",
        "Survival rate by gender?",
        "Show survival rate by class",
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
                    "survival rates, or request a visualization."),
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
        answer, image = None, None
        if use_openai and openai_key:
            # Use OpenAI with vector database
            answer = answer_with_openai(question, openai_key)
        elif not use_openai:
            # Use local model with vector database
            local_answer = answer_with_local_model(question)
            if local_answer == "VISUALIZATION_REQUEST":
                # Handle visualization requests
                answer, image = answer_question(question)
            else:
                answer = local_answer
        elif use_backend and api_key:
            # Use FastAPI backend
            answer, image = answer_with_api(question, api_key, backend_url)
        else:
            # Use local rule-based engine as final fallback
            answer, image = answer_question(question)

    content_html = answer.replace("\n", "<br>")
    st.markdown(f"<div class='bot-bubble'>{content_html}</div>", unsafe_allow_html=True)
    if image:
        st.image(base64.b64decode(image), use_column_width="auto")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "image": image,
    })
