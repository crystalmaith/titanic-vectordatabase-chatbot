"""
Titanic Vector Database Chat — Coast & Calm Theme
Builds a FAISS vector store from the Titanic CSV and answers any question
using OpenAI embeddings + GPT-4o RAG pipeline.
"""

import os, io, json, re, textwrap, hashlib, base64
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic · Vector Explorer",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ──────────────────────────────────────────────────────────────────
OCEAN   = "#5D768B"
BEIGE   = "#C8B39B"
IVORY   = "#FBEFE5"
DARK    = "#3D5468"
LIGHT   = "#EEE4DA"
ACCENT  = "#8FB3C9"

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;1,500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
  font-family: 'DM Sans', sans-serif;
  background: {IVORY};
  color: {DARK};
}}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
  background: linear-gradient(175deg, {DARK} 0%, {OCEAN} 60%, #4a6478 100%) !important;
  border-right: none;
}}
[data-testid="stSidebar"] * {{ color: {IVORY} !important; }}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {{
  background: rgba(255,255,255,0.1) !important;
  border: 1px solid rgba(251,239,229,0.35) !important;
  color: {IVORY} !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
}}
[data-testid="stSidebar"] input::placeholder {{ color: rgba(251,239,229,0.45) !important; }}
[data-testid="stSidebar"] .stSelectbox > div {{ background: rgba(255,255,255,0.1) !important; }}

/* ── hero ── */
.hero {{
  padding: 1.4rem 0 0.6rem 0;
}}
.hero-title {{
  font-family: 'Playfair Display', serif;
  font-size: 2.8rem;
  font-weight: 700;
  color: {OCEAN};
  letter-spacing: -0.01em;
  line-height: 1.1;
}}
.hero-em {{
  font-style: italic;
  color: {BEIGE};
}}
.hero-sub {{
  font-size: 0.75rem;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: {BEIGE};
  margin-top: 4px;
}}
.divider {{
  border: none;
  border-top: 1.5px solid {BEIGE};
  margin: 0.8rem 0 1.4rem 0;
  opacity: 0.5;
}}

/* ── chat bubbles ── */
.user-wrap {{ display: flex; justify-content: flex-end; margin-bottom: 12px; }}
.user-bubble {{
  background: linear-gradient(135deg, {OCEAN}, {DARK});
  color: {IVORY};
  padding: 13px 18px;
  border-radius: 18px 18px 3px 18px;
  max-width: 75%;
  font-size: 0.93rem;
  line-height: 1.55;
  box-shadow: 0 3px 12px rgba(61,84,104,0.22);
}}
.bot-wrap {{ display: flex; justify-content: flex-start; margin-bottom: 12px; }}
.bot-bubble {{
  background: {LIGHT};
  color: {DARK};
  padding: 14px 20px;
  border-radius: 18px 18px 18px 3px;
  max-width: 80%;
  font-size: 0.93rem;
  line-height: 1.65;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  border-left: 3px solid {OCEAN};
}}
.bot-bubble b, .bot-bubble strong {{ color: {OCEAN}; }}

/* ── stat cards ── */
.stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 0.5rem 0 1rem 0; }}
.stat-card {{
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(251,239,229,0.2);
  border-radius: 10px;
  padding: 10px 12px;
}}
.stat-label {{
  font-size: 0.65rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: {BEIGE};
  margin-bottom: 3px;
}}
.stat-val {{
  font-family: 'Playfair Display', serif;
  font-size: 1.4rem;
  font-weight: 700;
  color: {IVORY};
}}

/* ── badge ── */
.badge {{
  display: inline-block;
  background: rgba(143,179,201,0.25);
  border: 1px solid {ACCENT};
  color: {IVORY};
  border-radius: 20px;
  padding: 3px 10px;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  margin: 3px 3px 3px 0;
}}

/* ── index status ── */
.index-ready {{
  background: rgba(100,180,120,0.15);
  border: 1px solid rgba(100,180,120,0.4);
  color: {IVORY};
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.82rem;
}}
.index-waiting {{
  background: rgba(200,179,155,0.2);
  border: 1px solid rgba(200,179,155,0.4);
  color: {IVORY};
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.82rem;
}}

/* ── input ── */
[data-testid="stChatInput"] textarea {{
  background: {LIGHT} !important;
  border: 1.5px solid {BEIGE} !important;
  color: {DARK} !important;
  border-radius: 14px !important;
  font-family: 'DM Sans', sans-serif !important;
}}
[data-testid="stChatInput"] textarea:focus {{
  border-color: {OCEAN} !important;
  box-shadow: 0 0 0 3px rgba(93,118,139,0.12) !important;
}}
[data-testid="stChatInput"] button {{
  background: {OCEAN} !important;
  border-radius: 10px !important;
}}

/* ── example buttons ── */
div[data-testid="stButton"] > button {{
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(251,239,229,0.3) !important;
  color: {IVORY} !important;
  border-radius: 8px !important;
  font-size: 0.8rem !important;
  text-align: left !important;
  padding: 6px 12px !important;
  transition: all 0.2s !important;
}}
div[data-testid="stButton"] > button:hover {{
  background: rgba(255,255,255,0.16) !important;
  border-color: {BEIGE} !important;
}}

#MainMenu, header, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Load Dataset ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "titanic.csv")
df = pd.read_csv(DATA_PATH)


# ── Build rich text documents from the dataset ───────────────────────────────
@st.cache_data
def build_documents(csv_path: str):
    """
    Convert every row + aggregate stats into a list of text chunks
    that will be embedded into the vector store.
    """
    df = pd.read_csv(csv_path)
    docs = []

    # 1. One document per passenger row
    for _, row in df.iterrows():
        survived = "survived" if row["Survived"] == 1 else "did not survive"
        sex = row["Sex"]
        age = f"{row['Age']:.0f} years old" if pd.notna(row["Age"]) else "unknown age"
        pclass_map = {1: "first class", 2: "second class", 3: "third class"}
        pclass = pclass_map.get(int(row["Pclass"]), "unknown class")
        fare = f"£{row['Fare']:.2f}" if pd.notna(row["Fare"]) else "unknown fare"
        port_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
        port = port_map.get(str(row["Embarked"]), "unknown port")
        cabin = row["Cabin"] if pd.notna(row["Cabin"]) else "no cabin recorded"
        siblings = int(row["SibSp"])
        parents = int(row["Parch"])

        text = (
            f"Passenger {int(row['PassengerId'])}: {row['Name']} was a {age} {sex} travelling in {pclass}. "
            f"They {survived}. Ticket fare: {fare}. "
            f"Embarked from {port}. Cabin: {cabin}. "
            f"Travelling with {siblings} sibling(s)/spouse and {parents} parent(s)/child(ren). "
            f"Ticket number: {row['Ticket']}."
        )
        docs.append(text)

    # 2. Aggregate statistics documents
    total = len(df)
    survived_count = df["Survived"].sum()
    survival_rate = df["Survived"].mean() * 100
    avg_age = df["Age"].mean()
    avg_fare = df["Fare"].mean()
    median_fare = df["Fare"].median()
    male_count = (df["Sex"] == "male").sum()
    female_count = (df["Sex"] == "female").sum()

    docs.append(
        f"OVERALL STATISTICS: The Titanic dataset contains {total} passengers. "
        f"{survived_count} survived ({survival_rate:.1f}%). "
        f"{total - survived_count} did not survive ({100 - survival_rate:.1f}%). "
        f"There were {male_count} male passengers ({male_count/total*100:.1f}%) and "
        f"{female_count} female passengers ({female_count/total*100:.1f}%). "
        f"Average age: {avg_age:.1f} years. "
        f"Average fare: £{avg_fare:.2f}. Median fare: £{median_fare:.2f}."
    )

    # 3. Class breakdown
    for cls in [1, 2, 3]:
        sub = df[df["Pclass"] == cls]
        sr = sub["Survived"].mean() * 100
        name = {1: "First", 2: "Second", 3: "Third"}[cls]
        docs.append(
            f"CLASS BREAKDOWN - {name} Class (Pclass={cls}): "
            f"{len(sub)} passengers, {sub['Survived'].sum()} survived ({sr:.1f}% survival rate). "
            f"Average fare: £{sub['Fare'].mean():.2f}. "
            f"Average age: {sub['Age'].mean():.1f} years. "
            f"{(sub['Sex']=='male').sum()} male, {(sub['Sex']=='female').sum()} female."
        )

    # 4. Gender breakdown
    for sex in ["male", "female"]:
        sub = df[df["Sex"] == sex]
        sr = sub["Survived"].mean() * 100
        docs.append(
            f"GENDER BREAKDOWN - {sex.upper()} passengers: "
            f"{len(sub)} total, {sub['Survived'].sum()} survived ({sr:.1f}% survival rate). "
            f"Average age: {sub['Age'].mean():.1f} years. "
            f"Average fare: £{sub['Fare'].mean():.2f}. "
            f"Class 1: {(sub['Pclass']==1).sum()}, Class 2: {(sub['Pclass']==2).sum()}, "
            f"Class 3: {(sub['Pclass']==3).sum()}."
        )

    # 5. Embarkation port breakdown
    port_map_full = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    for code, name in port_map_full.items():
        sub = df[df["Embarked"] == code]
        if len(sub) == 0:
            continue
        sr = sub["Survived"].mean() * 100
        docs.append(
            f"EMBARKATION PORT - {name} ({code}): "
            f"{len(sub)} passengers boarded from {name}. "
            f"{sub['Survived'].sum()} survived ({sr:.1f}% survival rate). "
            f"Average fare: £{sub['Fare'].mean():.2f}. "
            f"Class distribution: 1st={( sub['Pclass']==1).sum()}, "
            f"2nd={(sub['Pclass']==2).sum()}, 3rd={(sub['Pclass']==3).sum()}."
        )

    # 6. Age-group breakdown
    bins = [(0, 12, "children (0-12)"), (13, 17, "teenagers (13-17)"),
            (18, 35, "young adults (18-35)"), (36, 60, "middle-aged adults (36-60)"),
            (61, 120, "seniors (61+)")]
    for lo, hi, label in bins:
        sub = df[(df["Age"] >= lo) & (df["Age"] <= hi)]
        if len(sub) == 0:
            continue
        sr = sub["Survived"].mean() * 100
        docs.append(
            f"AGE GROUP - {label.upper()}: "
            f"{len(sub)} passengers, {sub['Survived'].sum()} survived ({sr:.1f}% survival rate). "
            f"{(sub['Sex']=='male').sum()} male, {(sub['Sex']=='female').sum()} female."
        )

    # 7. Fare quartile breakdown
    quartiles = df["Fare"].quantile([0.25, 0.5, 0.75])
    docs.append(
        f"FARE STATISTICS: Minimum fare: £{df['Fare'].min():.2f}. "
        f"25th percentile: £{quartiles[0.25]:.2f}. "
        f"Median (50th): £{quartiles[0.5]:.2f}. "
        f"75th percentile: £{quartiles[0.75]:.2f}. "
        f"Maximum fare: £{df['Fare'].max():.2f}. "
        f"Average fare: £{df['Fare'].mean():.2f}."
    )

    # 8. Family size analysis
    df_tmp = df.copy()
    df_tmp["FamilySize"] = df_tmp["SibSp"] + df_tmp["Parch"]
    for fs in sorted(df_tmp["FamilySize"].unique()):
        sub = df_tmp[df_tmp["FamilySize"] == fs]
        sr = sub["Survived"].mean() * 100
        label = "alone" if fs == 0 else f"with {fs} family member(s)"
        docs.append(
            f"FAMILY SIZE {fs} (travelling {label}): "
            f"{len(sub)} passengers, survival rate {sr:.1f}%."
        )

    # 9. Notable passengers (highest/lowest fares, oldest, youngest)
    docs.append(
        f"NOTABLE PASSENGERS: "
        f"Most expensive ticket: {df.loc[df['Fare'].idxmax(), 'Name']} paid £{df['Fare'].max():.2f}. "
        f"Cheapest non-zero ticket: {df[df['Fare']>0].loc[df[df['Fare']>0]['Fare'].idxmin(), 'Name']} paid £{df[df['Fare']>0]['Fare'].min():.2f}. "
        f"Oldest passenger: {df.loc[df['Age'].idxmax(), 'Name']} aged {df['Age'].max():.0f}. "
        f"Youngest passenger: {df.loc[df['Age'].idxmin(), 'Name']} aged {df['Age'].min():.0f}."
    )

    return docs


# ── Vector Store (pure numpy cosine similarity — no FAISS dependency) ─────────
def cosine_similarity_matrix(query_vec, doc_vecs):
    """Compute cosine similarity between query and all doc vectors."""
    q = np.array(query_vec)
    D = np.array(doc_vecs)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    D_norms = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-10)
    return D_norms @ q_norm


def get_embeddings_batch(texts, client, model="text-embedding-3-small", batch_size=50):
    """Get embeddings for a list of texts in batches."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([r.embedding for r in response.data])
    return all_embeddings


@st.cache_resource
def build_vector_store(api_key_hash: str, csv_path: str):
    """
    Build the vector store. Cached by api_key_hash so it only runs once per key.
    Returns (documents, embeddings_array) or raises on error.
    """
    from openai import OpenAI
    # Re-derive key from session state (cache can't hold secrets)
    api_key = st.session_state.get("openai_key", "")
    client = OpenAI(api_key=api_key)

    docs = build_documents(csv_path)
    embeddings = get_embeddings_batch(docs, client)
    return docs, np.array(embeddings, dtype=np.float32)


def retrieve(query: str, docs, emb_matrix, client, k=8):
    """Retrieve top-k relevant documents for the query."""
    response = client.embeddings.create(input=[query], model="text-embedding-3-small")
    q_vec = np.array(response.data[0].embedding, dtype=np.float32)
    scores = cosine_similarity_matrix(q_vec, emb_matrix)
    top_k = np.argsort(scores)[::-1][:k]
    return [docs[i] for i in top_k], [float(scores[i]) for i in top_k]


def rag_answer(question: str, docs, emb_matrix, client, model="gpt-4o"):
    """RAG pipeline: retrieve → augment prompt → generate answer."""
    relevant_docs, scores = retrieve(question, docs, emb_matrix, client)
    context = "\n\n".join(relevant_docs)

    system_prompt = """You are a brilliant data analyst and historian specializing in the Titanic disaster.
You have access to a vector database built from the Titanic passenger dataset.
Use ONLY the provided context to answer questions accurately.
Be conversational, friendly, and precise. Use **bold** for key numbers and names.
If the question asks for a list, present it clearly.
If something cannot be determined from the context, say so honestly.
Never make up data."""

    user_prompt = f"""Context from the Titanic vector database:
---
{context}
---

Question: {question}

Answer based on the context above:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content, relevant_docs[:3], scores[:3]


# ── Chart helpers ─────────────────────────────────────────────────────────────
PALETTE = [OCEAN, BEIGE, ACCENT, "#D9C5AF", DARK]

def _apply_theme(fig):
    fig.patch.set_facecolor(IVORY)
    for ax in fig.axes:
        ax.set_facecolor(LIGHT)
        ax.tick_params(colors=OCEAN, labelsize=9)
        ax.xaxis.label.set_color(OCEAN)
        ax.yaxis.label.set_color(OCEAN)
        ax.title.set_color(OCEAN)
        for spine in ax.spines.values():
            spine.set_edgecolor(BEIGE)

def maybe_plot(question: str):
    """Generate a chart if the question seems to ask for one."""
    q = question.lower()
    fig = None
    if any(w in q for w in ["histogram", "distribution", "spread"]):
        for col in ["age", "fare", "sibsp", "parch"]:
            if col in q:
                actual = next((c for c in df.columns if c.lower() == col), None)
                if actual:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.hist(df[actual].dropna(), bins=24, color=OCEAN, edgecolor=IVORY, lw=0.5)
                    ax.set_xlabel(actual); ax.set_ylabel("Count")
                    ax.set_title(f"Distribution of {actual}", fontweight="bold", fontsize=13)
                    _apply_theme(fig)
                    break
    elif any(w in q for w in ["bar chart", "bar graph", "breakdown", "by class", "by gender", "by sex", "by port", "by embark"]):
        for col, label in [("Pclass","Passenger Class"), ("Sex","Gender"), ("Embarked","Port")]:
            if any(k in q for k in [col.lower(), label.lower()]):
                vc = df[col].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(vc.index.astype(str), vc.values, color=PALETTE[:len(vc)], edgecolor=IVORY, lw=0.5)
                ax.set_xlabel(label); ax.set_ylabel("Count")
                ax.set_title(f"Passengers by {label}", fontweight="bold", fontsize=13)
                for i, (x, v) in enumerate(zip(vc.index, vc.values)):
                    ax.text(i, v + 1, str(v), ha="center", color=OCEAN, fontsize=9)
                _apply_theme(fig)
                break
    elif any(w in q for w in ["pie", "proportion", "percentage breakdown"]):
        for col, label in [("Sex","Gender"), ("Pclass","Class"), ("Embarked","Port"), ("Survived","Survival")]:
            if any(k in q for k in [col.lower(), label.lower()]):
                vc = df[col].value_counts()
                fig, ax = plt.subplots(figsize=(5.5, 5.5))
                ax.pie(vc.values, labels=vc.index.astype(str), colors=PALETTE[:len(vc)],
                       autopct="%1.1f%%", startangle=140,
                       textprops={"color": DARK, "fontsize": 10},
                       wedgeprops={"edgecolor": IVORY, "lw": 1.5})
                ax.set_title(f"{label} Distribution", fontsize=13, fontweight="bold", color=OCEAN)
                fig.patch.set_facecolor(IVORY)
                break
    elif "survival" in q and any(w in q for w in ["by", "across", "per"]):
        for col, label in [("Sex","Gender"), ("Pclass","Class"), ("Embarked","Port")]:
            if any(k in q for k in [col.lower(), label.lower()]):
                grp = df.groupby(col)["Survived"].mean() * 100
                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.bar(grp.index.astype(str), grp.values,
                              color=PALETTE[:len(grp)], edgecolor=IVORY, lw=0.5)
                ax.set_xlabel(label); ax.set_ylabel("Survival Rate (%)")
                ax.set_title(f"Survival Rate by {label}", fontweight="bold", fontsize=13)
                ax.set_ylim(0, 100)
                for bar, val in zip(bars, grp.values):
                    ax.text(bar.get_x()+bar.get_width()/2, val+1.5, f"{val:.1f}%",
                            ha="center", color=DARK, fontsize=9)
                _apply_theme(fig)
                break
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style='text-align:center; padding:1.8rem 0 1rem'>
      <div style='font-size:2.8rem'>⚓</div>
      <div style='font-family:"Playfair Display",serif; font-size:1.9rem; font-weight:700; letter-spacing:0.02em; color:{IVORY}'>
        Coast &amp; Calm
      </div>
      <div style='font-size:0.68rem; letter-spacing:0.24em; opacity:0.6; margin-top:3px; color:{BEIGE}'>
        ─── TITANIC EXPLORER ───
      </div>
    </div>
    <hr style='border-color:rgba(251,239,229,0.2); margin: 0 0 1rem 0'>
    """, unsafe_allow_html=True)

    # API key
    st.markdown("### 🔑 OpenAI API Key")
    api_key_input = st.text_input("", type="password", placeholder="sk-...",
                                   label_visibility="collapsed")
    if api_key_input:
        st.session_state["openai_key"] = api_key_input

    model_choice = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                                 index=0, label_visibility="visible")

    st.markdown("<hr style='border-color:rgba(251,239,229,0.2); margin:1rem 0'>", unsafe_allow_html=True)

    # Index status
    has_key = bool(st.session_state.get("openai_key", ""))
    has_index = "vector_docs" in st.session_state

    if has_index:
        n_docs = len(st.session_state["vector_docs"])
        st.markdown(f"""<div class='index-ready'>
          ✅ <b>Vector DB Ready</b><br>
          <span style='font-size:0.78rem; opacity:0.85'>{n_docs} document chunks indexed</span>
        </div>""", unsafe_allow_html=True)
    elif has_key:
        st.markdown(f"""<div class='index-waiting'>
          ⏳ <b>Ready to index</b><br>
          <span style='font-size:0.78rem; opacity:0.85'>Press "Build Vector DB" to start</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='index-waiting'>
          🔒 <b>Enter API key first</b><br>
          <span style='font-size:0.78rem; opacity:0.85'>OpenAI key required to build index</span>
        </div>""", unsafe_allow_html=True)

    if has_key and not has_index:
        if st.button("🚀 Build Vector DB", use_container_width=True):
            with st.spinner("Embedding 400+ passenger records…"):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=st.session_state["openai_key"])
                    docs = build_documents(DATA_PATH)
                    embeddings = get_embeddings_batch(docs, client)
                    st.session_state["vector_docs"] = docs
                    st.session_state["vector_emb"] = np.array(embeddings, dtype=np.float32)
                    st.success(f"✅ Indexed {len(docs)} chunks!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    if has_index:
        if st.button("🗑️ Clear Index", use_container_width=True):
            del st.session_state["vector_docs"]
            del st.session_state["vector_emb"]
            st.rerun()

    st.markdown("<hr style='border-color:rgba(251,239,229,0.2); margin:1rem 0'>", unsafe_allow_html=True)

    # Dataset stats
    st.markdown("### 📊 Quick Stats")
    st.markdown(f"""
    <div class='stats-grid'>
      <div class='stat-card'>
        <div class='stat-label'>Passengers</div>
        <div class='stat-val'>{len(df)}</div>
      </div>
      <div class='stat-card'>
        <div class='stat-label'>Survived</div>
        <div class='stat-val'>{df["Survived"].sum()}</div>
      </div>
      <div class='stat-card'>
        <div class='stat-label'>Avg Age</div>
        <div class='stat-val'>{df["Age"].mean():.1f}</div>
      </div>
      <div class='stat-card'>
        <div class='stat-label'>Avg Fare</div>
        <div class='stat-val'>£{df["Fare"].mean():.0f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(251,239,229,0.2); margin:1rem 0'>", unsafe_allow_html=True)

    # Examples
    st.markdown("### 💬 Try These")
    examples = [
        "What was the survival rate for women vs men?",
        "Show me the histogram of passenger ages",
        "Who paid the most expensive ticket?",
        "How many passengers embarked from each port?",
        "What % survived in first class vs third class?",
        "Tell me about passengers who were children",
        "What was the average fare for each class?",
        "Show survival rate by gender as a bar chart",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["prefill"] = ex


# ── Main Chat Area ────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <div class='hero-title'>Titanic <span class='hero-em'>Vector</span> Explorer</div>
  <div class='hero-sub'>⚓ &nbsp; Semantic Search · RAG · OpenAI Embeddings · 2025</div>
</div>
<hr class='divider'>
""", unsafe_allow_html=True)

# How it works expander
with st.expander("🧠 How this works"):
    st.markdown(f"""
    <div style='font-size:0.9rem; color:{DARK}; line-height:1.7'>
    <b>1. Document Creation</b> — Every Titanic passenger row is converted into a rich natural-language text chunk,
    plus aggregate statistics (by class, gender, port, age group, fare quartiles, family size).<br><br>
    <b>2. Embedding</b> — OpenAI's <code>text-embedding-3-small</code> model converts each chunk into a
    high-dimensional vector (1536 dimensions), stored in memory as a NumPy matrix.<br><br>
    <b>3. Retrieval</b> — When you ask a question, your query is also embedded, then cosine similarity finds
    the most semantically relevant chunks from the database.<br><br>
    <b>4. Generation</b> — The top-8 retrieved chunks are passed as context to GPT-4o, which synthesizes
    a grounded, accurate answer — a full RAG pipeline.
    </div>
    """, unsafe_allow_html=True)

# Init messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": ("Welcome aboard! 🌊 I'm your Titanic vector database analyst.\n\n"
                    "Enter your **OpenAI API key** in the sidebar and press **Build Vector DB** "
                    "to embed the entire Titanic dataset. Then ask me anything — from individual passengers "
                    "to statistics, survival rates, fares, and more."),
        "image": None,
        "sources": None,
    })

# Render history
for msg in st.session_state.messages:
    content_html = msg["content"].replace("\n", "<br>")
    if msg["role"] == "user":
        st.markdown(f"<div class='user-wrap'><div class='user-bubble'>{content_html}</div></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-wrap'><div class='bot-bubble'>{content_html}</div></div>",
                    unsafe_allow_html=True)
        if msg.get("image"):
            st.image(msg["image"], use_column_width="auto")
        if msg.get("sources"):
            with st.expander("🔍 Retrieved sources", expanded=False):
                for i, (src, score) in enumerate(zip(msg["sources"], msg.get("scores", [])), 1):
                    st.markdown(f"""
                    <div style='background:{LIGHT}; border-radius:8px; padding:10px; margin-bottom:8px;
                         border-left:3px solid {OCEAN}; font-size:0.82rem; color:{DARK}'>
                      <b style='color:{OCEAN}'>#{i} · similarity {score:.3f}</b><br>
                      {src[:300]}{'…' if len(src)>300 else ''}
                    </div>""", unsafe_allow_html=True)

# Prefill from sidebar buttons
prefill = st.session_state.pop("prefill", None)
question = st.chat_input("Ask anything about the Titanic passengers…") or prefill

if question:
    # Show user bubble immediately
    q_html = question.replace("\n", "<br>")
    st.markdown(f"<div class='user-wrap'><div class='user-bubble'>{q_html}</div></div>",
                unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": question})

    if not st.session_state.get("openai_key"):
        answer = "⚠️ Please enter your **OpenAI API key** in the sidebar to use the vector database."
        sources, scores = None, None
        fig = None
    elif "vector_docs" not in st.session_state:
        answer = "⚠️ Please click **Build Vector DB** in the sidebar first to index the dataset."
        sources, scores = None, None
        fig = None
    else:
        with st.spinner("Searching vector database…"):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=st.session_state["openai_key"])
                answer, sources, scores = rag_answer(
                    question,
                    st.session_state["vector_docs"],
                    st.session_state["vector_emb"],
                    client,
                    model=model_choice,
                )
                fig = maybe_plot(question)
            except Exception as e:
                answer = f"❌ Error: {e}"
                sources, scores, fig = None, None, None

    # Render bot response
    answer_html = answer.replace("\n", "<br>")
    st.markdown(f"<div class='bot-wrap'><div class='bot-bubble'>{answer_html}</div></div>",
                unsafe_allow_html=True)

    img_bytes = None
    if fig:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        img_bytes = buf.read()
        plt.close(fig)
        st.image(img_bytes, use_column_width="auto")

    if sources:
        with st.expander("🔍 Retrieved sources", expanded=False):
            for i, (src, score) in enumerate(zip(sources, scores or []), 1):
                st.markdown(f"""
                <div style='background:{LIGHT}; border-radius:8px; padding:10px; margin-bottom:8px;
                     border-left:3px solid {OCEAN}; font-size:0.82rem; color:{DARK}'>
                  <b style='color:{OCEAN}'>#{i} · similarity {score:.3f}</b><br>
                  {src[:300]}{'…' if len(src)>300 else ''}
                </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "image": img_bytes,
        "sources": sources,
        "scores": scores,
    })
