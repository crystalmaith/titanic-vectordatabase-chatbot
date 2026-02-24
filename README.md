# 🌊 Titanic Chat Agent — Coast & Calm

A beautiful Titanic dataset explorer built with Streamlit + FastAPI + LangChain, themed around the **Coast & Calm** brand palette.

## Palette
| Name | Hex |
|---|---|
| Ocean Deep Blue | `#5D768B` |
| Warm Sandy Beige | `#C8B39B` |
| Ivory Breeze | `#FBEFE5` |

---

## 🚀 Quick Start (Streamlit Only)

The Streamlit app works standalone (no API key needed) with a built-in rule-based analysis engine.

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Full LangChain Mode (FastAPI + Claude)

### 1. Start the FastAPI backend
```bash
uvicorn backend:app --reload --port 8000
```

### 2. Start Streamlit
```bash
streamlit run app.py
```

### 3. Configure in UI
- Toggle **"Use FastAPI Backend"** in the sidebar
- Enter your **Anthropic API Key**
- The backend URL defaults to `http://localhost:8000`

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this folder to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select the repo, branch, and set **Main file: `app.py`**
4. In **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy** — your app will be live in ~2 minutes!

> **Note**: For full LangChain agent mode on Streamlit Cloud, you also need to deploy the FastAPI backend separately (e.g., on Railway, Render, or Fly.io) and update the Backend URL in the sidebar.

---

## 📁 Project Structure

```
titanic_agent/
├── app.py              # Streamlit frontend
├── backend.py          # FastAPI + LangChain agent
├── titanic.csv         # Dataset
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml     # Theme (Coast & Calm colours)
```

---

## 💬 Example Questions

- "What percentage of passengers were male?"
- "Show me a histogram of passenger ages"
- "What was the average ticket fare?"
- "How many passengers embarked from each port?"
- "Show survival rate by gender"
- "Show a pie chart of passenger classes"

---

## 🏗️ Architecture

```
User → Streamlit (app.py)
          │
          ├── Built-in engine (no API key needed)
          │
          └── FastAPI (backend.py)
                    │
                    └── LangChain ReAct Agent
                              │
                              ├── get_basic_stats tool
                              ├── query_dataframe tool
                              ├── plot_histogram tool
                              ├── plot_bar tool
                              ├── plot_survival_by_group tool
                              └── plot_pie tool
```
