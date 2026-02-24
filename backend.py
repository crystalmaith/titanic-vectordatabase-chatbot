"""
Titanic Dataset Chat Agent - FastAPI Backend
Uses LangChain agent with pandas tools to analyze the Titanic dataset
"""

import os
import io
import base64
import json
import re
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

# ── Palette from the moodboard ───────────────────────────────────────────────
OCEAN_BLUE  = "#5D768B"
SANDY_BEIGE = "#C8B39B"
IVORY       = "#FBEFE5"

# ── Load dataset ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "titanic.csv")
df = pd.read_csv(DATA_PATH)

# Pre-compute a few friendly stats once
_STATS = {
    "total_passengers": len(df),
    "male_pct": round(df["Sex"].value_counts(normalize=True).get("male", 0) * 100, 1),
    "female_pct": round(df["Sex"].value_counts(normalize=True).get("female", 0) * 100, 1),
    "avg_age": round(df["Age"].mean(), 1),
    "avg_fare": round(df["Fare"].mean(), 2),
    "survival_rate": round(df["Survived"].mean() * 100, 1),
}


# ── Matplotlib theme ──────────────────────────────────────────────────────────
def _apply_theme(fig, ax_list=None):
    fig.patch.set_facecolor(IVORY)
    axes = ax_list or fig.axes
    for ax in axes:
        ax.set_facecolor("#EEE4DA")
        ax.tick_params(colors=OCEAN_BLUE)
        ax.xaxis.label.set_color(OCEAN_BLUE)
        ax.yaxis.label.set_color(OCEAN_BLUE)
        ax.title.set_color(OCEAN_BLUE)
        for spine in ax.spines.values():
            spine.set_edgecolor(SANDY_BEIGE)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


# ── LangChain Tools ───────────────────────────────────────────────────────────

@tool
def get_basic_stats(query: str) -> str:
    """Return basic statistics about the Titanic dataset.
    Use for questions about total passengers, averages, percentages, counts."""
    return json.dumps(_STATS)


@tool
def query_dataframe(pandas_query: str) -> str:
    """Run a simple pandas expression on the Titanic DataFrame (variable name: df).
    Return the result as a string. Only read operations are allowed.
    Example inputs: 'df["Sex"].value_counts()', 'df.groupby("Pclass")["Survived"].mean()'"""
    try:
        # Safety guard – no writes
        for banned in ["to_csv", "to_json", "to_excel", "drop", "del ", "__"]:
            if banned in pandas_query:
                return "Operation not allowed."
        result = eval(pandas_query, {"df": df, "pd": pd, "np": np})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


@tool
def plot_histogram(column: str) -> str:
    """Create a histogram for a numeric column of the Titanic dataset.
    Valid columns: Age, Fare, SibSp, Parch, Pclass.
    Returns a base64-encoded PNG image string prefixed with 'IMAGE:'."""
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {list(df.columns)}"
    numeric = df[column].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(numeric, bins=25, color=OCEAN_BLUE, edgecolor=IVORY, linewidth=0.6)
    ax.set_xlabel(column, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Distribution of {column}", fontsize=13, fontweight="bold")
    _apply_theme(fig)
    return "IMAGE:" + _fig_to_b64(fig)


@tool
def plot_bar(column: str) -> str:
    """Create a bar chart for a categorical column of the Titanic dataset.
    Valid columns: Sex, Embarked, Pclass, Survived.
    Returns a base64-encoded PNG image string prefixed with 'IMAGE:'."""
    if column not in df.columns:
        return f"Column '{column}' not found."
    counts = df[column].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=[OCEAN_BLUE, SANDY_BEIGE, "#8FB3C9", "#D9C5AF"][:len(counts)],
                  edgecolor=IVORY, linewidth=0.6)
    ax.set_xlabel(column, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Passengers by {column}", fontsize=13, fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom",
                color=OCEAN_BLUE, fontsize=9)
    _apply_theme(fig)
    return "IMAGE:" + _fig_to_b64(fig)


@tool
def plot_survival_by_group(groupby_column: str) -> str:
    """Show survival rate broken down by a categorical column.
    Valid columns: Sex, Pclass, Embarked.
    Returns a base64-encoded PNG image string prefixed with 'IMAGE:'."""
    if groupby_column not in df.columns:
        return f"Column '{groupby_column}' not found."
    grp = df.groupby(groupby_column)["Survived"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(grp.index.astype(str), grp.values,
                  color=[OCEAN_BLUE, SANDY_BEIGE, "#8FB3C9", "#D9C5AF"][:len(grp)],
                  edgecolor=IVORY, linewidth=0.6)
    ax.set_xlabel(groupby_column, fontsize=11)
    ax.set_ylabel("Survival Rate (%)", fontsize=11)
    ax.set_title(f"Survival Rate by {groupby_column}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f"{val:.1f}%", ha="center", va="bottom",
                color=OCEAN_BLUE, fontsize=9)
    _apply_theme(fig)
    return "IMAGE:" + _fig_to_b64(fig)


@tool
def plot_pie(column: str) -> str:
    """Create a pie chart for a categorical column.
    Valid columns: Sex, Survived, Pclass, Embarked.
    Returns a base64-encoded PNG image string prefixed with 'IMAGE:'."""
    if column not in df.columns:
        return f"Column '{column}' not found."
    counts = df[column].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = [OCEAN_BLUE, SANDY_BEIGE, "#8FB3C9", "#D9C5AF"][:len(counts)]
    ax.pie(counts.values, labels=counts.index.astype(str),
           colors=colors, autopct="%1.1f%%", startangle=140,
           textprops={"color": OCEAN_BLUE, "fontsize": 11},
           wedgeprops={"edgecolor": IVORY, "linewidth": 1.5})
    ax.set_title(f"{column} Distribution", fontsize=13, fontweight="bold", color=OCEAN_BLUE)
    fig.patch.set_facecolor(IVORY)
    return "IMAGE:" + _fig_to_b64(fig)


# ── Build LangChain ReAct Agent ────────────────────────────────────────────────

TOOLS = [get_basic_stats, query_dataframe, plot_histogram,
         plot_bar, plot_survival_by_group, plot_pie]

SYSTEM_PROMPT = """You are a helpful and friendly data analyst specializing in the Titanic dataset.
You have access to the following tools to answer questions:

{tools}

Tool names: {tool_names}

Dataset columns: PassengerId, Survived (0/1), Pclass (1/2/3), Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked (C/Q/S)
Total rows: 418 passengers

Use the ReAct format:
Thought: what you need to do
Action: tool_name
Action Input: the input to the tool
Observation: the result
... (repeat as needed)
Thought: I now have enough information
Final Answer: your friendly, clear answer

When a tool returns a string starting with 'IMAGE:', that is a visualization — mention to the user that a chart has been generated.
Always be conversational, accurate, and concise.

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(SYSTEM_PROMPT)


def build_agent(api_key: str) -> AgentExecutor:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=api_key,
        temperature=0,
        max_tokens=1500,
    )
    agent = create_react_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=False,
                         handle_parsing_errors=True, max_iterations=6)


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="Titanic Chat Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    question: str
    api_key: str


class ChatResponse(BaseModel):
    answer: str
    image_b64: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    agent = build_agent(req.api_key)
    result = agent.invoke({"input": req.question})
    output = result.get("output", "")

    # Check intermediate steps for any image
    image_b64 = None
    for step in result.get("intermediate_steps", []):
        obs = step[1] if isinstance(step, (list, tuple)) and len(step) > 1 else ""
        if isinstance(obs, str) and obs.startswith("IMAGE:"):
            image_b64 = obs[6:]
            break

    # Clean IMAGE: token from final answer text
    clean_answer = re.sub(r"IMAGE:[A-Za-z0-9+/=]+", "[chart generated]", output)
    return ChatResponse(answer=clean_answer, image_b64=image_b64)


@app.get("/health")
def health():
    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
