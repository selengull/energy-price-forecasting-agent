"""Streamlit interface: chat with the analyst agent and explore the 2018 forecasts.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import io
import json
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent.agent import EnergyAnalystAgent
from src.agent.database import run_query
from src.agent.tools import service
from src.plot_style import PALETTE

st.set_page_config(page_title="Energy Price Analyst", page_icon="⚡", layout="wide")

TEXT_SECONDARY = "#52514e"
EXAMPLE_QUESTIONS = [
    "What was the average price in each year?",
    "Which hour of the day is the most expensive on weekdays?",
    "What was the model's biggest error in 2018 and when did it happen?",
    "Why did the model forecast that price for 18 September 2018 at 20:00?",
]


# ---- Helpers -------------------------------------------------------------------

def llm_is_available(agent: EnergyAnalystAgent) -> bool:
    try:
        agent.client.models.list()
        return True
    except Exception:  # noqa: BLE001 - any connection problem means "not available"
        return False


def show_tool_result(name: str, result: str) -> None:
    """Render a tool result as a table (SQL) or JSON (forecast / explanation)."""
    if result.startswith("Error"):
        st.error(result)
    elif name == "query_database":
        st.dataframe(pd.read_csv(io.StringIO(result.split("\n(showing")[0])), hide_index=True)
    else:
        st.json(json.loads(result))


def show_tool_calls(tool_calls) -> None:
    for call in tool_calls:
        with st.expander(f"🔧 {call.name}", expanded=False):
            if call.name == "query_database":
                st.code(call.arguments.get("sql", ""), language="sql")
            else:
                st.write(call.arguments)
            show_tool_result(call.name, call.result)


def style_figure(fig: go.Figure, title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16)),
        yaxis_title=y_title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


# ---- Page --------------------------------------------------------------------------

st.title("⚡ Spanish Electricity Price Analyst")
st.caption("A LightGBM model forecasts prices 24 hours ahead; an LLM agent answers questions using SQL, the model and SHAP.")

if "agent" not in st.session_state:
    st.session_state.agent = EnergyAnalystAgent()
    st.session_state.history = []  # list of (role, text, tool_calls)
agent: EnergyAnalystAgent = st.session_state.agent

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Data:** Spain, hourly, 2015–2018\n"
        "- **Model:** LightGBM, test MAE **3.80 €/MWh** on 2018 (naive baseline: 5.21)\n"
        f"- **LLM:** `{agent.model}`"
    )
    if llm_is_available(agent):
        st.success("LLM server connected", icon="✅")
    else:
        st.error("LLM server not reachable. Start the LM Studio server and load the model.", icon="⚠️")
    if st.button("🗑️ New chat", width="stretch"):
        agent.reset()
        st.session_state.history = []
        st.rerun()

chat_tab, explorer_tab = st.tabs(["💬 Ask the analyst", "📈 Forecast explorer"])

# ---- Tab 1: chat -------------------------------------------------------------------

with chat_tab:
    if not st.session_state.history:
        st.markdown("**Try one of these:**")
        columns = st.columns(2)
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            if columns[i % 2].button(example, width="stretch"):
                st.session_state.pending_question = example
                st.rerun()

    for role, text, tool_calls in st.session_state.history:
        with st.chat_message(role):
            if tool_calls:
                show_tool_calls(tool_calls)
            st.markdown(text)

    question = st.chat_input("Ask about prices, forecasts or what drives them…")
    question = question or st.session_state.pop("pending_question", None)

    if question:
        st.session_state.history.append(("user", question, []))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking and calling tools…"):
                try:
                    reply = agent.ask(question)
                    answer, calls = reply.answer, reply.tool_calls
                except Exception as exc:  # noqa: BLE001
                    answer, calls = f"⚠️ Could not reach the LLM: {exc}", []
            show_tool_calls(calls)
            st.markdown(answer)
        st.session_state.history.append(("assistant", answer, calls))
        st.rerun()  # redraw so the example buttons disappear once the chat has started

# ---- Tab 2: forecast explorer --------------------------------------------------------

with explorer_tab:
    left, right = st.columns([1, 3])
    with left:
        day = st.date_input("Day in 2018", value=date(2018, 11, 6), min_value=date(2018, 1, 1), max_value=date(2018, 12, 31))
    day_data = run_query(f"SELECT * FROM forecasts WHERE time LIKE '{day:%Y-%m-%d}%' ORDER BY time")
    day_data["hour"] = pd.to_datetime(day_data["time"]).dt.hour

    with left:
        mae_model = (day_data["forecast"] - day_data["actual"]).abs().mean()
        mae_official = (day_data["price_day_ahead"] - day_data["actual"]).abs().mean()
        st.metric("Model error (MAE)", f"{mae_model:.2f} €", delta=f"{mae_model - mae_official:.2f} € vs official", delta_color="inverse")
        st.metric("Average actual price", f"{day_data['actual'].mean():.2f} €")

    with right:
        fig = go.Figure()
        fig.add_scatter(x=day_data["hour"], y=day_data["actual"], name="Actual price", line=dict(color=PALETTE[0], width=3))
        fig.add_scatter(x=day_data["hour"], y=day_data["forecast"], name="Model forecast", line=dict(color=PALETTE[1], width=3, dash="dash"))
        fig.add_scatter(x=day_data["hour"], y=day_data["price_day_ahead"], name="Official day-ahead", line=dict(color=TEXT_SECONDARY, width=1.5))
        fig.update_xaxes(title="Hour of day", dtick=3)
        st.plotly_chart(style_figure(fig, f"Prices on {day:%A, %d %B %Y}", "€/MWh"), width="stretch")

    st.subheader("Why did the model forecast this price?")
    hour = st.slider("Hour", 0, 23, 20)
    try:
        explanation = service.explain(f"{day:%Y-%m-%d} {hour:02d}:00", top_n=8)
    except Exception:  # noqa: BLE001 - e.g. the hour skipped or repeated on daylight-saving days
        st.info("No forecast for this hour (daylight-saving time change). Pick another hour.")
        st.stop()

    drivers = pd.DataFrame(explanation["top_drivers"]).iloc[::-1]
    colors = ["#e34948" if c > 0 else PALETTE[0] for c in drivers["contribution"]]
    labels = [f"{d} = {v:g}" for d, v in zip(drivers["description"], drivers["value"])]

    col_a, col_b = st.columns([3, 1])
    with col_a:
        bar = go.Figure(go.Bar(x=drivers["contribution"], y=labels, orientation="h", marker_color=colors,
                               hovertemplate="%{x:+.2f} €/MWh<extra></extra>"))
        bar.add_vline(x=0, line_color=TEXT_SECONDARY, line_width=1)
        bar.update_xaxes(title="Contribution to the forecast (€/MWh)")
        st.plotly_chart(style_figure(bar, f"Top drivers at {hour:02d}:00 (red = pushed up, blue = pushed down)", ""),
                        width="stretch")
    with col_b:
        actual = day_data.loc[day_data["hour"] == hour, "actual"]
        st.metric("Average forecast (base)", f"{explanation['base_value']:.2f} €")
        st.metric("This forecast", f"{explanation['prediction']:.2f} €",
                  delta=f"{explanation['prediction'] - explanation['base_value']:+.2f} € vs base", delta_color="off")
        if not actual.empty:
            st.metric("Actual price", f"{actual.iloc[0]:.2f} €")
