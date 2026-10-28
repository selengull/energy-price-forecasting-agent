"""The LLM analyst agent.

The agent runs a simple loop:

1. Send the conversation and the list of tools to the LLM.
2. If the LLM asks to call tools, run them and send the results back.
3. Repeat until the LLM answers in plain text (or a step limit is reached).

It talks to any OpenAI-compatible server. By default this is LM Studio running a
local model, so it costs nothing; the settings live in the `.env` file.
"""

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

from src.agent.database import SCHEMA_DESCRIPTION
from src.agent.tools import TOOL_SCHEMAS, call_tool

load_dotenv()

SYSTEM_PROMPT = f"""You are an energy market analyst assistant for the Spanish electricity market.
You answer questions using the tools below. Never invent numbers: every figure in your answer must come from a tool result.

Tools:
- query_database: SQL over historical data (2015-2018) and over the model's 2018 forecasts.
- forecast_price: the model's forecast for one specific hour in 2018.
- explain_forecast: why the model forecast that price (SHAP drivers in EUR/MWh).
- run_scenario: what-if analysis, e.g. how the price would change with more wind, more solar or higher demand.

Database schema:
{SCHEMA_DESCRIPTION}

Guidelines:
- Prices are in EUR/MWh. Times are local Madrid time.
- Users may write in any language (e.g. Turkish). Convert dates yourself before calling a tool:
  "13 Kasım 2018 saat 12" means 2018-11-13 12:00. Always answer in the user's language.
- If a question is vague (e.g. "consumption", "spending", "tüketim", "harcama"), do not refuse: pick the closest
  data (electricity demand in MW, or the price in EUR/MWh), state your assumption in one sentence, and answer.
- Never do arithmetic in your head: compute sums, products and ratios inside the SQL query.
- Prefer aggregated SQL queries (AVG, MIN, MAX, GROUP BY) over raw rows.
- When explaining a forecast, describe the top 2-3 drivers in plain language.
- If a tool returns an error, fix your input and try again.
- Keep answers short and clear: a direct answer first, then the key supporting numbers.
"""


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: str


@dataclass
class AgentReply:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class EnergyAnalystAgent:
    def __init__(self, max_steps: int = 6, model: str | None = None):
        self.client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("LLM_API_KEY", "lm-studio"),  # LM Studio ignores the key
        )
        self.model = model or os.getenv("LLM_MODEL", "qwen3-4b-instruct-2507")
        self.max_steps = max_steps
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(self, question: str) -> AgentReply:
        self.messages.append({"role": "user", "content": question})
        calls: list[ToolCall] = []

        for _ in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                temperature=0.2,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                answer = _strip_thinking(message.content or "")
                self.messages.append({"role": "assistant", "content": answer})
                return AgentReply(answer=answer, tool_calls=calls)

            # The model wants to use tools: record its request, run each tool, send results back.
            self.messages.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = call_tool(tool_call.function.name, arguments)
                calls.append(ToolCall(tool_call.function.name, arguments, result))
                self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        return AgentReply(answer="Sorry, I could not finish this answer within the step limit.", tool_calls=calls)

    def reset(self) -> None:
        self.messages = self.messages[:1]


def _strip_thinking(text: str) -> str:
    """Some local models (e.g. Qwen3) wrap their reasoning in <think> tags; hide it from the user."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


if __name__ == "__main__":
    agent = EnergyAnalystAgent()
    print("Energy analyst agent. Type 'quit' to exit.\n")
    while (question := input("You: ").strip()).lower() not in {"quit", "exit"}:
        reply = agent.ask(question)
        for call in reply.tool_calls:
            print(f"  [tool] {call.name}({call.arguments})")
        print(f"\nAgent: {reply.answer}\n")
