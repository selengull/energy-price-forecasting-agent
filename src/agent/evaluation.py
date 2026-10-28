"""Evaluate the analyst agent on questions with known answers.

For every test case we check two things:
1. Tool choice: did the agent call the right tool?
2. Answer:      does the final answer contain the correct number (or keyword)?

Expected answers are computed from the database and the model at run time,
so the answer key can never drift out of sync with the data.

Run from the project root (LM Studio server must be running):
    python -m src.agent.evaluation
    python -m src.agent.evaluation --model qwen3.5-4b     # compare another model
"""

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from src.agent.agent import EnergyAnalystAgent
from src.agent.database import run_query
from src.agent.tools import service

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def sql_value(sql: str) -> Callable[[], float]:
    return lambda: float(run_query(sql).iloc[0, 0])


@dataclass
class Case:
    category: str
    question: str
    expected_tools: list[str]
    expected_number: Callable[[], float] | None = None  # computed lazily from data / model
    expected_keywords: list[str] = field(default_factory=list)  # pass if any appears (case-insensitive)


CASES = [
    # ---- Historical data (SQL) ----
    Case("sql", "What was the average electricity price in 2016?", ["query_database"],
         sql_value("SELECT AVG(price) FROM hourly WHERE year = 2016")),
    Case("sql", "What was the highest hourly price in the whole dataset?", ["query_database"],
         sql_value("SELECT MAX(price) FROM hourly")),
    Case("sql", "What was the lowest hourly price in 2018?", ["query_database"],
         sql_value("SELECT MIN(price) FROM hourly WHERE year = 2018")),
    Case("sql", "What was the average price on weekends in 2017?", ["query_database"],
         sql_value("SELECT AVG(price) FROM hourly WHERE year = 2017 AND is_weekend = 1")),
    Case("sql", "What was the average electricity demand in July 2018, in MW?", ["query_database"],
         sql_value("SELECT AVG(demand_mw) FROM hourly WHERE year = 2018 AND month = 7")),
    Case("sql", "What was the average wind generation in 2015, in MW?", ["query_database"],
         sql_value("SELECT AVG(wind_mw) FROM hourly WHERE year = 2015")),
    Case("sql", "How many hours in 2018 had a price above 80 EUR/MWh?", ["query_database"],
         sql_value("SELECT COUNT(*) FROM hourly WHERE year = 2018 AND price > 80")),
    Case("sql", "What was the average temperature in August 2018?", ["query_database"],
         sql_value("SELECT AVG(temperature_c) FROM hourly WHERE year = 2018 AND month = 8")),
    Case("sql", "Which month of 2017 had the highest average price?", ["query_database"],
         expected_keywords=["january"]),
    Case("sql", "Which year had the lowest average price?", ["query_database"],
         expected_keywords=["2016"]),
    Case("sql", "Which hour of the day is the most expensive on weekdays?", ["query_database"],
         expected_keywords=["20:00", "8 pm", "8:00 pm", "hour 20"]),

    # ---- Model performance (SQL on forecasts table) ----
    Case("performance", "What was the model's mean absolute error in 2018?", ["query_database"],
         sql_value("SELECT AVG(abs_error) FROM forecasts")),
    Case("performance", "What was the mean absolute error of the official day-ahead price in 2018?", ["query_database"],
         sql_value("SELECT AVG(ABS(actual - price_day_ahead)) FROM forecasts")),
    Case("performance", "In how many hours of 2018 was the model's error larger than 10 EUR/MWh?", ["query_database"],
         sql_value("SELECT COUNT(*) FROM forecasts WHERE abs_error > 10")),
    Case("performance", "What was the model's average error in March 2018?", ["query_database"],
         sql_value("SELECT AVG(abs_error) FROM forecasts WHERE time LIKE '2018-03%'")),

    # ---- Single forecasts ----
    Case("forecast", "What did the model forecast for 15 June 2018 at 12:00?", ["forecast_price", "query_database"],
         lambda: service.forecast("2018-06-15 12:00")["forecast_eur_mwh"]),
    Case("forecast", "What was the actual price on 10 February 2018 at 09:00?", ["forecast_price", "query_database"],
         lambda: service.forecast("2018-02-10 09:00")["actual_eur_mwh"]),

    # ---- Explanations ----
    Case("explain", "What was the biggest driver of the model's forecast for 18 September 2018 at 20:00?",
         ["explain_forecast"], expected_keywords=["24 hours", "24h", "yesterday", "previous day", "day before"]),
    Case("explain", "What was the biggest driver of the model's forecast for 1 April 2018 at 04:00?",
         ["explain_forecast"], expected_keywords=["day-ahead", "day ahead", "official"]),

    # ---- Questions in Turkish (dates must be converted by the agent) ----
    Case("turkish", "13 Kasım 2018 saat 12'de elektrik fiyatı ne kadardı?", ["forecast_price", "query_database"],
         sql_value("SELECT price FROM hourly WHERE time = '2018-11-13 12:00:00'")),
    Case("turkish", "2017 yılında ortalama elektrik fiyatı neydi?", ["query_database"],
         sql_value("SELECT AVG(price) FROM hourly WHERE year = 2017")),
    Case("turkish", "2018'de modelin ortalama hatası ne kadardı?", ["query_database"],
         sql_value("SELECT AVG(abs_error) FROM forecasts")),

    # ---- What-if scenarios ----
    Case("scenario", "How would the price on 6 November 2018 at 08:00 change if wind generation were 5 GW higher?",
         ["run_scenario"], lambda: service.scenario("2018-11-06 08:00", wind_change_mw=5000)["price_change_eur_mwh"]),
    Case("scenario", "What if demand were 3 GW higher on 6 November 2018 at 15:00? How much would the price change?",
         ["run_scenario"], lambda: service.scenario("2018-11-06 15:00", demand_change_mw=3000)["price_change_eur_mwh"]),
]


def numbers_in(text: str) -> list[float]:
    """All numbers in a text, read both in English ('1,234.5') and Turkish/European ('1.234,5') format."""
    numbers = []
    for token in re.findall(r"-?\d[\d.,]*\d|-?\d", text):
        english = token.replace(",", "")
        european = token.replace(".", "").replace(",", ".")
        for candidate in (english, european):
            try:
                numbers.append(float(candidate))
            except ValueError:
                pass
    return numbers


def answer_is_correct(answer: str, expected: float | None, keywords: list[str]) -> bool:
    if expected is not None:
        if float(expected).is_integer():
            tolerance = 0.5  # counts must match exactly
        else:
            tolerance = max(abs(expected) * 0.005, 0.05)  # 0.5 % or 0.05, whichever is larger
        return any(abs(abs(n) - abs(expected)) <= tolerance for n in numbers_in(answer))
    return any(k.lower() in answer.lower() for k in keywords)


def run(model: str | None = None, repeats: int = 1) -> pd.DataFrame:
    """Ask every question `repeats` times: LLM answers vary between runs, so one run is not enough."""
    rows = []
    for i, case in enumerate(CASES, start=1):
        expected = case.expected_number() if case.expected_number else None
        for attempt in range(1, repeats + 1):
            agent = EnergyAnalystAgent(model=model)

            start = time.perf_counter()
            try:
                reply = agent.ask(case.question)
                answer, tools_used = reply.answer, [c.name for c in reply.tool_calls]
            except Exception as exc:  # noqa: BLE001 - a crash counts as a failed case
                answer, tools_used = f"ERROR: {exc}", []
            seconds = time.perf_counter() - start

            tool_ok = any(t in case.expected_tools for t in tools_used)
            answer_ok = answer_is_correct(answer, expected, case.expected_keywords)
            mark = "PASS" if answer_ok else "FAIL"
            run_label = f" run {attempt}/{repeats}" if repeats > 1 else ""
            print(f"[{i:2d}/{len(CASES)}{run_label}] {mark}  tool {'ok ' if tool_ok else 'BAD'}  {seconds:5.1f}s  {case.question}")
            if not answer_ok:
                target = f"{expected:.2f}" if expected is not None else " / ".join(case.expected_keywords)
                print(f"         expected: {target}")
                print(f"         answer:   {answer[:200]!r}")

            rows.append({
                "category": case.category, "question": case.question, "attempt": attempt,
                "expected": expected if expected is not None else "|".join(case.expected_keywords),
                "answer": answer, "tools_used": ",".join(tools_used),
                "tool_correct": tool_ok, "answer_correct": answer_ok, "seconds": round(seconds, 1),
            })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, model: str) -> dict:
    by_category = results.groupby("category")["answer_correct"].mean().round(3).to_dict()
    per_question = results.groupby("question", sort=False)["answer_correct"].mean()
    return {
        "model": model,
        "questions": int(results["question"].nunique()),
        "runs_per_question": int(results["attempt"].max()),
        "always_correct": int((per_question == 1).sum()),
        "sometimes_wrong": per_question[per_question < 1].round(2).to_dict(),
        "answer_accuracy": round(results["answer_correct"].mean(), 3),
        "tool_accuracy": round(results["tool_correct"].mean(), 3),
        "avg_seconds": round(results["seconds"].mean(), 1),
        "answer_accuracy_by_category": by_category,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="LLM model name (defaults to LLM_MODEL in .env)")
    parser.add_argument("--repeats", type=int, default=3, help="how many times to ask each question (default 3)")
    args = parser.parse_args()

    model_name = args.model or EnergyAnalystAgent().model
    print(f"Evaluating {model_name} on {len(CASES)} questions x {args.repeats} runs...\n")
    results = run(args.model, args.repeats)
    summary = summarize(results, model_name)

    REPORTS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    results.to_csv(REPORTS_DIR / f"eval_results_{slug}.csv", index=False)
    (REPORTS_DIR / f"eval_summary_{slug}.json").write_text(json.dumps(summary, indent=2))

    print(f"\nAnswer accuracy: {summary['answer_accuracy']:.0%}   Tool accuracy: {summary['tool_accuracy']:.0%}   "
          f"Avg time: {summary['avg_seconds']}s")
    print("By category:", {k: f"{v:.0%}" for k, v in summary["answer_accuracy_by_category"].items()})
    print(f"Always correct: {summary['always_correct']}/{summary['questions']} questions")
    for question, rate in summary["sometimes_wrong"].items():
        print(f"  correct {rate:.0%} of the time: {question}")
