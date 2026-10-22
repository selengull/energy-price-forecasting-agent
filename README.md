# ⚡ Energy Price Forecasting Agent

Forecasting hourly day-ahead electricity prices in the Spanish market with machine learning, plus an LLM-powered analyst assistant that answers natural-language questions about the market.

> 🚧 Work in progress

## Project Goals

1. **Forecast** hourly electricity prices 24 hours ahead and beat strong baselines.
2. **Explain** what drives prices (wind, solar, gas, demand, weather) using SHAP.
3. **Ask** questions in plain English through an LLM agent that queries the data, runs the model, and plots results.

## Dataset

[Hourly energy demand, generation and weather — Spain (2015–2018)](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) from Kaggle.

Download the two CSV files and place them in `data/raw/`:

```
data/raw/energy_dataset.csv
data/raw/weather_features.csv
```

## Exploratory Analysis Highlights

Full analysis: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

![Price over time](reports/figures/price_over_time.png)

- **Strong seasonality:** two daily peaks on weekdays (9–12h and 19–21h), cheaper weekends, cheapest in spring.
- **Generation mix matters:** coal, gas and demand push prices up; wind pushes them down.
- **The official forecast is biased:** it has an MAE of **10.5 €/MWh** and underestimates the actual price by ≈ 8 € on average.

![Seasonality](reports/figures/seasonality.png)

## Forecasting Results

Full analysis: [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb)

**Task:** predict the price 24 hours ahead, using only information available at forecast time. Trained on 2015–2017, tested on the unseen year **2018**.

| Model | MAE (€/MWh) | RMSE (€/MWh) | vs. naive |
|---|---:|---:|---:|
| **LightGBM** | **3.80** | **5.40** | **+27%** |
| Ridge regression | 4.04 | 5.75 | +22% |
| LightGBM (without official forecast) | 4.16 | 5.93 | +20% |
| Naive (same hour yesterday) | 5.21 | 7.70 | baseline |
| Seasonal naive (same hour last week) | 6.39 | 9.64 | −23% |
| Official day-ahead forecast | 8.86 | 11.89 | −70% |

> **Why the naive baseline, not the official forecast?** The official figure is the day-ahead *market* price, which is systematically ≈ 6 € below the final price. A simple "same as yesterday" forecast already beats it, so it is the honest benchmark.

![Forecast week](reports/figures/forecast_week.png)

## Explainability (SHAP)

Full analysis: [`notebooks/03_explainability.ipynb`](notebooks/03_explainability.ipynb)

Every forecast is broken down into per-feature contributions, so we can answer *"why does the model expect this price?"*.

![SHAP beeswarm](reports/figures/shap_beeswarm.png)

- The official day-ahead price and recent prices are the main drivers.
- Weekends lower the forecast; more forecast wind lowers it, higher residual load raises it.
- **Failure case:** on Easter Sunday night 2018 the model followed an extremely low market signal too closely (predicted 13.9 €, actual 32.8 €).

## Statistical Analysis in R

Full analysis: [`r_analysis/`](r_analysis/README.md)

A regression with Newey-West standard errors (to account for autocorrelation) shows that **each extra GW of forecast wind lowers the price by ≈ 1 €/MWh** (95% CI −1.15 to −0.87), holding demand, solar and calendar effects constant. This is the *merit order effect* of renewables.

![Wind effect](reports/figures/r_wind_effect.png)

## LLM Analyst Agent

Code: [`src/agent/`](src/agent/)

An agent that answers natural-language questions about the market by deciding which tool to call:

| Tool | What it does |
|---|---|
| `query_database` | Runs a read-only SQL query over historical data (2015–2018) and the model's 2018 forecasts |
| `forecast_price` | Returns the model's forecast for a specific hour, with the actual and official prices |
| `explain_forecast` | Explains a forecast with its top SHAP drivers |

The agent runs a tool-calling loop: the LLM requests a tool, Python executes it, the result is sent back, and this repeats until the LLM writes a final answer.

```
You: What was the model's biggest error in 2018 and when did it happen?
  [tool] query_database(SELECT time, abs_error FROM forecasts WHERE year = 2018 ...)   -> error: no such column
  [tool] query_database(SELECT time, abs_error FROM forecasts ORDER BY abs_error DESC LIMIT 1)

Agent: The model's biggest error in 2018 was €34.66/MWh, at 19:00 on March 11, 2018.
```

**Design choices**
- **Runs locally for free** with [LM Studio](https://lmstudio.ai) and `qwen3-4b-instruct-2507` (fits on a 4 GB GPU). Any OpenAI-compatible endpoint works by changing `.env`.
- **Grounded answers:** the system prompt requires every number to come from a tool result.
- **Safe SQL:** only single `SELECT` statements, executed on a read-only connection.
- **Self-correction:** tool errors are returned to the LLM so it can fix its own query (see example above).
- **Bounded:** at most 6 tool-calling steps per question.

## Project Structure

```
├── data/            # raw and processed data (not tracked by git)
├── notebooks/       # exploratory analysis
├── src/
│   ├── data/        # loading and cleaning
│   ├── features/    # feature engineering
│   ├── models/      # training, evaluation, prediction
│   └── agent/       # LLM agent and tools
├── app/             # Streamlit interface
├── r_analysis/      # statistical analysis in R
├── reports/figures/ # generated plots
└── tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

python -m src.data.load          # clean the raw data
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb   # train and save the model
python -m src.agent.database     # build the SQLite database for the agent
```

To run the agent, start the LM Studio server with `qwen3-4b-instruct-2507` loaded, copy `.env.example` to `.env`, then:

```bash
python -m src.agent.agent
```

## Roadmap

- [x] Data loading, cleaning and EDA
- [x] Baseline and gradient boosting forecasting models
- [x] SHAP explainability and R statistical analysis
- [x] LLM analyst agent with tools
- [ ] Streamlit app
- [ ] Final results and demo
