# ⚡ Energy Price Forecasting Agent

Forecasting hourly day-ahead electricity prices in the Spanish market with machine learning, plus an LLM-powered analyst assistant that answers natural-language questions about the market.

> 🚧 Work in progress

## Project Goals

1. **Forecast** hourly electricity prices and beat the official TSO forecast included in the dataset.
2. **Explain** what drives prices (wind, solar, gas, demand, weather) using SHAP.
3. **Ask** questions in plain English through an LLM agent that queries the data, runs the model, and plots results.

## Dataset

[Hourly energy demand, generation and weather — Spain (2015–2018)](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) from Kaggle.

Download the two CSV files and place them in `data/raw/`:

```
data/raw/energy_dataset.csv
data/raw/weather_features.csv
```

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
```

## Roadmap

- [ ] Data loading, cleaning and EDA
- [ ] Baseline and gradient boosting forecasting models
- [ ] SHAP explainability and R statistical analysis
- [ ] LLM analyst agent with tools
- [ ] Streamlit app
- [ ] Final results and demo
