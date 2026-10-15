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

- [x] Data loading, cleaning and EDA
- [x] Baseline and gradient boosting forecasting models
- [ ] SHAP explainability and R statistical analysis
- [ ] LLM analyst agent with tools
- [ ] Streamlit app
- [ ] Final results and demo
