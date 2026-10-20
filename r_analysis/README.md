# Statistical Analysis in R: Does Wind Lower Prices?

Script: [`wind_price_effect.R`](wind_price_effect.R)

## Question

The EDA showed that windy hours tend to be cheaper. But windy hours also differ in other ways (season, time of day, demand).
**Is there a real effect of wind on the price once those factors are held constant, and how large is it?**

## Method

Linear regression on 35,064 hourly observations (2015–2018):

```
price ~ wind_forecast + solar_forecast + load_forecast + hour + weekday + month + year
```

- Forecast values (not actuals) are used, since they are what the market knows in advance.
- Hourly prices are strongly **autocorrelated**: one hour looks a lot like the next. Ordinary standard errors would
  therefore overstate our confidence, so we use **Newey-West (HAC) standard errors** with a 24-hour lag.

## Results

| Model | Wind effect (€/MWh per GW) | 95% CI | p-value | R² |
|---|---:|---:|---:|---:|
| Wind only | −0.99 | −1.18 to −0.81 | < 0.001 | 0.05 |
| With controls | **−1.01** | **−1.15 to −0.87** | < 0.001 | 0.62 |

![Wind effect](../reports/figures/r_wind_effect.png)

## Interpretation

- **Each additional GW of forecast wind lowers the price by about 1 €/MWh**, even after controlling for demand, solar and calendar effects.
  Spanish wind ranges from ~0 to 17 GW, so wind alone can move prices by up to ~17 €/MWh.
- **Solar has a similar effect** (−1.24 € per GW), and **demand raises prices** (+0.70 € per GW). This is the "merit order effect":
  zero-marginal-cost renewables push expensive gas and coal plants out of the market.
- The Newey-West standard error is **4.5× larger** than the ordinary one. Ignoring autocorrelation would make the result
  look far more precise than it really is. The effect is still clearly significant.

## Run it

```bash
Rscript r_analysis/wind_price_effect.R
```

Requires the R packages `ggplot2`, `sandwich` and `lmtest`.
