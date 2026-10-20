# Does wind generation lower electricity prices in Spain?
#
# We regress the hourly price on the day-ahead wind forecast while controlling
# for demand, solar, hour of day, weekday, month and year. Because hourly prices
# are strongly autocorrelated, ordinary standard errors would be far too small,
# so we use Newey-West (HAC) standard errors.
#
# Run from the project root:
#   Rscript r_analysis/wind_price_effect.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(sandwich)
  library(lmtest)
})

# ---- 1. Load data ----------------------------------------------------------
raw <- read.csv("data/raw/energy_dataset.csv")

df <- data.frame(
  time  = as.POSIXct(substr(raw$time, 1, 19), format = "%Y-%m-%d %H:%M:%S", tz = "Europe/Madrid"),
  price = raw$price.actual,
  wind  = raw$forecast.wind.onshore.day.ahead / 1000,  # MW -> GW
  solar = raw$forecast.solar.day.ahead / 1000,
  load  = raw$total.load.forecast / 1000
)
df <- na.omit(df)

df$hour    <- factor(format(df$time, "%H"))
df$weekday <- factor(weekdays(df$time))
df$month   <- factor(format(df$time, "%m"))
df$year    <- factor(format(df$time, "%Y"))

cat("Observations:", nrow(df), "\n\n")

# ---- 2. Models -------------------------------------------------------------
# Simple: price explained by wind only.
simple <- lm(price ~ wind, data = df)

# Full: same, but holding demand, solar and calendar effects constant.
full <- lm(price ~ wind + solar + load + hour + weekday + month + year, data = df)

# Newey-West standard errors (lag of 24 hours to cover the daily cycle).
nw_simple <- coeftest(simple, vcov = NeweyWest(simple, lag = 24, prewhite = FALSE))
nw_full   <- coeftest(full,   vcov = NeweyWest(full,   lag = 24, prewhite = FALSE))

report <- function(name, model, test) {
  est <- test["wind", "Estimate"]
  se  <- test["wind", "Std. Error"]
  p   <- test["wind", "Pr(>|t|)"]
  cat(sprintf("%-8s wind effect: %6.2f EUR/MWh per GW  (95%% CI %.2f to %.2f, p = %.2g, R2 = %.2f)\n",
              name, est, est - 1.96 * se, est + 1.96 * se, p, summary(model)$r.squared))
}

report("Simple", simple, nw_simple)
report("Full", full, nw_full)

naive_se <- summary(full)$coefficients["wind", "Std. Error"]
cat(sprintf("\nStandard error of the wind effect: ordinary %.3f vs Newey-West %.3f (%.1fx larger)\n",
            naive_se, nw_full["wind", "Std. Error"], nw_full["wind", "Std. Error"] / naive_se))

cat("\nOther controls in the full model (Newey-West):\n")
print(round(nw_full[c("solar", "load"), ], 3))

# ---- 3. Plot ---------------------------------------------------------------
# Partial effect: price after removing everything except wind (added-variable plot).
df$price_resid <- resid(lm(price ~ solar + load + hour + weekday + month + year, data = df))
df$wind_resid  <- resid(lm(wind  ~ solar + load + hour + weekday + month + year, data = df))

p <- ggplot(df, aes(wind_resid, price_resid)) +
  geom_bin2d(bins = 70) +
  scale_fill_gradient(low = "#cde2fb", high = "#104281", trans = "log10", name = "Hours") +
  geom_smooth(method = "lm", formula = y ~ x, colour = "#eb6834", linewidth = 1.2, se = FALSE) +
  labs(
    title = "Effect of wind on price, holding demand, solar and calendar constant",
    subtitle = sprintf("Each extra GW of forecast wind lowers the price by %.2f EUR/MWh",
                       -nw_full["wind", "Estimate"]),
    x = "Forecast wind (GW, adjusted)", y = "Price (EUR/MWh, adjusted)"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        plot.background = element_rect(fill = "#fcfcfb", colour = NA))

ggsave("reports/figures/r_wind_effect.png", p, width = 9, height = 5, dpi = 150)
cat("\nSaved reports/figures/r_wind_effect.png\n")
