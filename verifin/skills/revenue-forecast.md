---
name: revenue-forecast
description: Forecast monthly revenue with calibrated (conformal) prediction intervals.
triggers: [forecast, predict, projection, project, next, outlook, expect, guidance]
tools: [forecast]
---
# Revenue forecast
1. Call `forecast(segment?, horizon, alpha=0.1)`.
2. For each month report the point forecast and the interval [lo, hi], and state the nominal coverage.
3. Report backtest MAPE for the model vs the seasonal-naive baseline so the reader can judge skill.
4. Never present the point forecast without its interval. Cite every number.
