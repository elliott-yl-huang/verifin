---
name: revenue-variance
description: Explain why revenue changed between two periods using a price / volume / mix bridge.
triggers: [why, drove, driver, bridge, change, changed, grew, growth, declined, variance between, vs, compared, price, mix]
tools: [revenue_variance, calc]
---
# Revenue variance (price-volume-mix)
1. Resolve both periods (YYYY, YYYY-Qn or YYYY-MM). If only one is given, compare with the prior period.
2. Call `revenue_variance(period_a, period_b, segment)`.
3. Report: revenue A, revenue B, change ($ and %), then the price, volume and mix effects
   (they sum exactly to the change), then the biggest product driver.
4. Cite every number with its fact id, e.g. `$1.23B [F4]`. Never compute numbers in your head;
   use `calc` for any derived figure.
