---
name: budget-vs-actual
description: Compare actual revenue to plan/budget for a period, by segment.
triggers: [budget, plan, forecast vs, target, beat, miss, missed, actual vs, against plan]
tools: [budget_variance, calc]
---
# Budget vs actual
1. Call `budget_variance(period, segment?)`.
2. Lead with the total: actual, plan, $ variance, % variance. Then list segments ordered by
   absolute variance. Say "favorable" for positive revenue variance, "unfavorable" for negative.
3. Cite every number with its fact id.
