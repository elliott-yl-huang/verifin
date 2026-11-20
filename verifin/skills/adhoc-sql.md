---
name: adhoc-sql
description: Answer any other quantitative question with read-only SQL over the warehouse.
triggers: [how many, list, which, top, sql, query, table]
tools: [run_sql, calc]
---
# Ad-hoc SQL
Schema: sales(period 'YYYY-MM', segment, product, units, asp, revenue); budget(period, segment, revenue_plan);
opex(period, cost_center, account, amount); journal_entries(je_id, period, posted_at, account, vendor, amount, posted_by, description).
1. Write one SELECT. Name numeric columns meaningfully (`revenue`, `amount`, `n`).
2. Each numeric cell becomes a fact; cite them. Writes are blocked by the database authorizer.
