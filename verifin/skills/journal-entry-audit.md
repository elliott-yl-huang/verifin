---
name: journal-entry-audit
description: Screen journal entries for audit risk (duplicates, threshold-splitting, off-hours, outliers, Benford).
triggers: [audit, anomaly, anomalies, fraud, suspicious, journal, duplicate, controls, benford, risky]
tools: [journal_anomalies]
---
# Journal-entry audit screen
1. Call `journal_anomalies(period?, top=5)`.
2. Report entries reviewed, entries flagged and the Benford MAD with its conformity verdict.
3. List the top-risk entries with id, amount and reasons; recommend follow-up (supporting docs,
   approver review). These are leads for a human reviewer, not conclusions.
