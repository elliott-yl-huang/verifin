# verifin: verifiable finance agents

[![ci](https://github.com/elliott-yl-huang/verifin/actions/workflows/ci.yml/badge.svg)](https://github.com/elliott-yl-huang/verifin/actions/workflows/ci.yml)

**An LLM finance agent that cannot state a number it cannot prove.** Every figure in an answer has to trace back,
through a lineage graph, to a SQL query or a deterministic model. Anything else gets sent back to the model to fix,
and if the model can't fix it, the number is redacted.

```
$ verifin ask "Why did Data Center revenue change in 2025-Q4 vs 2025-Q3?" --audit audit.json
Data Center revenue grew from $3.99B [F1] in 2025-Q3 to $4.69B [F2] in 2025-Q4, a change of $695.53M [F3] (+17.4% [F4]).
Bridge: price effect $13.80M [F5], volume effect $613.95M [F6], mix effect $67.77M [F7].
Largest product driver: Accelerator-H at $573.12M [F8].

[offline | skill=revenue-variance | claims=8 | grounded=100% | repairs=0 | VERIFIED]

$ verifin verify audit.json --text "Data Center revenue grew to \$4.80B [F2], up 17.4% [F4]."
 citation_mismatch  $4.80B [F2]
          grounded  17.4% [F4]
```

## Why this exists

Finance teams are adopting agentic AI quickly: NVIDIA's NeMo Agent Toolkit, NIM microservices and AI-Q blueprint, and
its published work on [agentic AI in financial services](https://blogs.nvidia.com/blog/financial-services-agentic-ai)
and [multi-agent financial signal discovery](https://developer.nvidia.com/blog/automating-and-optimizing-financial-signal-discovery-with-multi-agent-systems/), are examples.
In finance, though, a fluent answer with one wrong figure is worse than no answer. Typical agent stacks focus on
*orchestration* (tool calling, routing, retrieval). They rarely give you a hard guarantee that the numbers in the
prose match the data. verifin fills that gap:

| Gap in typical finance agents | What verifin does |
|---|---|
| Hallucinated or mis-transcribed numbers | **Claim verifier** parses every number in the prose (`$1.23B`, `-4.5%`, `350 bps`, `1,234`) and accepts it only if a ledger fact *rounds to it at the precision displayed* |
| Wrong citations ("$4.80B [F2]" when F2 is $4.69B) | Per-claim citation check: `grounded`, `uncited`, `citation_mismatch`, `ungrounded` |
| The model doing arithmetic in its head | `calc` tool: a safe AST evaluator over fact ids that records lineage (`F12 = (F3-F2)/F2*100`) |
| Fail-open behavior | **verify → repair → redact** loop: the verifier's feedback goes back to the LLM; numbers still unverified after N repairs become `[UNVERIFIED]` |
| No audit trail | `--audit` writes a self-contained pack (answer, claims, fact lineage, SQL, tool trace) that `verifin verify` can re-check offline |
| Agents that can write to the warehouse | SQL guard enforced by the **SQLite authorizer** (not regex), which also hides eval ground-truth tables |
| Point forecasts with no uncertainty | Split-**conformal** intervals calibrated per horizon on rolling-origin backtests, plus a seasonal-naive benchmark |
| "Looks good" demos | An **eval harness with CI gates**: SQL-oracle accuracy, grounding rate, red-team hallucination detection, anomaly precision/recall, out-of-sample interval coverage |

## Architecture

```
question ─► skill router ─► planner ──────────────► tools ───────────────► Fact Ledger (value, unit, source SQL, inputs)
            (skills/*.md)    • LLM (NIM / any        • run_sql (guarded)                 │
                               OpenAI-compatible)    • revenue_variance (PVM)            ▼
                             • offline playbooks     • budget_variance       narrative with [F#] citations
                                                     • forecast (conformal)              │
                                                     • journal_anomalies                 ▼
                                                     • opex_breakdown        claim verifier ──fail──► repair (≤N) ──► redact
                                                     • calc (lineage)                    │ pass
                                                                                         ▼
                                                                         answer + audit pack (JSON)
```

| Module | What it teaches |
|---|---|
| [`provenance.py`](verifin/provenance.py) | Fact ledger, lineage DAG, rounding-aware claim verification, safe expression eval |
| [`tools.py`](verifin/tools.py) | Price/volume/mix variance (exact decomposition), budget vs actual, ridge time-series model with CV-selected λ, conformal prediction, Benford's law (Nigrini MAD), robust z-scores, duplicate/threshold-splitting/off-hours controls |
| [`agent.py`](verifin/agent.py) | Tool-calling loop, tool schemas, self-repair from verifier feedback, deterministic fallback planner |
| [`skills/`](verifin/skills) | Skill files (frontmatter + playbook), loaded progressively via `load_skill` |
| [`llm.py`](verifin/llm.py) | Zero-dependency OpenAI-compatible client (NVIDIA NIM by default) |
| [`evals.py`](verifin/evals.py) | Golden set with SQL oracles, adversarial perturbation, detector P/R, coverage |
| [`data.py`](verifin/data.py) | Seeded synthetic warehouse (fictional segments/products) with labelled audit anomalies |

## Quickstart

```bash
pip install -e ".[dev]"
verifin ask --offline "How did 2025 revenue compare to budget?"
verifin ask --offline "Forecast Data Center revenue for the next 3 months"
verifin ask --offline "Audit the journal entries for suspicious activity"
verifin eval            # exits non-zero if any quality gate fails
pytest -q

# With an LLM (any OpenAI-compatible endpoint; defaults to NVIDIA's hosted NIM API)
export NVIDIA_API_KEY=nvapi-...                     # or VERIFIN_API_KEY
export VERIFIN_MODEL=meta/llama-3.1-70b-instruct    # optional
export VERIFIN_BASE_URL=https://integrate.api.nvidia.com/v1   # optional (vLLM/Ollama also work)
verifin ask "Which product had the highest revenue in 2025 and how much?"

docker build -t verifin . && docker run --rm verifin eval
```

## Current eval results (offline planner)

| Metric | Value | Gate |
|---|---|---|
| Skill routing accuracy | 1.00 | ≥ 0.90 |
| Answer accuracy vs SQL oracle | 1.00 | ≥ 0.90 |
| Grounding rate (claims traced to facts) | 1.00 | = 1.00 |
| Red-team detection (±7% perturbed cited numbers + fabricated uncited figures) | 1.00 | = 1.00 |
| JE anomaly recall / precision | 0.95 / 0.75 | recall ≥ 0.90 |
| Out-of-sample 90% interval coverage (60 forecasts, 4 origins) | 0.87 | ≥ 0.80 |

## Design notes

- **Rounding-aware matching.** `$1.23B` claims the value lies in [1.225B, 1.235B]. Tolerance comes from the digits
  the writer actually displayed, so `$1,234.6M` and `$1.23B` can both cite the same fact while `$1.25B` is rejected.
- **Units are typed.** A `%` claim can match only a `pct` fact and a `$` claim only a `usd` fact. `bps` is converted to percent.
- **Not claims:** years, `YYYY-MM` dates, `Q3`, ids like `JE00120`, `[F12]` citations and small bare integers ("3 months").
- **Sign-agnostic by design.** Prose says "declined by $5M" rather than "-$5M", so magnitudes are compared and direction
  words are left to the narrative layer. Checking direction is on the roadmap.
- **The PVM bridge is exact.** price = Σ(ASP₁−ASP₀)·U₁, volume = ASP̄₀·(U₁−U₀), mix = Σ ASP₀·U₁·(s₁−s₀), and a test
  asserts they sum to ΔRevenue.

## Roadmap

- Direction/comparative checks ("grew", "beat plan") against fact signs
- NeMo Agent Toolkit workflow wrapper and MCP server exposing the verifier as a tool
- cuDF backend for large GL tables; a learned anomaly scorer (isolation forest) blended with the rule scores
- LLM-mode eval runs (accuracy and repair-rate per model) published from CI

MIT licensed. All data is synthetic and fictional.
