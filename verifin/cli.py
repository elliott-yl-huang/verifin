"""Command line: verifin ask | eval | verify | build-db"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from .agent import Agent
from .data import build
from .llm import ChatClient
from .provenance import Fact, Ledger, verify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verifin", description="Verifiable finance agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--db"); a.add_argument("--audit")
    a.add_argument("--offline", action="store_true", help="ignore API keys; use deterministic playbooks")
    e = sub.add_parser("eval"); e.add_argument("--json", action="store_true")
    v = sub.add_parser("verify", help="re-verify an answer against an audit pack")
    v.add_argument("audit"); v.add_argument("--text", help="text to check (default: the pack's answer)")
    b = sub.add_parser("build-db"); b.add_argument("path")
    args = ap.parse_args(argv)

    if args.cmd == "build-db":
        build(args.path); print(f"wrote {args.path}"); return 0
    if args.cmd == "ask":
        con = sqlite3.connect(args.db) if args.db else build()
        ans = Agent(con, None if args.offline else ChatClient.from_env()).ask(args.question)
        print(ans.text)
        v = ans.verification
        print(f"\n[{ans.mode} | skill={ans.skill} | claims={len(v.claims)} | grounded={v.grounding_rate:.0%} | "
              f"repairs={ans.repairs} | {'VERIFIED' if v.ok else 'REDACTED'}]", file=sys.stderr)
        if args.audit:
            json.dump(ans.audit_pack(), open(args.audit, "w"), indent=2, default=str)
        return 0 if v.ok else 2
    if args.cmd == "eval":
        from .evals import run
        r = run()
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            for k, val in r["metrics"].items():
                print(f"{k:>20}: {val:.3f}" if isinstance(val, float) else f"{k:>20}: {val}")
        return 0 if r["metrics"]["passed"] else 1
    if args.cmd == "verify":
        pack = json.load(open(args.audit))
        L = Ledger()
        for k, f in pack["facts"].items():
            L.facts[k] = Fact(**{**f, "inputs": tuple(f["inputs"])})
        res = verify(args.text or pack["answer"], L)
        for c in res.claims:
            print(f"{c.status:>18}  {c.text}")
        return 0 if res.ok else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
