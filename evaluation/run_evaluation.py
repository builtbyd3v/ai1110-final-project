"""Evaluation harness for VibeMatch.

Runs evaluation/cases.json against the real service pipeline with a mocked
Gemini client (default, CI-safe) or the live API (--live). Writes
evaluation/results.md with a Pass/Fail matrix and metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai_service import VibeMatchService
from src.rag import load_documents_from_csv

CASES_PATH = Path(__file__).parent / "cases.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def _make_mock_client(case: Dict[str, Any]) -> MagicMock:
    client = MagicMock()
    if case.get("mock_error"):
        client.models.generate_content.side_effect = Exception("Simulated API outage")
    else:
        response = MagicMock()
        response.text = json.dumps(case["mock_response"])
        client.models.generate_content.return_value = response
    return client


def _known_titles(service: VibeMatchService) -> set:
    return {doc.title.lower() for doc in service.docs}


def evaluate_case(
    case: Dict[str, Any],
    docs,
    live: bool = False,
) -> Dict[str, Any]:
    service = VibeMatchService(docs=docs) if live else VibeMatchService(
        docs=docs, client=_make_mock_client(case)
    )
    expect = case.get("expect", {})

    # Simulate the outage even in live mode: retrieval stays real (embeddings),
    # only the generation call is forced to fail so the fallback path runs.
    if live and case.get("mock_error"):
        service.client.models.generate_content = MagicMock(
            side_effect=Exception("Simulated API outage")
        )

    start = time.perf_counter()
    result = service.recommend(
        query=case["query"],
        user_prefs=case.get("user_prefs", {}),
        k=case.get("k", 5),
        mode=case.get("mode", "balanced"),
        offline=not live,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    recs = result.recommendations
    titles = [r["title"] for r in recs]
    known = _known_titles(service)

    grounded = all(t.lower() in known for t in titles)
    cited = all(r.get("source") and r.get("evidence") for r in recs)

    checks: List[Dict[str, Any]] = []

    if "top_titles_any" in expect:
        hit = any(t in titles for t in expect["top_titles_any"])
        checks.append({
            "check": f"Top titles include one of {expect['top_titles_any']}",
            "pass": hit,
        })
    if expect.get("must_be_grounded"):
        checks.append({"check": "All recommendations grounded in catalog", "pass": grounded})
    checks.append({"check": "All recommendations cite source + evidence", "pass": cited})
    if "expect_fallback" in expect:
        checks.append({
            "check": f"Fallback used == {expect['expect_fallback']}",
            "pass": result.fallback_used == expect["expect_fallback"],
        })
    if "max_results" in expect:
        checks.append({
            "check": f"Result count <= {expect['max_results']}",
            "pass": len(recs) <= expect["max_results"],
        })
    if "forbidden_titles" in expect:
        leaked = [t for t in expect["forbidden_titles"] if t in titles]
        checks.append({
            "check": f"Forbidden titles absent {expect['forbidden_titles']}",
            "pass": not leaked,
        })

    return {
        "id": case["id"],
        "name": case["name"],
        "pass": all(c["pass"] for c in checks),
        "checks": checks,
        "titles": titles,
        "fallback_used": result.fallback_used,
        "latency_ms": latency_ms,
        "summary": result.summary,
        "notes": case.get("notes", ""),
    }


def render_results(outcomes: List[Dict[str, Any]], live: bool) -> str:
    lines = [
        "# VibeMatch Evaluation Results",
        "",
        f"Mode: **{'live Gemini API' if live else 'mocked Gemini (CI-safe)'}**",
        "",
        "| Case | Name | Result | Latency (ms) | Top recommendations |",
        "|---|---|---|---|---|",
    ]
    for o in outcomes:
        status = "PASS" if o["pass"] else "FAIL"
        titles = ", ".join(o["titles"][:3]) or "(none)"
        lines.append(f"| {o['id']} | {o['name']} | {status} | {o['latency_ms']:.0f} | {titles} |")

    lines.append("")
    lines.append("## Check-level detail")
    lines.append("")
    for o in outcomes:
        lines.append(f"### {o['id']}: {o['name']} — {'PASS' if o['pass'] else 'FAIL'}")
        lines.append("")
        for c in o["checks"]:
            mark = "x" if c["pass"] else " "
            lines.append(f"- [{mark}] {c['check']}")
        lines.append(f"- Fallback used: {o['fallback_used']}")
        if o["notes"]:
            lines.append(f"- Notes: {o['notes']}")
        lines.append("")

    passed = sum(1 for o in outcomes if o["pass"])
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Cases passed: {passed}/{len(outcomes)}")
    avg = sum(o["latency_ms"] for o in outcomes) / len(outcomes)
    lines.append(f"- Average latency: {avg:.0f} ms")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VibeMatch evaluation cases.")
    parser.add_argument("--live", action="store_true", help="Use the real Gemini API")
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    docs = load_documents_from_csv("data/songs.csv")
    outcomes = [evaluate_case(case, docs, live=args.live) for case in cases]

    report = render_results(outcomes, live=args.live)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)

    return 0 if all(o["pass"] for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
