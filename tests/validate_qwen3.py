"""Validation suite: does Qwen3-8B catch C1-level English errors?

This is NOT a pytest suite — it's a live validation script that sends
real messages to the LLM and checks whether it identifies the embedded
errors.  Each scenario has one or more deliberate errors and the expected
correction.

Run with: python tests/validate_qwen3.py

Scoring:
  PASS  — model identified the error (correction appears in metadata)
  PARTIAL — model corrected something but missed the key error
  FAIL  — model didn't catch the error at all
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, "src")

from language_tutor import db, llm

SCENARIOS = [
    # 1. Preposition error (C1 — subtle)
    {
        "id": 1,
        "category": "preposition",
        "input": "I really depend of my team to deliver good results at work.",
        "expected_errors": ["depend of → depend on"],
        "description": "depend ON (not OF) — common even for advanced learners",
    },
    # 2. Collocation error (C1)
    {
        "id": 2,
        "category": "collocation",
        "input": "I need to do a decision about my career before the end of the year.",
        "expected_errors": ["do a decision → make a decision"],
        "description": "MAKE a decision (not DO) — classic collocation mistake",
    },
    # 3. Article error (C1 — zero article)
    {
        "id": 3,
        "category": "article",
        "input": "The life is too short to worry about the small things.",
        "expected_errors": ["The life → Life"],
        "description": "Zero article with abstract nouns (life, love, time)",
    },
    # 4. Word order (C1 — adverb placement)
    {
        "id": 4,
        "category": "word_order",
        "input": "I always have wanted to visit Japan and experience the culture.",
        "expected_errors": ["always have wanted → have always wanted"],
        "description": "Adverb between auxiliary and main verb",
    },
    # 5. Conditional error (C1 — third conditional)
    {
        "id": 5,
        "category": "grammar",
        "input": "If I would have known about the meeting, I would have attended it.",
        "expected_errors": ["If I would have known → If I had known"],
        "description": "Third conditional: IF + past perfect, not IF + would have",
    },
    # 6. Subject-verb agreement (C1 — tricky)
    {
        "id": 6,
        "category": "grammar",
        "input": "The number of people who uses AI tools at work have increased dramatically.",
        "expected_errors": ["who uses → who use", "have increased → has increased"],
        "description": "Double agreement error: people USE, the number HAS",
    },
    # 7. False friend / vocabulary (C1)
    {
        "id": 7,
        "category": "vocabulary",
        "input": "I am very sensible about climate change and try to reduce my carbon footprint.",
        "expected_errors": ["sensible → sensitive"],
        "description": "sensible = prudent; sensitive = emotionally aware (false friend)",
    },
    # 8. Register mismatch (C1 — formal context)
    {
        "id": 8,
        "category": "register",
        "input": "The research findings are gonna impact how companies approach sustainability.",
        "expected_errors": ["gonna → going to"],
        "description": "Informal 'gonna' in academic/formal context",
    },
    # 9. Phrasal verb error (C1)
    {
        "id": 9,
        "category": "phrasal_verb",
        "input": "I need to look up for new opportunities in the AI field.",
        "expected_errors": ["look up for → look for"],
        "description": "look FOR opportunities (not look UP for) — wrong phrasal verb",
    },
    # 10. Subtle tense error (C1 — present perfect vs past simple)
    {
        "id": 10,
        "category": "grammar",
        "input": "I lived in London for three years and I still enjoy the city very much.",
        "expected_errors": ["lived → have lived"],
        "description": "Still ongoing → present perfect (have lived), not past simple",
    },
    # 11. Relative clause error (C1)
    {
        "id": 11,
        "category": "grammar",
        "input": "The company which CEO resigned last week is now looking for a replacement.",
        "expected_errors": ["which CEO → whose CEO"],
        "description": "Possessive relative pronoun: WHOSE (not which)",
    },
    # 12. Near-miss vocabulary (C1 — subtle word choice)
    {
        "id": 12,
        "category": "vocabulary",
        "input": "The teacher said that my essay was very good and I should continue to do progress.",
        "expected_errors": ["do progress → make progress"],
        "description": "MAKE progress (not DO progress) — another MAKE/DO confusion",
    },
]


@dataclass
class Result:
    scenario_id: int
    category: str
    description: str
    status: str  # PASS, PARTIAL, FAIL
    expected: list[str]
    found: list[str]
    tutor_response: str
    latency_s: float


def run_scenario(scenario: dict) -> Result:
    """Run a single validation scenario against the LLM."""
    tutor = llm.TutorLLM(due_cards=[])

    start = time.time()
    response = tutor.chat(scenario["input"])
    latency = time.time() - start

    # Collect all corrections found
    found = []
    for c in response.metadata.corrections:
        user_said = c.get("user_said", "")
        corrected = c.get("corrected", "")
        found.append(f"{user_said} → {corrected}")

    # Check if the key errors were caught
    expected = scenario["expected_errors"]
    hits = 0
    for exp in expected:
        exp_parts = exp.lower().replace("→", "->").split("->")
        exp_wrong = exp_parts[0].strip() if len(exp_parts) > 0 else ""
        exp_right = exp_parts[1].strip() if len(exp_parts) > 1 else ""

        for f in found:
            f_lower = f.lower()
            # Check if either the wrong or right part appears in what was found
            if exp_wrong in f_lower or exp_right in f_lower:
                hits += 1
                break

    if hits == len(expected):
        status = "PASS"
    elif hits > 0:
        status = "PARTIAL"
    else:
        status = "FAIL"

    return Result(
        scenario_id=scenario["id"],
        category=scenario["category"],
        description=scenario["description"],
        status=status,
        expected=expected,
        found=found,
        tutor_response=response.message[:150],
        latency_s=round(latency, 1),
    )


def main() -> None:
    print("=" * 70)
    print("Qwen3-8B C1 Error Detection Validation")
    print(f"Scenarios: {len(SCENARIOS)}")
    print("=" * 70)
    print()

    results: list[Result] = []

    for scenario in SCENARIOS:
        print(f"[{scenario['id']:2d}/12] {scenario['category']:15s} — ", end="", flush=True)
        result = run_scenario(scenario)
        results.append(result)

        icon = {"PASS": "✅", "PARTIAL": "⚠️ ", "FAIL": "❌"}.get(result.status, "?")
        print(f"{icon} {result.status:8s} ({result.latency_s}s)")

        if result.status != "PASS":
            print(f"         Expected: {result.expected}")
            print(f"         Found:    {result.found if result.found else '(none)'}")
        print()

    # Summary
    passed = sum(1 for r in results if r.status == "PASS")
    partial = sum(1 for r in results if r.status == "PARTIAL")
    failed = sum(1 for r in results if r.status == "FAIL")
    avg_latency = sum(r.latency_s for r in results) / len(results)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  PASS:    {passed}/{len(results)}")
    print(f"  PARTIAL: {partial}/{len(results)}")
    print(f"  FAIL:    {failed}/{len(results)}")
    print(f"  Avg latency: {avg_latency:.1f}s")
    print()

    rate = (passed + partial * 0.5) / len(results) * 100
    print(f"  Detection rate: {rate:.0f}%")
    if rate >= 70:
        print("  Verdict: ACCEPTABLE for MVP — model catches most C1 errors")
    elif rate >= 50:
        print("  Verdict: MARGINAL — consider supplementing with CoEdIT or larger model")
    else:
        print("  Verdict: INSUFFICIENT — need a better model or API fallback")

    # Detailed report for failures
    failures = [r for r in results if r.status in ("FAIL", "PARTIAL")]
    if failures:
        print()
        print("DETAILS — Failed/Partial scenarios:")
        print("-" * 70)
        for r in failures:
            print(f"  #{r.scenario_id} [{r.category}] {r.description}")
            print(f"    Expected: {r.expected}")
            print(f"    Found:    {r.found if r.found else '(none)'}")
            print(f"    Response: {r.tutor_response}...")
            print()


if __name__ == "__main__":
    main()
