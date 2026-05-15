#!/usr/bin/env python3
"""
eval.py  –  Replay the 10 sample conversations against a running service.

Usage:
    python eval.py [base_url]

Default base_url: http://localhost:8000
"""

import json
import sys
import time
import httpx

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

# ─── Typed expected shortlists (from sample conversations) ──────────────────

TRACES = [
    {
        "id": "C1",
        "desc": "Senior leadership (CXOs/Directors) – selection vs leadership benchmark",
        "turns": [
            "We need a solution for senior leadership.",
            "The pool consists of CXOs, director-level positions; people with more than 15 years of experience.",
            "Selection — comparing candidates against a leadership benchmark.",
            "Perfect, that's what we need.",
        ],
        "expected": [
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ Universal Competency Report 2.0",
            "OPQ Leadership Report",
        ],
    },
    {
        "id": "C2",
        "desc": "Senior Rust engineer – systems/networking, no Rust test in catalog",
        "turns": [
            "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?",
            "Yes, go ahead. Should I also add a cognitive test for this level?",
            "That works. Thanks.",
        ],
        "expected": [
            "Smart Interview Live Coding",
            "Linux Programming (General)",
            "Networking and Implementation (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C3",
        "desc": "500 entry-level contact centre agents – English US",
        "turns": [
            "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?",
            "English.",
            "US.",
            "Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?",
            "Perfect — new simulation for volume, old solution for finalists. Confirmed.",
        ],
        "expected": [
            "SVAR Spoken English (US) (New)",
            "Contact Center Call Simulation (New)",
            "Entry Level Customer Serv-Retail & Contact Center",
            "Customer Service Phone Simulation",
        ],
    },
    {
        "id": "C4",
        "desc": "Graduate financial analysts – numerical + finance + SJT",
        "turns": [
            "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test.",
            "Good. Can you also add a situational judgement element — work-context decision making for graduates?",
            "That covers it. Numerical + Graduate Scenarios as first filter, domain tests for shortlisted candidates.",
        ],
        "expected": [
            "SHL Verify Interactive – Numerical Reasoning",
            "Financial Accounting (New)",
            "Basic Statistics (New)",
            "Graduate Scenarios",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C5",
        "desc": "Sales org re-skilling audit",
        "turns": [
            "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?",
            "What's the difference between OPQ and OPQ MQ Sales Report?",
            "Clear. We'll use OPQ for everyone and add MQ only where we want motivators in the Sales Report; keeping the five solutions as our audit stack.",
        ],
        "expected": [
            "Global Skills Assessment",
            "Global Skills Development Report",
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ MQ Sales Report",
            "Sales Transformation 2.0 - Individual Contributor",
        ],
    },
    {
        "id": "C6",
        "desc": "Plant operators – chemical facility, industrial safety",
        "turns": [
            "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?",
            "What's the difference between the DSI and the Safety & Dependability 8.0?",
            "We're industrial. The 8.0 bundle is the right fit. Confirmed.",
        ],
        "expected": [
            "Manufac. & Indust. - Safety & Dependability 8.0",
            "Workplace Health and Safety (New)",
        ],
    },
    {
        "id": "C7",
        "desc": "Bilingual healthcare admin – South Texas, HIPAA, Spanish-capable",
        "turns": [
            "We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical. What assessments work?",
            "They're functionally bilingual — English fluent for written work. Go with the hybrid.",
            "Are we legally required under HIPAA to test all staff who touch patient records? And does this SHL test satisfy that requirement?",
            "Understood. Keep the shortlist as-is.",
        ],
        "expected": [
            "HIPAA (Security)",
            "Medical Terminology (New)",
            "Microsoft Word 365 - Essentials (New)",
            "Dependability and Safety Instrument (DSI)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C8",
        "desc": "Admin assistants – Excel/Word quick screen, then add simulations",
        "turns": [
            "I need to quickly screen admin assistants for Excel and Word daily.",
            "In that case, I am OK with adding a simulation - we want to capture the capabilities.",
            "That's good.",
        ],
        "expected": [
            "Microsoft Excel 365 (New)",
            "Microsoft Word 365 (New)",
            "MS Excel (New)",
            "MS Word (New)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C9",
        "desc": "Senior Full-Stack (backend-leaning) Java engineer",
        "turns": [
            'Here\'s the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker. Will own end-to-end microservice delivery, contribute to architectural decisions, and mentor mid-level engineers. Strong CI/CD and cloud-native experience required."',
            "Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant. Angular is occasional — they'd review frontend PRs but not own features.",
            "Senior IC. They lead design on their own services but don't manage other engineers directly.",
            "Add AWS and Docker. Drop REST — the API design signal will already come through in Spring and the live interview.",
            "On Java — they'd be working on existing services, not greenfield. Is the Advanced level the right pick?",
            "Do we really need Verify G+ on top of all the technical tests? Feels redundant.",
            "Keep Verify G+. Locking it in.",
        ],
        "expected": [
            "Core Java (Advanced Level) (New)",
            "Spring (New)",
            "SQL (New)",
            "Amazon Web Services (AWS) Development (New)",
            "Docker (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C10",
        "desc": "Graduate management trainee – cognitive + SJT, user drops OPQ",
        "turns": [
            "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates.",
            "But can you remove the OPQ32r and replace it with something shorter? Candidates complain it takes too long.",
            "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios.",
        ],
        "expected": [
            "SHL Verify Interactive G+",
            "Graduate Scenarios",
        ],
    },
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def recall_at_k(recommended: list[str], expected: list[str], k: int = 10) -> float:
    if not expected:
        return 1.0
    top_k = set(r.lower() for r in recommended[:k])
    hits  = sum(1 for e in expected if e.lower() in top_k)
    return hits / len(expected)


def check_schema(resp: dict) -> list[str]:
    errors = []
    if "reply" not in resp:
        errors.append("missing 'reply'")
    if "recommendations" not in resp:
        errors.append("missing 'recommendations'")
    elif not isinstance(resp["recommendations"], list):
        errors.append("'recommendations' is not a list")
    elif len(resp["recommendations"]) > 10:
        errors.append(f"'recommendations' has {len(resp['recommendations'])} items (max 10)")
    if "end_of_conversation" not in resp:
        errors.append("missing 'end_of_conversation'")
    return errors


# ─── Main ────────────────────────────────────────────────────────────────────

def run_trace(trace: dict, client: httpx.Client) -> dict:
    history = []
    final_recs: list[str] = []
    schema_errors: list[str] = []
    turn_count = 0
    eoc_seen = False

    print(f"\n{'─'*60}")
    print(f"[{trace['id']}] {trace['desc']}")

    for user_turn in trace["turns"]:
        if eoc_seen:
            break
        turn_count += 1
        if turn_count > 8:
            print("  ⚠ Turn cap (8) would be exceeded – stopping")
            break

        history.append({"role": "user", "content": user_turn})

        t0 = time.time()
        try:
            r = client.post(f"{BASE}/chat", json={"messages": history}, timeout=30)
            latency = time.time() - t0
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  Turn {turn_count}: ERROR – {e}")
            schema_errors.append(f"turn {turn_count}: HTTP error")
            break

        errs = check_schema(data)
        schema_errors.extend([f"turn {turn_count}: {e}" for e in errs])

        recs = data.get("recommendations", [])
        reply = data.get("reply", "")[:120].replace("\n", " ")
        eoc   = data.get("end_of_conversation", False)

        print(f"  T{turn_count} ({latency:.1f}s)  recs={len(recs)}  eoc={eoc}")
        print(f"    reply: {reply}…")

        if recs:
            final_recs = [rec["name"] for rec in recs]

        history.append({"role": "assistant", "content": data.get("reply", "")})

        if eoc:
            eoc_seen = True

    recall = recall_at_k(final_recs, trace["expected"])
    print(f"  Recall@10 = {recall:.2f}  ({len(final_recs)} recs, {len(trace['expected'])} expected)")

    missing = [e for e in trace["expected"] if e.lower() not in {r.lower() for r in final_recs}]
    if missing:
        print(f"  Missing: {missing}")

    return {
        "id": trace["id"],
        "recall": recall,
        "schema_errors": schema_errors,
        "turns_used": turn_count,
        "eoc_seen": eoc_seen,
    }


def main() -> None:
    print(f"Evaluating against {BASE}")

    # Health check
    try:
        r = httpx.get(f"{BASE}/health", timeout=10)
        assert r.json().get("status") == "ok", r.text
        print("✓ /health OK")
    except Exception as e:
        print(f"✗ /health FAILED: {e}")
        sys.exit(1)

    results = []
    with httpx.Client() as client:
        for trace in TRACES:
            try:
                res = run_trace(trace, client)
                results.append(res)
            except Exception as e:
                print(f"  TRACE ERROR: {e}")
                results.append({"id": trace["id"], "recall": 0.0, "schema_errors": [str(e)], "turns_used": 0, "eoc_seen": False})

    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    mean_recall = sum(r["recall"] for r in results) / len(results)
    total_schema_errors = sum(len(r["schema_errors"]) for r in results)
    total_eoc = sum(1 for r in results if r["eoc_seen"])

    for r in results:
        status = "✓" if not r["schema_errors"] else "✗"
        print(f"  {status} [{r['id']}]  recall={r['recall']:.2f}  turns={r['turns_used']}  eoc={r['eoc_seen']}  schema_errors={len(r['schema_errors'])}")

    print(f"\nMean Recall@10 : {mean_recall:.3f}")
    print(f"Schema errors  : {total_schema_errors}")
    print(f"EOC triggered  : {total_eoc}/{len(results)}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
