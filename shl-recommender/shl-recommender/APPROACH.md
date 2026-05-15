# SHL Assessment Recommender – Approach Document

## 1. Problem Decomposition

The task requires an agent that can:
- **Clarify** vague hiring intent into actionable role context
- **Retrieve** relevant assessments from the SHL catalog (122 items)
- **Recommend** 1–10 grounded shortlists with exact catalog URLs
- **Refine** shortlists mid-conversation without losing state
- **Compare** two assessments using catalog data only
- **Refuse** off-topic requests (legal advice, hiring guidance, prompt injection)

The primary challenge is **context engineering**: the entire 122-item catalog must be injected into every request so the LLM never hallucinates an assessment or URL.

---

## 2. Architecture

```
POST /chat
  │
  ▼
FastAPI (stateless)
  │  receives full conversation history each call
  ▼
System Prompt = behavior rules + full catalog text (injected at startup)
  │
  ▼
Claude claude-sonnet-4-20250514 via Anthropic API
  │  returns strict JSON: { reply, recommendations[], end_of_conversation }
  ▼
Sanitizer: filter recommendations to catalog-only URLs
  │
  ▼
ChatResponse { reply, recommendations, end_of_conversation }
```

**No vector store / RAG needed.** The entire catalog fits in the context window (~30k tokens of catalog text). This eliminates retrieval errors, keeps latency low, and makes the system fully deterministic for catalog grounding.

---

## 3. Context Engineering

### Catalog injection
Every catalog item is serialized as structured text (NAME / URL / TYPE / KEYS / DURATION / LEVELS / LANGUAGES / DESC). The full block is injected into the system prompt on every call via `_SYSTEM_TEMPLATE.format(catalog=CATALOG_TEXT)`.

### Behavioral rules in system prompt
The system prompt encodes six explicit rules covering: clarify, recommend, refine, compare, refuse, and end_of_conversation. Each rule maps to a sample conversation pattern from the provided traces.

### Heuristics for recall improvement
Role-type → assessment heuristics (e.g. "executive → OPQ32r + Leadership Report", "contact centre → SVAR + CC Simulation + Entry Level CS", "safety-critical → DSI or Safety 8.0") are embedded as selection guidance. This is the key driver of Recall@10 versus a generic "choose what matches" instruction.

### Output format enforcement
The LLM is instructed to return only a JSON object. A fallback parser strips markdown fences and falls back to regex extraction if needed, preventing schema errors from breaking the evaluator.

---

## 4. Agent Design Decisions

| Decision | Rationale |
|---|---|
| Stateless API | Matches spec; simpler deployment; conversation history replayed each turn |
| Full catalog in context | Eliminates retrieval errors; 122 items fits easily; no vector store complexity |
| Structured JSON output | Directly parseable by evaluator; no post-processing ambiguity |
| URL sanitizer | Hard guarantee: any URL not in the catalog is rejected, never returned |
| `end_of_conversation` only on explicit user signal | Prevents premature termination on phrases like "that's helpful" |
| OPQ32r default inclusion | Matches expected shortlists across 8/10 traces; dropped only on explicit user request |
| Compare turns → recommendations: [] | Prevents stale shortlist from persisting through a comparison-only turn |

---

## 5. Evaluation Approach

**eval.py** replays all 10 public conversation traces and reports:
- **Recall@10** per trace and mean across all traces
- **Schema compliance** (required fields, 1–10 recs, correct types)
- **Turn cap compliance** (≤8 turns per conversation)
- **EOC trigger accuracy**

Failure modes tested:
- Vague first turn (C1) → must clarify, not recommend
- No catalog match (C2, Rust) → must acknowledge gap, suggest closest alternatives
- Mid-conversation refinement (C8, C9, C10) → must update shortlist without restart
- Comparison question (C3, C5, C6) → must answer from catalog data, no hallucination
- Legal question (C7) → must refuse, not answer
- Explicit user item removal (C10, OPQ dropped) → must honor the removal

---

## 6. What Didn't Work

- **RAG / vector retrieval**: Tested a keyword-similarity approach first. Retrieval errors (wrong items surfaced for "Java") caused hallucinations. Full catalog injection was more reliable and simpler.
- **Asking for test_type in output**: Initial prompt asked for a human-readable test type. Changed to the single-char codes from the catalog after the evaluator schema check failed.
- **Overly aggressive EOC**: First version set `end_of_conversation: true` on any positive user message. Fixed by restricting to explicit closure phrases.

---

## 7. Stack

| Layer | Choice | Reason |
|---|---|---|
| API framework | FastAPI | Fast, async, automatic OpenAPI docs |
| LLM | Claude claude-sonnet-4-20250514 | Strong instruction following; reliable JSON output |
| Catalog storage | JSON file, loaded at startup | Simple; no DB dependency; 122 items fits in memory |
| Deployment | Render (render.yaml included) | Free tier, auto-deploy from GitHub, supports env vars |
| Evaluation | Custom eval.py replay harness | Mirrors the grading harness described in the assignment |

**AI tools used**: Claude assisted with boilerplate FastAPI structure and eval script skeleton. All design decisions, prompt engineering, heuristics, and catalog handling were written and verified manually.
