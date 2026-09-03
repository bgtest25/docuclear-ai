# DocuClear AI

Enterprise-grade Legal Contract Harmonization & Risk Engine. A 6-agent LangGraph
assembly line that parses unstructured contract text, audits it against corporate
compliance policy, iteratively redrafts high-risk clauses, pulls on-demand
jurisdiction intelligence, and gates final approval behind a human-in-the-loop
review when automated remediation can't fully resolve the risk.

## Architecture

```
START -> Security Guard -> Clause Parser -> Risk Auditor -+-> Executive Gate -> END
                |                               ^          |
                +-> END (malicious input)       |          +-> Regulatory Monitor -> Redrafter
                                                 +--------------------------------------+
```

1. **Inbound Security Guard** — pattern-matches incoming text for prompt-injection /
   social-engineering attempts before anything else runs. Flags `security_flag` and
   routes straight to `END` on a hit.
2. **Clause Parser** — extracts `Limitation of Liability`, `Indemnification`,
   `Intellectual Property Ownership`, and `Governing Law / Jurisdiction` clauses into
   a structured `MasterContractReport`.
3. **Legal Risk Auditor** — asks Claude to score each clause `LOW`/`MED`/`HIGH`
   against corporate policy (liability capped at $1,000,000; IP must remain
   proprietary to the buyer).
4. **Redrafting Engineer** — rewrites any `HIGH`-risk clause to bring it into
   compliance while preserving the commercial relationship, incrementing
   `review_rounds`.
5. **Regulatory Monitor** — runs a live DuckDuckGo search for jurisdiction-specific
   compliance standards when a clause depends on state law (e.g. "governing law of
   Delaware"), feeding the result back into the Redrafter's context.
6. **Executive Gate** — final QC. If every clause is compliant, the contract is
   approved. If `review_rounds >= 3` or a clause is still unmitigated, it fires a
   native LangGraph `interrupt()` and hands the report to human General Counsel for
   manual override.

Every run is checkpointed via `SqliteSaver` into `audit_trail.db`, so execution
trees (including paused interrupts) persist across process restarts.

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install langgraph langchain-anthropic langchain-community pydantic python-dotenv duckduckgo-search langgraph-checkpoint-sqlite
cp .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY` to a live key. If your key is
**identity-linked** (rather than scoped to a single workspace), you'll also need to
set `ANTHROPIC_WORKSPACE_ID` — find it in the Anthropic Console under
Settings → Workspaces, or in the workspace's URL.

## Running the tests

```bash
python run_tests.py
```

Runs three isolated scenarios on separate checkpointer threads:

| Scenario | Input | Expected outcome |
|---|---|---|
| 1. Compliant vendor agreement | Liability capped at $500k, IP to buyer | Approved with 0 review rounds |
| 2. Hostile vendor MSA | Unlimited liability, IP waived, Delaware law | Risk Auditor flags violations, Redrafter iterates, may trigger the Executive Gate interrupt if unresolved after 3 rounds |
| 3. Malicious injection attempt | Attempts to override system instructions | Blocked by the Security Guard before reaching any other agent or the LLM |

## Known limitations

- Risk scoring and clause extraction are LLM-driven, so results can vary slightly
  between runs even with the same input.
- The Regulatory Monitor's web search quality depends on DuckDuckGo result
  availability at runtime.
