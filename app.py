"""DocuClear AI - Legal Contract Harmonization & Risk Engine.

6-agent LangGraph assembly line:
  1. Inbound Security Guard  - prompt-injection / social-engineering firewall
  2. Clause Parser           - structural extraction of key clauses
  3. Legal Risk Auditor      - corporate policy compliance check
  4. Redrafting Engineer     - iterative compliant rewrite
  5. Regulatory Monitor      - on-demand jurisdiction web intelligence
  6. Executive Gate          - final QC + human-in-the-loop interrupt
"""

import os
import re
import sqlite3
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import interrupt

load_dotenv()

MODEL_NAME = "claude-sonnet-5"
LIABILITY_CAP = 1_000_000
MAX_REVIEW_ROUNDS = 3

US_STATES = [
    "Delaware", "California", "New York", "Texas", "Nevada", "Florida",
    "Illinois", "Massachusetts", "Washington", "New Jersey",
]

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"override (previous |prior )?(developer |system )?instructions",
    r"disregard (previous |prior |all )?(instructions|rules|policy)",
    r"you are now",
    r"new system prompt",
    r"set risk_score to low",
    r"mark (all |every )?(liability|clause)s? as approved",
    r"bypass (corporate|compliance) (threshold|polic)",
    r"reveal (your |the )?(system prompt|instructions)",
    r"act as (if|though) you",
]


# ---------------------------------------------------------------------------
# 1. Data Integrity Layer
# ---------------------------------------------------------------------------

class LegalClause(BaseModel):
    clause_type: str
    original_text: str
    risk_score: Literal["LOW", "MED", "HIGH"] = "LOW"
    redrafted_text: str = ""


class MasterContractReport(BaseModel):
    contractor_name: str
    parsed_clauses: list[LegalClause] = Field(default_factory=list)
    compliance_passed: bool = False
    review_rounds: int = 0


class ParsedClauseIntake(BaseModel):
    """Structured-output schema used by the Clause Parser Agent."""
    contractor_name: str
    clauses: list[LegalClause]


class RedraftedClause(BaseModel):
    """Structured-output schema used by the Redrafting Engineer Agent."""
    redrafted_text: str


class RiskAssessment(BaseModel):
    """Structured-output schema used by the Legal Risk Auditor Agent."""
    risk_score: Literal["LOW", "MED", "HIGH"]
    rationale: str


# ---------------------------------------------------------------------------
# 2. Graph State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    report: MasterContractReport
    security_flag: bool


def _get_source_text(state: AgentState) -> str:
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _llm() -> ChatAnthropic:
    headers = {}
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace_id:
        headers["anthropic-workspace-id"] = workspace_id
    return ChatAnthropic(model=MODEL_NAME, default_headers=headers or None)


# ---------------------------------------------------------------------------
# Agent 1: Inbound Security Guard Node (Sanitizer)
# ---------------------------------------------------------------------------

def security_guard_node(state: AgentState) -> dict:
    text = _get_source_text(state)
    lowered = text.lower()

    hit = next((p for p in INJECTION_PATTERNS if re.search(p, lowered)), None)

    if hit:
        return {
            "security_flag": True,
            "messages": [AIMessage(
                content=(
                    "[AGENT 1 - SECURITY GUARD] Malicious input intercepted. "
                    f"Pattern matched: '{hit}'. Execution halted, request dropped."
                )
            )],
        }

    return {
        "security_flag": False,
        "messages": [AIMessage(content="[AGENT 1 - SECURITY GUARD] Input clean. Routing to Clause Parser.")],
    }


def route_after_security(state: AgentState) -> Literal["clause_parser", "__end__"]:
    return END if state["security_flag"] else "clause_parser"


# ---------------------------------------------------------------------------
# Agent 2: Clause Parser Agent (Inbound Structural Ingestion)
# ---------------------------------------------------------------------------

def clause_parser_node(state: AgentState) -> dict:
    text = _get_source_text(state)

    extractor = _llm().with_structured_output(ParsedClauseIntake)
    prompt = (
        "You are a legal clause extraction engine. From the raw contract text below, "
        "identify the contractor/counterparty name and extract every clause matching "
        "these categories where present: 'Limitation of Liability', 'Indemnification', "
        "'Intellectual Property Ownership', and 'Governing Law / Jurisdiction'. "
        "For each clause, capture the exact original_text. Do not invent clauses that "
        "aren't present in the source text.\n\n"
        f"CONTRACT TEXT:\n{text}"
    )
    result: ParsedClauseIntake = extractor.invoke(prompt)

    report = MasterContractReport(
        contractor_name=result.contractor_name,
        parsed_clauses=result.clauses,
        compliance_passed=False,
        review_rounds=0,
    )

    return {
        "report": report,
        "messages": [AIMessage(
            content=f"[AGENT 2 - CLAUSE PARSER] Extracted {len(result.clauses)} clause(s) "
                    f"for contractor '{result.contractor_name}'."
        )],
    }


# ---------------------------------------------------------------------------
# Agent 3: Legal Risk Auditor Agent (Corporate Compliance Check)
# ---------------------------------------------------------------------------

CORPORATE_POLICY = (
    f"1. Limitation of Liability: total liability exposure must not exceed ${LIABILITY_CAP:,}. "
    "Unlimited, uncapped, or ambiguous liability language is a violation.\n"
    "2. Intellectual Property Ownership: all IP created or delivered under the contract must "
    "remain solely and exclusively proprietary to the Buyer. Any waiver of Buyer's IP rights, "
    "or any grant/retention of ownership or license by the Seller/Contractor, is a violation.\n"
    "Clauses outside these two categories (e.g. governing law) carry no inherent violation "
    "unless they conflict with policies 1 or 2."
)


def _assess_clause_risk(clause: LegalClause) -> Literal["LOW", "MED", "HIGH"]:
    text = clause.redrafted_text or clause.original_text
    assessor = _llm().with_structured_output(RiskAssessment)
    prompt = (
        "You are a corporate legal compliance auditor. Assess this clause strictly against "
        "the corporate policies below and assign a risk_score of LOW, MED, or HIGH. Assign HIGH "
        "only for an actual violation of policy 1 or 2; assign MED for ambiguous or incomplete "
        "language that doesn't clearly violate policy but lacks clarity; assign LOW when the "
        "clause clearly complies.\n\n"
        f"CORPORATE POLICY:\n{CORPORATE_POLICY}\n\n"
        f"CLAUSE TYPE: {clause.clause_type}\n"
        f"CLAUSE TEXT: {text}"
    )
    assessment: RiskAssessment = assessor.invoke(prompt)
    return assessment.risk_score


def risk_auditor_node(state: AgentState) -> dict:
    report = state["report"]
    for clause in report.parsed_clauses:
        clause.risk_score = _assess_clause_risk(clause)

    high = [c.clause_type for c in report.parsed_clauses if c.risk_score == "HIGH"]
    summary = (
        f"[AGENT 3 - RISK AUDITOR] {len(high)} HIGH-risk clause(s) found: {high}"
        if high else "[AGENT 3 - RISK AUDITOR] No policy violations detected."
    )
    return {"report": report, "messages": [AIMessage(content=summary)]}


def route_after_audit(state: AgentState) -> Literal["executive_gate", "regulatory_monitor", "redrafter"]:
    report = state["report"]
    any_high = any(c.risk_score == "HIGH" for c in report.parsed_clauses)
    if not any_high:
        return "executive_gate"

    if report.review_rounds >= MAX_REVIEW_ROUNDS:
        return "executive_gate"

    needs_jurisdiction_lookup = any(
        "governing law" in c.clause_type.lower() or "jurisdiction" in c.clause_type.lower()
        for c in report.parsed_clauses
    )
    return "regulatory_monitor" if needs_jurisdiction_lookup else "redrafter"


# ---------------------------------------------------------------------------
# Agent 5: Regulatory Monitor Agent (On-Demand Web Intelligence Lookup)
# ---------------------------------------------------------------------------

def regulatory_monitor_node(state: AgentState) -> dict:
    report = state["report"]
    jurisdiction_clauses = [
        c for c in report.parsed_clauses
        if "governing law" in c.clause_type.lower() or "jurisdiction" in c.clause_type.lower()
    ]

    findings = []
    for clause in jurisdiction_clauses:
        state_name = next((s for s in US_STATES if s.lower() in clause.original_text.lower()), None)
        if not state_name:
            continue
        query = f"{state_name} state contract law limitation of liability compliance standards 2026"
        try:
            search = DuckDuckGoSearchRun()
            result = search.run(query)
        except Exception as exc:  # network/tooling may be unavailable
            result = f"(regulatory lookup unavailable: {exc})"
        findings.append(f"{state_name}: {result[:500]}")

    summary = (
        "[AGENT 5 - REGULATORY MONITOR] " + (" | ".join(findings) if findings
        else "No jurisdiction-specific clause required external lookup.")
    )
    return {"messages": [AIMessage(content=summary)]}


# ---------------------------------------------------------------------------
# Agent 4: Redrafting Engineer Agent (Iterative Negotiator)
# ---------------------------------------------------------------------------

def redrafter_node(state: AgentState) -> dict:
    report = state["report"]
    llm = _llm().with_structured_output(RedraftedClause)

    regulatory_context = "\n".join(
        m.content for m in state["messages"]
        if isinstance(m, AIMessage) and "REGULATORY MONITOR" in m.content
    )

    for clause in report.parsed_clauses:
        if clause.risk_score != "HIGH":
            continue
        prompt = (
            "You are a corporate contract negotiator. Rewrite the following clause so it "
            "strictly complies with these corporate policies: liability may not exceed "
            f"${LIABILITY_CAP:,}, and all intellectual property must remain proprietary to "
            "the buyer. Preserve as much of the original commercial relationship and intent "
            "as possible while achieving strict compliance.\n\n"
            f"CLAUSE TYPE: {clause.clause_type}\n"
            f"ORIGINAL TEXT: {clause.original_text}\n"
            f"RELEVANT REGULATORY CONTEXT: {regulatory_context or 'none'}"
        )
        redraft: RedraftedClause = llm.invoke(prompt)
        clause.redrafted_text = redraft.redrafted_text

    report.review_rounds += 1

    return {
        "report": report,
        "messages": [AIMessage(
            content=f"[AGENT 4 - REDRAFTING ENGINEER] Completed review round {report.review_rounds}."
        )],
    }


# ---------------------------------------------------------------------------
# Agent 6: Executive Gate Node (Compliance Safeguard)
# ---------------------------------------------------------------------------

def executive_gate_node(state: AgentState) -> dict:
    report = state["report"]
    unmitigated = [c for c in report.parsed_clauses if c.risk_score == "HIGH"]
    rounds_exhausted = report.review_rounds >= MAX_REVIEW_ROUNDS

    if not unmitigated and not rounds_exhausted:
        report.compliance_passed = True
        return {
            "report": report,
            "messages": [AIMessage(content="[AGENT 6 - EXECUTIVE GATE] Contract compliant. Clean export approved.")],
        }

    decision = interrupt({
        "reason": "review_rounds_exhausted" if rounds_exhausted else "unmitigated_high_risk_clause",
        "unmitigated_clauses": [c.clause_type for c in unmitigated],
        "review_rounds": report.review_rounds,
        "report": report.model_dump(),
        "instructions": "Return {'override': true} to approve manually, or {'override': false} to reject.",
    })

    override = bool(isinstance(decision, dict) and decision.get("override"))
    report.compliance_passed = override

    return {
        "report": report,
        "messages": [AIMessage(
            content=f"[AGENT 6 - EXECUTIVE GATE] Human General Counsel override={'GRANTED' if override else 'DENIED'}."
        )],
    }


# ---------------------------------------------------------------------------
# Graph assembly + persistence
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("security_guard", security_guard_node)
    builder.add_node("clause_parser", clause_parser_node)
    builder.add_node("risk_auditor", risk_auditor_node)
    builder.add_node("redrafter", redrafter_node)
    builder.add_node("regulatory_monitor", regulatory_monitor_node)
    builder.add_node("executive_gate", executive_gate_node)

    builder.add_edge(START, "security_guard")
    builder.add_conditional_edges("security_guard", route_after_security, {
        "clause_parser": "clause_parser",
        "__end__": END,
    })
    builder.add_edge("clause_parser", "risk_auditor")
    builder.add_conditional_edges("risk_auditor", route_after_audit, {
        "executive_gate": "executive_gate",
        "regulatory_monitor": "regulatory_monitor",
        "redrafter": "redrafter",
    })
    builder.add_edge("regulatory_monitor", "redrafter")
    builder.add_edge("redrafter", "risk_auditor")
    builder.add_edge("executive_gate", END)

    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_trail.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("app", "MasterContractReport"), ("app", "LegalClause")])
    checkpointer = SqliteSaver(conn, serde=serde)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
