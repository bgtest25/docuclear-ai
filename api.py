"""FastAPI service wrapping the DocuClear AI LangGraph pipeline.

Exposes the compiled 6-agent graph as a small HTTP API so it can be driven
from a UI (or any other client) instead of only via run_tests.py.

Run locally with:
    uvicorn api:app --reload --port 8000
"""

import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app import graph, MasterContractReport

app = FastAPI(title="DocuClear AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SubmitContractRequest(BaseModel):
    contract_text: str


class ResumeRequest(BaseModel):
    override: bool


class ContractResponse(BaseModel):
    thread_id: str
    security_flag: bool
    interrupted: bool
    interrupt_payload: Optional[dict[str, Any]] = None
    report: Optional[dict[str, Any]] = None
    agent_log: list[str]


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _serialize(thread_id: str, result: dict) -> ContractResponse:
    agent_log = [
        m.content for m in result.get("messages", [])
        if isinstance(m, AIMessage)
    ]

    interrupted = "__interrupt__" in result
    interrupt_payload = result["__interrupt__"][0].value if interrupted else None

    report = result.get("report")
    report_dict = report.model_dump() if isinstance(report, MasterContractReport) else report

    return ContractResponse(
        thread_id=thread_id,
        security_flag=bool(result.get("security_flag", False)),
        interrupted=interrupted,
        interrupt_payload=interrupt_payload,
        report=report_dict,
        agent_log=agent_log,
    )


@app.post("/contracts", response_model=ContractResponse)
def submit_contract(req: SubmitContractRequest) -> ContractResponse:
    thread_id = str(uuid.uuid4())
    initial_state = {
        "messages": [HumanMessage(content=req.contract_text)],
        "report": MasterContractReport(contractor_name="Unknown", parsed_clauses=[]),
        "security_flag": False,
    }
    result = graph.invoke(initial_state, config=_config(thread_id))
    return _serialize(thread_id, result)


@app.get("/contracts/{thread_id}", response_model=ContractResponse)
def get_contract(thread_id: str) -> ContractResponse:
    state = graph.get_state(_config(thread_id))
    if not state.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    result = dict(state.values)
    if state.interrupts:
        result["__interrupt__"] = state.interrupts
    return _serialize(thread_id, result)


@app.post("/contracts/{thread_id}/resume", response_model=ContractResponse)
def resume_contract(thread_id: str, req: ResumeRequest) -> ContractResponse:
    state = graph.get_state(_config(thread_id))
    if not state.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    if not state.interrupts:
        raise HTTPException(status_code=409, detail="This contract has no pending interrupt")

    result = graph.invoke(Command(resume={"override": req.override}), config=_config(thread_id))
    return _serialize(thread_id, result)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
