"""SafeCDS LangGraph Orchestration & Closed-Loop State Machine (FR1, FR5, FR6, FR7)."""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from src.agents.llm_client import LLMClinicalAgent, PrescriptionHypothesis
from src.reasoning.verifier import ClinicalVerifier, VerificationResult
from src.retrieval.hybrid_retriever import HybridClinicalRetriever

logger = logging.getLogger("safecds.workflow")


class WorkflowState(TypedDict):
    """Auditable state schema tracked across LangGraph transitions."""
    patient_id: str
    conditions: List[str]
    clinical_notes: str
    retrieved_evidence: str
    proposed_drug: Optional[str]
    hypothesis: Optional[Dict[str, Any]]
    verification_result: Optional[Dict[str, Any]]
    axiom_feedback: List[str]
    retries: int
    max_retries: int
    status: str  # "PENDING", "APPROVED", "ESCALATED_TO_HUMAN"
    audit_trail: List[Dict[str, Any]]


class SafeCDSWorkflow:
    """End-to-end fail-closed neuro-symbolic clinical decision support state machine."""

    def __init__(
        self,
        retriever: Optional[HybridClinicalRetriever] = None,
        llm_agent: Optional[LLMClinicalAgent] = None,
        verifier: Optional[ClinicalVerifier] = None,
        max_retries: int = 2,
    ):
        self.retriever = retriever or HybridClinicalRetriever()
        self.llm_agent = llm_agent or LLMClinicalAgent()
        self.verifier = verifier or ClinicalVerifier()
        self.max_retries = max_retries
        self.app = self._build_graph()

    def _build_graph(self):
        """Construct LangGraph StateGraph workflow with closed-loop self-correction."""
        try:
            from langgraph.graph import END, StateGraph

            workflow = StateGraph(WorkflowState)

            # Define graph nodes
            workflow.add_node("ingest", self.node_ingest)
            workflow.add_node("retrieve", self.node_retrieve)
            workflow.add_node("generate", self.node_generate)
            workflow.add_node("verify", self.node_verify)
            workflow.add_node("feedback", self.node_feedback)
            workflow.add_node("finalize", self.node_finalize)
            workflow.add_node("escalate", self.node_escalate)

            # Define graph edges
            workflow.set_entry_point("ingest")
            workflow.add_edge("ingest", "retrieve")
            workflow.add_edge("retrieve", "generate")
            workflow.add_edge("generate", "verify")

            # Conditional routing based on formal verification outcome
            workflow.add_conditional_edges(
                "verify",
                self.decide_verification_route,
                {
                    "approved": "finalize",
                    "retry": "feedback",
                    "escalate": "escalate",
                },
            )

            # Self-correction loop: feedback -> generate
            workflow.add_edge("feedback", "generate")
            workflow.add_edge("finalize", END)
            workflow.add_edge("escalate", END)

            return workflow.compile()
        except ImportError:
            logger.warning("LangGraph not installed. Utilizing deterministic state-machine fallback engine.")
            return None

    # --- Graph Node Implementations ---

    def node_ingest(self, state: WorkflowState) -> WorkflowState:
        """FR1: Structured Ingestion and State Initialization."""
        state["retries"] = 0
        state["status"] = "PENDING"
        state["axiom_feedback"] = []
        state["audit_trail"] = []
        state["audit_trail"].append({
            "stage": "INGESTION",
            "message": f"Patient {state['patient_id']} state initialized.",
        })
        return state

    def node_retrieve(self, state: WorkflowState) -> WorkflowState:
        """FR2: Hybrid Dense-Sparse Guideline Retrieval with RRF."""
        query = f"{' '.join(state['conditions'])} {state['clinical_notes']}"
        evidence = self.retriever.get_evidence_string(query, top_k=3)
        state["retrieved_evidence"] = evidence
        state["audit_trail"].append({
            "stage": "RETRIEVAL",
            "query": query,
            "evidence_length": len(evidence),
        })
        return state

    def node_generate(self, state: WorkflowState) -> WorkflowState:
        """FR3: Schema-Constrained Hypothesis Generation (Pydantic V2)."""
        hypothesis: PrescriptionHypothesis = self.llm_agent.generate(
            patient_id=state["patient_id"],
            conditions=state["conditions"],
            notes=state["clinical_notes"],
            guidelines_evidence=state["retrieved_evidence"],
            axiom_feedback=state["axiom_feedback"],
        )
        state["proposed_drug"] = hypothesis.recommended_drug
        state["hypothesis"] = hypothesis.model_dump()
        state["audit_trail"].append({
            "stage": "GENERATION",
            "attempt": state["retries"] + 1,
            "proposal": hypothesis.model_dump(),
        })
        return state

    def node_verify(self, state: WorkflowState) -> WorkflowState:
        """FR4: Isolated Symbolic Verification via HermiT DL Reasoner."""
        v_result: VerificationResult = self.verifier.verify(
            patient_id=state["patient_id"],
            conditions=state["conditions"],
            proposed_drug=state["proposed_drug"],
        )
        state["verification_result"] = {
            "is_safe": v_result.is_safe,
            "status": v_result.status,
            "violated_axioms": v_result.violated_axioms,
            "feedback_message": v_result.feedback_message,
            "inferred_classes": v_result.inferred_classes,
            "latency_seconds": v_result.latency_seconds,
        }
        state["audit_trail"].append({
            "stage": "VERIFICATION",
            "attempt": state["retries"] + 1,
            "status": v_result.status,
            "violated_axioms": v_result.violated_axioms,
            "latency": v_result.latency_seconds,
        })
        return state

    def decide_verification_route(self, state: WorkflowState) -> str:
        """Route to finalize, self-correction retry, or human escalation."""
        v_result = state.get("verification_result", {})
        if v_result.get("is_safe", False):
            return "approved"
        elif state["retries"] < state.get("max_retries", self.max_retries):
            return "retry"
        else:
            return "escalate"

    def node_feedback(self, state: WorkflowState) -> WorkflowState:
        """FR5: Automated Axiom Feedback Injection for Closed-Loop Self-Correction."""
        state["retries"] += 1
        v_result = state["verification_result"]
        feedback = v_result.get("feedback_message", "Prescription contraindication detected.")
        state["axiom_feedback"].append(feedback)
        state["audit_trail"].append({
            "stage": "FEEDBACK_INJECTION",
            "retries_count": state["retries"],
            "feedback": feedback,
        })
        return state

    def node_finalize(self, state: WorkflowState) -> WorkflowState:
        """FR7: Audit Trail and Safe Output Delivery."""
        state["status"] = "APPROVED"
        state["audit_trail"].append({
            "stage": "FINALIZATION",
            "decision": "APPROVED",
            "approved_drug": state["proposed_drug"],
        })
        return state

    def node_escalate(self, state: WorkflowState) -> WorkflowState:
        """FR6: Fail-Closed Human Escalation Protocol."""
        state["status"] = "ESCALATED_TO_HUMAN"
        state["audit_trail"].append({
            "stage": "ESCALATION",
            "decision": "ESCALATED_TO_HUMAN",
            "reason": "Maximum automated self-correction attempts exhausted without safety verification.",
        })
        return state

    def run(self, initial_state: WorkflowState) -> WorkflowState:
        """Execute state machine to completion."""
        if self.app is not None:
            return self.app.invoke(initial_state)

        # Fallback pure-python state machine execution loop
        current = self.node_ingest(initial_state)
        current = self.node_retrieve(current)

        while True:
            current = self.node_generate(current)
            current = self.node_verify(current)
            route = self.decide_verification_route(current)

            if route == "approved":
                current = self.node_finalize(current)
                break
            elif route == "retry":
                current = self.node_feedback(current)
            else:
                current = self.node_escalate(current)
                break

        return current
