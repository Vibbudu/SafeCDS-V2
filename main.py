"""SafeCDS V2: Single-Patient Clinical Decision Support Demonstration Runner (FR1-FR7)."""

import argparse
import json
import logging
import sys
from typing import List

from src.agents.workflow import SafeCDSWorkflow, WorkflowState

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("safecds.main")


def run_single_patient(
    patient_id: str = "PT_002",
    conditions: List[str] = None,
    clinical_notes: str = None,
):
    """Run an individual patient through the SafeCDS neuro-symbolic pipeline."""
    if conditions is None:
        conditions = ["CKD_Stage_4_5", "Hypertension"]
    if clinical_notes is None:
        clinical_notes = (
            "Elderly patient with Stage 5 Chronic Kidney Disease (eGFR 14 mL/min/1.73m2) "
            "and uncontrolled systolic hypertension (165/95 mmHg)."
        )

    print("=" * 80)
    print(" SafeCDS: Neuro-Symbolic Agentic Clinical Decision Support (v2.0)")
    print("=" * 80)
    print(f"[*] Patient Identifier: {patient_id}")
    print(f"[*] Conditions:         {', '.join(conditions)}")
    print(f"[*] Clinical Summary:   {clinical_notes}")
    print("-" * 80)

    workflow = SafeCDSWorkflow(max_retries=2)

    initial_state: WorkflowState = {
        "patient_id": patient_id,
        "conditions": conditions,
        "clinical_notes": clinical_notes,
        "retrieved_evidence": "",
        "proposed_drug": None,
        "hypothesis": None,
        "verification_result": None,
        "axiom_feedback": [],
        "retries": 0,
        "max_retries": 2,
        "status": "PENDING",
        "audit_trail": [],
    }

    final_state = workflow.run(initial_state)

    print("\n[+] PIPELINE EXECUTION COMPLETED")
    print("-" * 80)
    print(f"Final Decision Status: {final_state['status']}")
    print(f"Final Prescribed Drug: {final_state.get('proposed_drug')}")
    print(f"Total Retries/Attempts:{final_state['retries']} retries (Attempts: {final_state['retries'] + 1})")

    hypothesis = final_state.get("hypothesis") or {}
    print(f"Dosage / Regimen:      {hypothesis.get('dosage_or_regimen')}")
    print(f"Clinical Rationale:    {hypothesis.get('clinical_rationale')}")
    print(f"Guideline Reference:   {hypothesis.get('guideline_reference')}")

    print("\n--- AUDIT TRAIL LOG ---")
    for step in final_state.get("audit_trail", []):
        stage = step.get("stage")
        if stage == "INGESTION":
            print(f"  [1. INGESTION]    {step.get('message')}")
        elif stage == "RETRIEVAL":
            print(f"  [2. RETRIEVAL]    Retrieved {step.get('evidence_length', 0)} chars of guideline evidence.")
        elif stage == "GENERATION":
            proposal = step.get("proposal", {})
            print(f"  [3. GENERATION]   Attempt {step.get('attempt')}: Proposing {proposal.get('recommended_drug')}")
        elif stage == "VERIFICATION":
            print(f"  [4. VERIFICATION] Attempt {step.get('attempt')}: Status={step.get('status')}, Violated={step.get('violated_axioms', [])}")
        elif stage == "FEEDBACK_INJECTION":
            print(f"  [5. SELF-CORRECT] Feedback: {step.get('feedback')}")
        elif stage == "FINALIZATION":
            print(f"  [6. FINALIZATION] Approved Drug: {step.get('approved_drug')}")
        elif stage == "ESCALATION":
            print(f"  [6. ESCALATION]   Escalated to human review: {step.get('reason')}")

    print("=" * 80)
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeCDS Single-Patient Runner")
    parser.add_argument("--patient-id", type=str, default="PT_002", help="Patient ID")
    parser.add_argument("--conditions", nargs="+", default=["CKD_Stage_4_5", "Hypertension"], help="Diagnosed conditions")
    parser.add_argument("--notes", type=str, default=None, help="Clinical notes")

    args = parser.parse_args()
    run_single_patient(
        patient_id=args.patient_id,
        conditions=args.conditions,
        clinical_notes=args.notes,
    )
