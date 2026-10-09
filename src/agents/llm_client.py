"""Local LLM Agent with strict Pydantic V2 schema validation (FR3).

Interacts with locally hosted Ollama runtime (LLaMA-3.2:3b).
"""

import json
import logging
import os
from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("safecds.agents")


class PrescriptionHypothesis(BaseModel):
    """Pydantic schema contract for generative model output."""
    recommended_drug: str = Field(
        ...,
        description="Name of the prescribed drug (e.g., Insulin, Amlodipine, Metformin, Lisinopril)",
    )
    dosage_or_regimen: str = Field(
        ...,
        description="Prescribed dosage or administration regimen",
    )
    clinical_rationale: str = Field(
        ...,
        description="Evidence-grounded medical justification for the selection",
    )
    guideline_reference: str = Field(
        ...,
        description="Clinical guideline supporting this recommendation (e.g., KDIGO, ADA, ACC/AHA)",
    )


class LLMClinicalAgent:
    """Manages prompting and JSON validation for the local generative model."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))

    def _build_prompt(
        self,
        patient_id: str,
        conditions: List[str],
        notes: str,
        guidelines_evidence: str,
        axiom_feedback: List[str],
    ) -> str:
        """Construct a system prompt enforcing schema output and incorporating feedback."""
        feedback_block = ""
        if axiom_feedback:
            feedback_block = "\n".join([f"CRITICAL SAFETY WARNING: {fb}" for fb in axiom_feedback])
            feedback_block = (
                f"\n--- PREVIOUS SAFETY VIOLATIONS (CORRECTION REQUIRED) ---\n"
                f"{feedback_block}\n"
                f"You MUST NOT prescribe any contraindicated drugs listed above!\n"
            )

        prompt = f"""You are a clinical decision support specialist focusing on cardiometabolic multimorbidity.
Formulate a safe pharmacological treatment proposal for the following patient.

PATIENT PROFILE:
Patient ID: {patient_id}
Diagnosed Conditions: {', '.join(conditions)}
Clinical Summary: {notes}

CLINICAL GUIDELINE EVIDENCE:
{guidelines_evidence}
{feedback_block}
INSTRUCTIONS:
1. Recommend an evidence-grounded medication safe for the patient's conditions.
2. If the patient has advanced Chronic Kidney Disease (CKD Stages 4-5), strictly avoid contraindicated drugs (Metformin, Lisinopril).
3. Return ONLY a valid JSON object matching this schema:
{{
  "recommended_drug": "<Drug Name: Insulin, Amlodipine, Metformin, or Lisinopril>",
  "dosage_or_regimen": "<Dosage description>",
  "clinical_rationale": "<Clinical reasoning>",
  "guideline_reference": "<Cited guideline, e.g. KDIGO 2023 or ADA 2024>"
}}
"""
        return prompt

    def generate(
        self,
        patient_id: str,
        conditions: List[str],
        notes: str,
        guidelines_evidence: str,
        axiom_feedback: Optional[List[str]] = None,
    ) -> PrescriptionHypothesis:
        """Query local Ollama instance and parse strictly against PrescriptionHypothesis schema."""
        axiom_feedback = axiom_feedback or []
        prompt = self._build_prompt(patient_id, conditions, notes, guidelines_evidence, axiom_feedback)

        try:
            import requests

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                },
            }

            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=45)
            if resp.status_code == 200:
                raw_json = resp.json().get("response", "{}")
                data = json.loads(raw_json)
                return PrescriptionHypothesis.model_validate(data)
            else:
                logger.warning("Ollama returned status code %s; using deterministic baseline.", resp.status_code)
                return self._fallback_generation(conditions, axiom_feedback)

        except Exception as e:
            logger.warning("Local Ollama connection failed (%s); using deterministic guideline fallback.", e)
            return self._fallback_generation(conditions, axiom_feedback)

    def _fallback_generation(
        self,
        conditions: List[str],
        axiom_feedback: List[str],
    ) -> PrescriptionHypothesis:
        """Deterministic guideline-driven generator for local offline environments or testing."""
        has_ckd = any("ckd" in c.lower() for c in conditions)
        has_t2dm = any("t2dm" in c.lower() or "diabetes" in c.lower() for c in conditions)
        has_htn = any("hypertension" in c.lower() or "htn" in c.lower() for c in conditions)

        has_metformin_violation = any("metformin" in fb.lower() for fb in axiom_feedback)
        has_acei_violation = any("lisinopril" in fb.lower() or "ace" in fb.lower() for fb in axiom_feedback)

        # Baseline without feedback might show first-line bias
        if not axiom_feedback:
            # If standard unguided baseline
            if has_t2dm and not has_ckd:
                return PrescriptionHypothesis(
                    recommended_drug="Metformin",
                    dosage_or_regimen="500mg BID orally",
                    clinical_rationale="First-line oral antihyperglycemic monotherapy for uncomplicated Type 2 Diabetes.",
                    guideline_reference="ADA 2024 Standards of Care",
                )
            if has_htn and not has_ckd:
                return PrescriptionHypothesis(
                    recommended_drug="Lisinopril",
                    dosage_or_regimen="10mg once daily",
                    clinical_rationale="Standard first-line ACE inhibitor for blood pressure reduction.",
                    guideline_reference="ACC/AHA 2017 Hypertension Guidelines",
                )
            if has_ckd and has_t2dm:
                return PrescriptionHypothesis(
                    recommended_drug="Insulin",
                    dosage_or_regimen="Basal insulin glargine 10 units daily",
                    clinical_rationale="Universal safe glycemic management in severe renal impairment (eGFR < 30).",
                    guideline_reference="KDIGO 2023 / ADA 2024",
                )
            if has_ckd and has_htn:
                return PrescriptionHypothesis(
                    recommended_drug="Amlodipine",
                    dosage_or_regimen="5mg once daily",
                    clinical_rationale="Dihydropyridine calcium channel blocker safe in severe renal decline without hyperkalaemia risk.",
                    guideline_reference="KDIGO 2023 Blood Pressure Management",
                )

        # Self-correction handling after feedback
        if has_metformin_violation or (has_ckd and has_t2dm):
            return PrescriptionHypothesis(
                recommended_drug="Insulin",
                dosage_or_regimen="Subcutaneous basal insulin regimen titrated to glycemic target",
                clinical_rationale="Self-corrected safe glycemic agent avoiding lactic acidosis risk in renal failure.",
                guideline_reference="ADA 2024 Antihyperglycaemic Therapy in Renal Impairment",
            )

        if has_acei_violation or (has_ckd and has_htn):
            return PrescriptionHypothesis(
                recommended_drug="Amlodipine",
                dosage_or_regimen="5mg-10mg orally once daily",
                clinical_rationale="Self-corrected calcium channel blocker avoiding RAAS blockade complications in advanced CKD.",
                guideline_reference="KDIGO 2023 Blood Pressure Guidelines",
            )

        # Default safe fallback
        return PrescriptionHypothesis(
            recommended_drug="Insulin",
            dosage_or_regimen="Conservative basal titration",
            clinical_rationale="Conservative metabolic management.",
            guideline_reference="KDIGO 2023",
        )

    def check_health(self) -> dict:
        """Check if local Ollama daemon is reachable and responding."""
        try:
            import requests

            resp = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                has_target = any(self.model_name in m for m in models)
                return {
                    "connected": True,
                    "provider": "Local Ollama Daemon",
                    "base_url": self.base_url,
                    "target_model": self.model_name,
                    "model_available": has_target,
                    "installed_models": models,
                    "mode": "ACTIVE_OLLAMA" if has_target else "LOCAL_DETERMINISTIC_SAFE",
                }
        except Exception:
            pass

        return {
            "connected": False,
            "provider": "Deterministic Guideline Engine (Zero-Latency Local Fallback)",
            "base_url": self.base_url,
            "target_model": self.model_name,
            "model_available": False,
            "installed_models": [],
            "mode": "LOCAL_DETERMINISTIC_SAFE",
        }

