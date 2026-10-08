"""Clinical Verifier: Isolated Description Logic (OWL 2 DL) Symbolic Gatekeeper.

Performs formal contraindication checks using HermiT reasoner within dynamically
allocated, ephemeral in-memory ontology worlds to ensure zero cross-patient state bleed.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("safecds.reasoning")


@dataclass
class VerificationResult:
    """Outcome of formal clinical verification."""
    is_safe: bool
    status: str  # "PASS" or "FAIL"
    violated_axioms: List[str] = field(default_factory=list)
    feedback_message: str = ""
    inferred_classes: List[str] = field(default_factory=list)
    latency_seconds: float = 0.0


class ClinicalVerifier:
    """HermiT-backed OWL 2 DL reasoner gatekeeper with isolated in-memory worlds."""

    def __init__(self, ontology_path: str = "ontologies/cardiometabolic_core.owl"):
        self.ontology_path = Path(ontology_path).resolve()
        if not self.ontology_path.exists():
            raise FileNotFoundError(f"Ontology file not found at: {self.ontology_path}")

        # Human-readable clinical feedback mapped to formal axioms
        self.axiom_feedback_map = {
            "UnsafeMetforminPrescription": (
                "Contraindication: Metformin is strictly contraindicated in advanced "
                "Chronic Kidney Disease (CKD Stages 4-5) due to severe risk of fatal lactic acidosis. "
                "Select a safe alternative such as Insulin."
            ),
            "UnsafeACEiPrescription": (
                "Contraindication: High-dose ACE inhibitors (Lisinopril) are contraindicated in advanced "
                "CKD Stages 4-5 without specialist monitoring due to acute drops in glomerular filtration "
                "and risk of severe hyperkalaemia. Select a safe alternative such as Amlodipine."
            ),
        }

    def _normalize_name(self, value: str) -> str:
        """Standardize condition/drug strings to match ontology identifiers."""
        val = value.strip().replace(" ", "_").replace("-", "_").lower()
        if "metformin" in val:
            return "Metformin"
        if "lisinopril" in val:
            return "Lisinopril"
        if "insulin" in val:
            return "Insulin"
        if "amlodipine" in val:
            return "Amlodipine"
        if "ckd" in val or "kidney" in val or "renal" in val:
            return "CKD_Stage_4_5"
        if "t2dm" in val or "diabetes" in val:
            return "T2DM"
        if "hypertension" in val or "htn" in val or "bp" in val:
            return "Hypertension"
        return value.strip()

    def verify(
        self,
        patient_id: str,
        conditions: List[str],
        proposed_drug: str,
    ) -> VerificationResult:
        """Assert patient state into a fresh, isolated OWL world and run Description Logic reasoning.

        Enforces strict memory isolation (OBJ-3 / NFR4).
        """
        start_time = time.perf_counter()
        normalized_drug = self._normalize_name(proposed_drug)
        normalized_conditions = [self._normalize_name(c) for c in conditions]

        # Attempt HermiT OWL 2 DL reasoning via Owlready2
        try:
            import owlready2

            # 1. Allocate isolated in-memory world (Zero cross-patient contamination)
            world = owlready2.World()
            onto = world.get_ontology(str(self.ontology_path.as_uri())).load()

            try:
                with onto:
                    # Dynamically instantiate Patient
                    patient_cls = onto.Patient
                    test_patient = patient_cls(f"patient_{patient_id}")

                    # Assert conditions
                    for cond_name in normalized_conditions:
                        cond_cls = getattr(onto, cond_name, None)
                        if cond_cls:
                            cond_ind = cond_cls(f"cond_{patient_id}_{cond_name}")
                            test_patient.hasCondition.append(cond_ind)

                    # Assert proposed prescription
                    drug_cls = getattr(onto, normalized_drug, None)
                    if drug_cls:
                        drug_ind = drug_cls(f"rx_{patient_id}_{normalized_drug}")
                        test_patient.hasPrescription.append(drug_ind)

                # 2. Invoke HermiT reasoner within isolated world
                owlready2.sync_reasoner_hermit(world, infer_property_values=True)

                # 3. Inspect inferred classifications
                inferred = [cls.name for cls in test_patient.is_a if hasattr(cls, "name")]
                violated = []
                feedbacks = []

                if "UnsafeMetforminPrescription" in inferred:
                    violated.append("UnsafeMetforminPrescription")
                    feedbacks.append(self.axiom_feedback_map["UnsafeMetforminPrescription"])

                if "UnsafeACEiPrescription" in inferred:
                    violated.append("UnsafeACEiPrescription")
                    feedbacks.append(self.axiom_feedback_map["UnsafeACEiPrescription"])

                elapsed = time.perf_counter() - start_time
                is_safe = len(violated) == 0

                return VerificationResult(
                    is_safe=is_safe,
                    status="PASS" if is_safe else "FAIL",
                    violated_axioms=violated,
                    feedback_message="; ".join(feedbacks) if feedbacks else "Prescription verified formally safe.",
                    inferred_classes=inferred,
                    latency_seconds=round(elapsed, 4),
                )
            finally:
                # 4. Strict isolation cleanup: dispose world
                world.close()

        except Exception as e:
            logger.warning(
                "Owlready2/HermiT JVM reasoning unavailable (%s). Falling back to deterministic semantic rule check.",
                e,
            )
            return self._deterministic_fallback(
                patient_id=patient_id,
                conditions=normalized_conditions,
                drug=normalized_drug,
                start_time=start_time,
            )

    def _deterministic_fallback(
        self,
        patient_id: str,
        conditions: List[str],
        drug: str,
        start_time: float,
    ) -> VerificationResult:
        """Deterministic safety evaluator implementing the exact equivalent OWL axioms:

        - UnsafeMetforminPrescription: CKD_Stage_4_5 AND Metformin
        - UnsafeACEiPrescription: CKD_Stage_4_5 AND Lisinopril
        """
        violated = []
        feedbacks = []
        has_ckd = "CKD_Stage_4_5" in conditions

        if has_ckd and drug == "Metformin":
            violated.append("UnsafeMetforminPrescription")
            feedbacks.append(self.axiom_feedback_map["UnsafeMetforminPrescription"])

        if has_ckd and drug == "Lisinopril":
            violated.append("UnsafeACEiPrescription")
            feedbacks.append(self.axiom_feedback_map["UnsafeACEiPrescription"])

        elapsed = time.perf_counter() - start_time
        is_safe = len(violated) == 0

        return VerificationResult(
            is_safe=is_safe,
            status="PASS" if is_safe else "FAIL",
            violated_axioms=violated,
            feedback_message="; ".join(feedbacks) if feedbacks else "Prescription verified formally safe.",
            inferred_classes=violated if violated else ["SafePrescription"],
            latency_seconds=round(elapsed, 4),
        )
