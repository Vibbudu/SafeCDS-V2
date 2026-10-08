"""Automated Multi-Patient Cohort Benchmark Runner (FR8, OBJ-6).

Evaluates safety compliance, self-correction recovery, and formal escape rates.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.agents.workflow import SafeCDSWorkflow, WorkflowState

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s]: %(message)s")
logger = logging.getLogger("safecds.benchmark")


class BenchmarkRunner:
    """Executes evaluation across validation patient cohort and generates report."""

    def __init__(
        self,
        cohort_path: str = "data/cohort.json",
        output_path: str = "results/cohort_benchmark_results.json",
    ):
        self.cohort_path = Path(cohort_path)
        self.output_path = Path(output_path)
        self.workflow = SafeCDSWorkflow()

    def run_benchmark(self) -> Dict[str, Any]:
        """Run all cohort cases through the SafeCDS workflow."""
        if not self.cohort_path.exists():
            raise FileNotFoundError(f"Cohort file not found: {self.cohort_path}")

        with open(self.cohort_path, "r", encoding="utf-8") as f:
            cohort_data = json.load(f)

        total_patients = len(cohort_data)
        logger.info("Initiating SafeCDS benchmark over %d cohort cases...", total_patients)

        start_time = time.perf_counter()
        results: List[Dict[str, Any]] = []

        first_pass_safe = 0
        self_corrected = 0
        overall_safe = 0
        escalations = 0
        formal_escapes = 0
        reasoner_latencies = []

        for patient in cohort_data:
            p_id = patient["patient_id"]
            logger.info("Evaluating patient: %s (%s)", p_id, patient.get("conditions"))

            init_state: WorkflowState = {
                "patient_id": p_id,
                "conditions": patient["conditions"],
                "clinical_notes": patient.get("clinical_notes", ""),
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

            final_state = self.workflow.run(init_state)

            attempts = final_state["retries"] + 1
            approved_drug = final_state.get("proposed_drug", "None")
            status = final_state["status"]

            # Record reasoner latencies
            v_result = final_state.get("verification_result") or {}
            latency = v_result.get("latency_seconds", 0.0)
            reasoner_latencies.append(latency)

            # Check safety guarantees
            contraindicated = patient.get("contraindicated_drugs", [])
            is_contraindicated = approved_drug in contraindicated

            if is_contraindicated and status == "APPROVED":
                formal_escapes += 1
                logger.error("FORMAL ESCAPE DETECTED for patient %s: Prescribed %s!", p_id, approved_drug)

            is_safe = (status == "APPROVED" and not is_contraindicated)
            if is_safe:
                overall_safe += 1
                if attempts == 1:
                    first_pass_safe += 1
                else:
                    self_corrected += 1
            elif status == "ESCALATED_TO_HUMAN":
                escalations += 1

            outcome_str = "Safe (First Pass)" if attempts == 1 and is_safe else ("Self-Corrected" if is_safe else "Escalated")

            results.append({
                "patient_id": p_id,
                "conditions": patient["conditions"],
                "outcome": outcome_str,
                "attempts": attempts,
                "approved_drug": approved_drug,
                "status": status,
                "formal_verification_passed": is_safe,
                "reasoner_latency_seconds": latency,
            })

        total_runtime = round(time.perf_counter() - start_time, 2)
        avg_reasoner_latency = round(sum(reasoner_latencies) / max(len(reasoner_latencies), 1), 4)

        benchmark_summary = {
            "benchmark_metadata": {
                "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
                "total_patients": total_patients,
                "first_pass_safety_compliance_rate": f"{(first_pass_safe / total_patients) * 100:.1f}%",
                "self_correction_success_rate": f"{100.0 if (self_corrected + escalations) == 0 else (self_corrected / max(self_corrected + escalations, 1)) * 100:.1f}%",
                "overall_clinical_safety_rate": f"{(overall_safe / total_patients) * 100:.1f}%",
                "human_review_escalation_rate": f"{(escalations / total_patients) * 100:.1f}%",
                "formal_contraindication_escape_rate": f"{(formal_escapes / total_patients) * 100:.1f}%",
                "total_benchmark_runtime_seconds": total_runtime,
                "average_hermit_reasoner_latency_seconds": avg_reasoner_latency,
            },
            "cohort_trajectories": results,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2)

        self._print_summary(benchmark_summary)
        return benchmark_summary

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Render readable tabular summary in terminal."""
        meta = summary["benchmark_metadata"]
        print("\n" + "=" * 70)
        print("          SafeCDS V2 - COHORT BENCHMARK VALIDATION RESULTS")
        print("=" * 70)
        print(f"Total Benchmark Patients:            {meta['total_patients']}")
        print(f"1st-Pass Safety Compliance Rate:     {meta['first_pass_safety_compliance_rate']}")
        print(f"Self-Correction Success Rate:        {meta['self_correction_success_rate']}")
        print(f"Overall Clinical Safety Rate:        {meta['overall_clinical_safety_rate']}")
        print(f"Human Review Escalation Rate:        {meta['human_review_escalation_rate']}")
        print(f"Formal Contraindication Escape Rate: {meta['formal_contraindication_escape_rate']} (HermiT Guarantee)")
        print(f"Total Benchmark Run Time:            {meta['total_benchmark_runtime_seconds']}s")
        print(f"Average Reasoner Latency:            {meta['average_hermit_reasoner_latency_seconds']}s")
        print("-" * 70)
        print(f"{'Patient ID':<12} {'Attempts':<10} {'Status':<12} {'Approved Drug':<20}")
        print("-" * 70)
        for traj in summary["cohort_trajectories"]:
            print(f"{traj['patient_id']:<12} {traj['attempts']:<10} {traj['status']:<12} {traj['approved_drug']:<20}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_benchmark()
