import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_system_status():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OPERATIONAL"
    assert "reasoner" in data
    assert "llm" in data
    assert "retrieval" in data
    print("[OK] /api/status test passed.")


def test_patients_list():
    res = client.get("/api/patients")
    assert res.status_code == 200
    patients = res.json()
    assert len(patients) == 6
    ids = [p["patient_id"] for p in patients]
    assert "PT_001" in ids and "PT_002" in ids
    print(f"[OK] /api/patients test passed ({len(patients)} cases).")


def test_patient_evaluation_safe():
    payload = {
        "patient_id": "PT_002",
        "conditions": ["CKD_Stage_4_5", "Hypertension"],
        "clinical_notes": "Patient with Stage 5 CKD and severe hypertension.",
        "vitals": {"egfr": 14, "sbp": 165, "hba1c": 8.5},
        "max_retries": 3,
    }
    res = client.post("/api/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "APPROVED"
    assert data["approved_drug"] in ["Amlodipine", "Insulin"]
    assert "inferred_diagnosis" in data
    assert len(data["inferred_diagnosis"]) > 0
    assert "proposed_treatment" in data
    assert "safety_certificate" in data
    assert data["safety_certificate"]["formal_escape_rate"] == 0.0
    print(f"[OK] /api/evaluate PT_002 test passed (Approved: {data['approved_drug']}, Diagnosis: {data['inferred_diagnosis']}).")


def test_ontology_endpoint():
    res = client.get("/api/ontology")
    assert res.status_code == 200
    data = res.json()
    assert "axioms" in data
    assert len(data["axioms"]) >= 2
    names = [a["name"] for a in data["axioms"]]
    assert "UnsafeMetforminPrescription" in names
    assert "UnsafeACEiPrescription" in names
    print(f"[OK] /api/ontology test passed ({len(data['axioms'])} axioms).")


def test_guidelines_endpoint():
    res = client.get("/api/guidelines")
    assert res.status_code == 200
    guidelines = res.json()
    assert len(guidelines) >= 4
    print(f"[OK] /api/guidelines test passed ({len(guidelines)} guidelines).")


def test_root_serves_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "SafeCDS" in res.text
    assert "Instrument Serif" in res.text
    print(f"[OK] GET / index.html test passed ({len(res.text)} bytes).")


def test_benchmark_results_endpoint():
    res = client.get("/api/benchmark-results")
    assert res.status_code == 200
    data = res.json()
    assert "benchmark_metadata" in data
    assert "cohort_trajectories" in data
    assert data["benchmark_metadata"]["formal_contraindication_escape_rate"] == "0.0%"
    print(f"[OK] /api/benchmark-results test passed (Escape Rate: {data['benchmark_metadata']['formal_contraindication_escape_rate']}).")


if __name__ == "__main__":
    test_system_status()
    test_patients_list()
    test_patient_evaluation_safe()
    test_ontology_endpoint()
    test_guidelines_endpoint()
    test_benchmark_results_endpoint()
    test_root_serves_html()
    print("\n[OK] ALL API TESTS PASSED SUCCESSFULLY!")
