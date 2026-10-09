# AGENTS.md — Developer & AI Agent Context Guide

> **Project**: SafeCDS (Version 2)  
> **Repository**: [https://github.com/Vibbudu/SafeCDS-V2.git](https://github.com/Vibbudu/SafeCDS-V2.git)  
> **Domain**: Neuro-Symbolic Artificial Intelligence, Verifiable Agentic RAG, Formal Methods

---

## 1. System Mission & Technical Overview

**SafeCDS** is an on-premise, fail-closed **Neuro-Symbolic Agentic Decision Support Framework** engineered for verifiable clinical diagnosis and safe prescribing in **Cardiometabolic Multimorbidity**.

### The Core Problem
In multimorbid clinical presentations—such as concurrent **Type 2 Diabetes Mellitus (T2DM)**, **Chronic Kidney Disease (CKD Stages 4–5)**, and **Essential Hypertension (HTN)**—disease-specific clinical practice guidelines conflict:
- **Metformin** is first-line for diabetes, but under advanced renal failure (eGFR < 30 mL/min/1.73m²), impaired renal clearance causes fatal lactic acidosis.
- **ACE Inhibitors (Lisinopril)** & ARBs are first-line for hypertension, but in advanced CKD they trigger efferent arteriolar vasodilation, precipitating acute GFR collapse and hyperkalaemia.
- Guidelines from specialized bodies (**ADA**, **KDIGO**, **ACC/AHA**) operate in silos and fail when naive generative agents attempt cross-guideline synthesis.

### The Neuro-Symbolic Solution
- **Neural Layer (Diagnostic & Hypothesis Proposer)**: Local LLM (`LLaMA-3.2:3b` via Ollama) retrieves authoritative guidelines via dense-sparse RAG with Reciprocal Rank Fusion ($k=60$), dynamically infers verified clinical diagnoses, and generates structured medication proposals.
- **Symbolic Layer (Formal Gatekeeper)**: Embedded Description Logic reasoner (**HermiT OWL 2 DL**) verifies candidate drugs against formal clinical safety axioms ($\mathcal{ALC}$ / $\mathcal{SROIQ}$) in ephemeral, isolated memory worlds.
- **Closed-Loop Self-Correction**: When contraindications are detected, violated axioms are translated into natural language feedback and injected back into a **LangGraph state machine** for automated correction (max 3 retries).

---

## 2. Core Architectural Invariants (Non-Negotiable Rules)

When developing or modifying SafeCDS, all contributors and AI assistants must strictly uphold these rules:

1. **Fail-Closed Prescribing**:
   No pharmacological recommendation may be output to a clinician without a formal verification pass from the symbolic reasoner (`status == 'APPROVED'`). If retries exhaust, execution must strictly transition to `ESCALATED_TO_HUMAN`.
2. **Strict Reasoning Memory Isolation (NFR4)**:
   Never reuse an in-memory ontology world across evaluations. Each evaluation must instantiate an ephemeral in-memory ontology world (`owlready2.World()`), assert facts, execute reasoning, extract inferences, and dispose/close the world to prevent cross-evaluation assertion leakage (*state bleed*).
3. **100% On-Premise Privacy (NFR5)**:
   Patient and clinical data must never leave the local environment. LLM inference runs via local Ollama (`localhost:11434`), and OWL reasoning runs locally via JVM HermiT. Zero external network API calls during inference.
4. **Strict Schema Contracts (NFR6)**:
   All generative outputs must strictly adhere to the `ClinicalHypothesis` Pydantic V2 schema (`diagnosis`, `clinical_rationale`, `proposed_treatment`). Unstructured free prose is disallowed for candidate proposals.
5. **Zero Formal Escape Rate (NFR1)**:
   The formal escape rate must remain **0.0%**. The HermiT reasoner provides sound mathematical guarantees for encoded contraindications.
6. **No Mocking or Hardcoding**:
   Do NOT mock or hardcode test cases to return static strings. All responses must be dynamically derived through retrieval, neural hypothesis generation, and formal symbolic verification.

---

## 3. Repository Structure & Module Conventions

```text
SafeCDS/
├── data/
│   ├── cohort.json                    # Benchmark validation patient cohort (with vitals/labs)
│   └── guidelines.json                # Curated passages from KDIGO, ADA, ACC/AHA
├── ontologies/
│   └── cardiometabolic_core.owl       # Formal OWL 2 DL TBox axioms & classes
├── results/
│   └── cohort_benchmark_results.json  # Persisted audit trails & benchmark metrics
├── src/
│   ├── api/
│   │   ├── app.py                     # FastAPI server for local processing & REST endpoints
│   │   └── static/                    # Presentation UI (Inter & Instrument Serif, light mode)
│   ├── agents/
│   │   ├── llm_client.py              # Ollama / Dynamic inference with ClinicalHypothesis
│   │   └── workflow.py                # LangGraph state machine & routing logic
│   ├── evaluation/
│   │   └── run_benchmark.py           # Automated cohort benchmark runner
│   ├── reasoning/
│   │   └── verifier.py                # HermiT DL reasoner in isolated worlds
│   └── retrieval/
│       └── hybrid_retriever.py        # FAISS dense + BM25 sparse + RRF (k=60)
├── main_web.py                        # Local interactive presentation web runner (Working Model)
├── main.py                            # CLI single-patient runner
├── test_retrieval.py                  # Unit verification for hybrid retrieval
├── tests/
│   └── test_api.py                    # Test suite for FastAPI endpoints & pipeline
├── requirements.txt                   # Dependency specifications
├── .env.example                       # Environment configuration template
├── .gitignore                         # Python/IDE/FAISS/Owlready2 ignores
├── LICENSE                            # MIT License
├── AGENTS.md                          # AI & developer context documentation
└── README.md                          # Public repository documentation
```

---

## 4. Formal Safety Axioms (Symbolic Knowledge Base)

The ontology located at `ontologies/cardiometabolic_core.owl` formalizes safety contraindications in Description Logic ($\mathcal{ALC}$ / $\mathcal{SROIQ}$):

### 1. Metformin Contraindication in Advanced CKD
$$\text{UnsafeMetforminPrescription} \equiv \text{Patient} \sqcap \exists\text{hasCondition}.\text{CKD\_Stage\_4\_5} \sqcap \exists\text{hasPrescription}.\text{Metformin}$$
*Rationale*: High risk of fatal lactic acidosis due to impaired tubular secretion.

### 2. ACE-Inhibitor Contraindication in Advanced CKD
$$\text{UnsafeACEiPrescription} \equiv \text{Patient} \sqcap \exists\text{hasCondition}.\text{CKD\_Stage\_4\_5} \sqcap \exists\text{hasPrescription}.\text{Lisinopril}$$
*Rationale*: Efferent arteriolar vasodilation causes acute GFR reduction and dangerous hyperkalaemia.

---

## 5. LangGraph State Machine Specification

The state is represented by `WorkflowState`:
```python
class WorkflowState(TypedDict):
    patient_id: str
    conditions: List[str]
    clinical_notes: str
    vitals: Optional[Dict[str, Any]]
    retrieved_evidence: str
    retrieved_docs: List[Dict[str, Any]]
    inferred_diagnosis: List[str]
    proposed_treatment: Optional[str]
    proposed_drug: Optional[str]
    hypothesis: Optional[Dict[str, Any]]
    verification_result: Optional[Dict[str, Any]]
    axiom_feedback: List[str]
    attempts_history: List[Dict[str, Any]]
    retries: int
    max_retries: int
    status: str       # "PENDING" | "APPROVED" | "ESCALATED_TO_HUMAN"
    audit_trail: List[Dict[str, Any]]
```

### State Machine Transitions:
```text
[Start]
   │
   ▼
[node_ingest] (FR1)
   │
   ▼
[node_retrieve] (FR2: FAISS + BM25 + RRF)
   │
   ▼
[node_generate] (FR3: Ollama / Dynamic Proposer + Pydantic) ◄──────┐
   │                                                               │ (Retry Loop, max 3)
   ▼                                                               │
[node_verify] (FR4: HermiT DL Reasoner in isolated world)          │
   │                                                               │
   ├───────────► If FAIL & retries < max: [node_feedback] (FR5) ───┘
   │
   ├───────────► If PASS: [node_finalize] (FR7: APPROVED) ──► [END]
   │
   └───────────► If FAIL & retries >= max: [node_escalate] (FR6: ESCALATED) ──► [END]
```

---

## 6. Guidelines for Collaborating Developers & AI Agents

1. **Keep Fallbacks Functional & Dynamic**:
   Dynamic reasoning handles offline environments gracefully without static mocks or hardcoded patient IDs.
2. **Windows Console Encoding**:
   When printing status symbols to stdout, avoid Unicode characters that trigger `charmap` codec errors on Windows consoles. Use ASCII tags like `[OK]`, `[PASS]`, `[FAIL]`, `[INFO]`.
3. **Commit Clean Code**:
   - Run `python test_retrieval.py`
   - Run `python tests/test_api.py`
   - Run `python -m src.evaluation.run_benchmark`
   - Run `python main.py`
   Ensure all commands exit with return code `0` before committing changes.
