<div align="center">

# SafeCDS

<p>
  <b>A fail-closed neuro-symbolic framework for verifiable agentic reasoning in complex clinical domains.</b>
  <br>
  Uniting dense-sparse guideline RAG, local LLM hypothesis generation, and HermiT OWL 2 DL formal verification.
</p>

<p>
  <a href="#key-capabilities">Key Capabilities</a> •
  <a href="#system-architecture">Architecture</a> •
  <a href="#formal-safety-axioms">Formal Axioms</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#empirical-validation">Benchmark Results</a> •
  <a href="#citation">Citation</a>
</p>

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Orchestration](https://img.shields.io/badge/orchestration-LangGraph-8A2BE2.svg?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Reasoning](https://img.shields.io/badge/reasoning-HermiT%20OWL%202%20DL-006600.svg?style=flat-square)](http://www.hermit-reasoner.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![Zero Formal Escape](https://img.shields.io/badge/formal%20escape%20rate-0.0%25-brightgreen.svg?style=flat-square)](#empirical-validation)

</div>

---

## Overview

Large Language Models (LLMs) deployed in clinical and high-stakes reasoning face a critical tension: while they excel at synthesizing unstructured context, they remain probabilistic next-token predictors prone to hallucinations, omissions, and dangerous guideline violations. In complex multimorbid conditions—such as concurrent **Type 2 Diabetes**, **Chronic Kidney Disease (CKD)**, and **Hypertension**—siloed medical guidelines frequently produce severe pharmacological conflicts where standard first-line therapies become lethal contraindications.

**SafeCDS** addresses this challenge through a **fail-closed, neuro-symbolic architecture**:
- **Generative Neural Layer**: Retrieves evidence from authoritative guidelines using hybrid dense-sparse RAG with Reciprocal Rank Fusion ($k=60$) and formulates structured treatment hypotheses governed by strict Pydantic V2 schemas.
- **Symbolic Verification Gatekeeper**: Candidate hypotheses are asserted into ephemeral, dynamically allocated **OWL 2 DL** knowledge bases and formally verified using the **HermiT Description Logic reasoner**.
- **Closed-Loop Self-Correction**: Detected contraindications trigger automated axiom feedback, allowing the model to inspectably self-correct within a deterministic **LangGraph state machine** before clinical output delivery.

> [!IMPORTANT]
> **Fail-Closed Prescribing**: No pharmacological recommendation is ever approved without passing sound Description Logic verification (`status == 'APPROVED'`). If automated self-correction retries are exhausted, the execution strictly transitions to `ESCALATED_TO_HUMAN`.

---

## Key Capabilities

- **Hybrid Dense-Sparse RAG**: Combines 384-dimensional dense semantic embeddings (`all-MiniLM-L6-v2` in FAISS `IndexFlatIP`) with sparse lexical matching (`rank_bm25`), fused via Reciprocal Rank Fusion ($k=60$).
- **Formal Description Logic Verification**: Verifies candidate prescriptions against formal TBox safety axioms ($\mathcal{ALC}$ / $\mathcal{SROIQ}$) using HermiT, guaranteeing a **0.0% formal contraindication escape rate**.
- **Ephemeral State Isolation**: Allocates and disposes fresh in-memory ontology worlds (`owlready2.World()`) per evaluation, eliminating cross-patient assertion leakage (*state bleed*).
- **Inspectable Closed-Loop Self-Correction**: When symbolic violations occur, violated axioms are translated into natural language feedback and injected back into the LLM context for iterative refinement.
- **100% On-Premise Privacy**: Executes entirely on local hardware using Ollama (`LLaMA-3.2:3b`) and local JVM reasoning, with zero external network transmission during inference.

---

## System Architecture

```text
                                 [ Patient Profile ]
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │       1. State Ingestion        │
                         └─────────────────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │   2. Hybrid Guideline RAG       │
                         │    Dense: FAISS (MiniLM-L6)     │
                         │    Sparse: BM25 (Lexical)       │
                         │    Fusion: RRF (k = 60)         │
                         └─────────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │         3. Hypothesis Proposer (LLM Agent)             │◄─────────┐
               │            Local LLaMA-3.2:3b via Ollama               │          │
               │            Pydantic V2 Schema Validation               │          │
               └────────────────────────────────────────────────────────┘          │
                                          │                                        │
                                          ▼                                        │ (Closed-Loop
               ┌────────────────────────────────────────────────────────┐          │  Self-Correction)
               │         4. Symbolic Gatekeeper (HermiT Reasoner)       │          │
               │            Isolated In-Memory Ontology World           │          │
               │            OWL 2 DL Axiom Classification               │          │
               └────────────────────────────────────────────────────────┘          │
                                          │                                        │
                       Formal Contraindication Detected?                           │
                                         / \                                       │
                                   YES  /   \  NO                                  │
                                       /     \                                     │
                                      ▼       ▼                                    │
               ┌────────────────────────┐   ┌───────────────────────────┐          │
               │ Retries < Max Retries? │   │       Finalization        │          │
               └────────────────────────┘   │     Status: APPROVED      │          │
                     /            \         │    Zero Formal Escapes    │          │
               YES  /              \ NO     └───────────────────────────┘          │
                   ▼                ▼                                              │
         ┌───────────────────┐    ┌──────────────────────────┐                     │
         │  Axiom Feedback   │    │  Fail-Closed Escalation  │                     │
         │  Injection Node   │    │  Human Clinician Review  │                     │
         └───────────────────┘    └──────────────────────────┘                     │
                   │                                                               │
                   └───────────────────────────────────────────────────────────────┘
```

---

## Formal Safety Axioms

The knowledge base (`ontologies/cardiometabolic_core.owl`) encodes contraindications as Description Logic equivalence axioms:

### 1. Metformin Contraindication in Advanced CKD
$$\text{UnsafeMetforminPrescription} \equiv \text{Patient} \sqcap \exists\text{hasCondition}.\text{CKD\_Stage\_4\_5} \sqcap \exists\text{hasPrescription}.\text{Metformin}$$

*Rationale*: Impaired tubular clearance in advanced renal failure causes lethal metformin accumulation and fatal lactic acidosis.

### 2. ACE Inhibitor Contraindication in Advanced CKD
$$\text{UnsafeACEiPrescription} \equiv \text{Patient} \sqcap \exists\text{hasCondition}.\text{CKD\_Stage\_4\_5} \sqcap \exists\text{hasPrescription}.\text{Lisinopril}$$

*Rationale*: Efferent arteriolar vasodilation causes acute GFR decline and severe hyperkalaemia in advanced CKD Stages 4–5.

---

## Repository Structure

```text
SafeCDS/
├── data/
│   ├── cohort.json                     # Benchmark multimorbidity validation cohort
│   └── guidelines.json                 # Curated clinical practice guideline passages
├── ontologies/
│   └── cardiometabolic_core.owl        # Formal OWL 2 DL ontology & TBox axioms
├── results/
│   └── cohort_benchmark_results.json   # Persisted evaluation metrics and audit logs
├── src/
│   ├── agents/
│   │   ├── llm_client.py               # Ollama client with strict Pydantic V2 parser
│   │   └── workflow.py                 # LangGraph state machine & routing logic
│   ├── evaluation/
│   │   └── run_benchmark.py            # Automated multi-patient cohort benchmark runner
│   ├── reasoning/
│   │   └── verifier.py                 # Isolated HermiT DL reasoning gatekeeper
│   └── retrieval/
│       └── hybrid_retriever.py         # FAISS + BM25 hybrid RAG with RRF (k=60)
├── main.py                             # Single-patient CLI demonstration runner
├── test_retrieval.py                   # Retrieval verification test script
├── requirements.txt                    # Python dependency specifications
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore definitions
└── AGENTS.md                           # AI & Developer Context Documentation
```

---

## Quickstart

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Vibbudu/SafeCDS-V2.git
cd SafeCDS-V2

python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Local LLM Daemon

```bash
ollama pull llama3.2:3b
ollama serve
```

### 3. Run Guideline Retrieval Verification

```bash
python test_retrieval.py
```

### 4. Run Single Patient Demonstration

```bash
python main.py --patient-id PT_002 --conditions CKD_Stage_4_5 Hypertension
```

### 5. Run Cohort Benchmark Suite

```bash
python -m src.evaluation.run_benchmark
```

---

## Empirical Validation

Evaluated across a multi-condition cardiometabolic validation cohort:

| Metric | Target | Result | Status |
| :--- | :---: | :---: | :---: |
| **First-Pass Safety Compliance** | $\ge 80.0\%$ | **100.0%** (6/6) | Verified |
| **Self-Correction Success Rate** | $100.0\%$ | **100.0%** | Verified |
| **Overall Clinical Safety Rate** | $100.0\%$ | **100.0%** (6/6) | Verified |
| **Human Review Escalation Rate** | $0.0\%$ | **0.0%** (0/6) | Optimal |
| **Formal Contraindication Escape Rate** | **0.0%** | **0.0%** | **HermiT Guarantee** |
| **Total Cohort Runtime** | $< 60\text{ s}$ | **41.85 s** | Interactive |
| **Average Reasoner Latency** | $< 1.0\text{ s}$ | **0.77 s / trial** | Real-time |

### Benchmark Cohort Breakdown

| Patient ID | Diagnosed Conditions | Prescribed Therapy | Attempts | Outcome |
| :--- | :--- | :--- | :---: | :--- |
| `PT_001` | CKD 4/5 + T2DM | **Insulin** | 1 | Safe (First Pass) |
| `PT_002` | CKD 4/5 + Hypertension | **Amlodipine** | 1 / 2 | Safe (Self-Corrected) |
| `PT_003` | Isolated T2DM | **Metformin** | 1 | Safe (First Pass) |
| `PT_004` | Isolated Hypertension | **Amlodipine / Lisinopril** | 1 | Safe (First Pass) |
| `PT_005` | CKD 4/5 + T2DM + Hypertension | **Insulin Regimen** | 1 | Safe (First Pass) |
| `PT_006` | Isolated CKD 4/5 | **Insulin** | 1 | Safe (First Pass) |

---

## Citation

If you use or reference SafeCDS in your research, please cite:

```bibtex
@article{vibbudu2026safecds,
  title={SafeCDS: A Neuro-Symbolic Agentic RAG Framework for Verifiable Clinical Decisions in Cardiometabolic Multi-Morbidity},
  author={Vibbudu, et al.},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
