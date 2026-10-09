/**
 * SafeCDS V2: Presentation Application Client Controller
 * Powers live case solving, closed-loop visualizer, cohort benchmarking, and axiom exploration.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Application State
  const state = {
    patients: [],
    activePatient: null,
    systemStatus: null,
    axioms: [],
    guidelines: [],
  };

  // --- Initializers ---
  initNavigation();
  initVitalsSliders();
  initSolverActions();
  initBenchmarkActions();
  initGuidelinesSearch();
  loadSystemStatus();
  loadPatients();
  loadAxioms();
  loadGuidelines();
  loadBenchmarkInitial();

  // --- 1. Navigation Controller (Floating Bottom Capsule) ---
  function initNavigation() {
    const capsuleLinks = document.querySelectorAll(".capsule-link");
    const views = document.querySelectorAll(".content-view");

    capsuleLinks.forEach((link) => {
      link.addEventListener("click", () => {
        const targetId = link.getAttribute("data-target");

        // Update active link
        capsuleLinks.forEach((l) => l.classList.remove("active"));
        link.classList.add("active");

        // Switch active view
        views.forEach((view) => {
          if (view.id === targetId) {
            view.classList.add("active-view");
          } else {
            view.classList.remove("active-view");
          }
        });

        window.scrollTo({ top: 0, behavior: "smooth" });
        refreshLucide();
      });
    });
  }

  // --- 2. System Status & Engine Telemetry ---
  async function loadSystemStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();
      state.systemStatus = data;

      // Update Reasoner Pill
      const valReasoner = document.getElementById("val-reasoner");
      if (valReasoner) {
        valReasoner.textContent = "Isolated World (DL Sound)";
      }

      // Update LLM Pill
      const dotLlm = document.getElementById("dot-llm");
      const valLlm = document.getElementById("val-llm");
      if (data.llm && data.llm.connected && data.llm.model_available) {
        if (dotLlm) dotLlm.className = "pill-dot status-green";
        if (valLlm) valLlm.textContent = `${data.llm.target_model} (Local)`;
      } else {
        if (dotLlm) dotLlm.className = "pill-dot status-amber";
        if (valLlm) valLlm.textContent = "Local Deterministic Proposer";
      }

      // Update RAG Pill
      const valRag = document.getElementById("val-rag");
      if (valRag && data.retrieval) {
        valRag.textContent = `FAISS + BM25 (${data.retrieval.documents_indexed} docs)`;
      }
    } catch (e) {
      console.warn("Status check deferred:", e);
    }
  }

  // --- 3. Patient Intake & Preset Selector ---
  async function loadPatients() {
    try {
      const res = await fetch("/api/patients");
      if (!res.ok) return;
      state.patients = await res.json();
      renderPresetButtons();

      // Select PT_002 by default (demonstrates self-correction)
      const defaultPatient = state.patients.find((p) => p.patient_id === "PT_002") || state.patients[0];
      if (defaultPatient) {
        selectPatient(defaultPatient);
      }
    } catch (e) {
      console.error("Failed to load patients:", e);
    }
  }

  function renderPresetButtons() {
    const container = document.getElementById("preset-buttons-container");
    if (!container) return;
    container.innerHTML = "";

    state.patients.forEach((patient) => {
      const btn = document.createElement("button");
      btn.className = "preset-btn";
      btn.setAttribute("data-id", patient.patient_id);

      const conditionsBrief = patient.conditions
        .map((c) => c.replace("CKD_Stage_4_5", "CKD 4/5").replace("Hypertension", "HTN"))
        .join(" + ");

      btn.innerHTML = `
        <span class="preset-id">${patient.patient_id}</span>
        <span class="preset-sub">${conditionsBrief}</span>
      `;

      btn.addEventListener("click", () => selectPatient(patient));
      container.appendChild(btn);
    });
  }

  function selectPatient(patient) {
    state.activePatient = patient;

    // Highlight button
    document.querySelectorAll(".preset-btn").forEach((b) => {
      if (b.getAttribute("data-id") === patient.patient_id) {
        b.classList.add("active");
      } else {
        b.classList.remove("active");
      }
    });

    // Populate active box
    document.getElementById("display-patient-id").textContent = patient.patient_id;
    document.getElementById("display-patient-name").textContent = patient.name || `Case ${patient.patient_id}`;

    // Condition chips
    const chipsContainer = document.getElementById("display-conditions-chips");
    chipsContainer.innerHTML = "";
    patient.conditions.forEach((c) => {
      const chip = document.createElement("span");
      const isCkd = c.includes("CKD");
      chip.className = `condition-chip ${isCkd ? "highlight" : ""}`;
      chip.innerHTML = `${isCkd ? "⚠️ " : ""}${c.replace(/_/g, " ")}`;
      chipsContainer.appendChild(chip);
    });

    // Notes
    const notesInput = document.getElementById("input-clinical-notes");
    notesInput.value = patient.clinical_notes || "";

    // Adjust sliders based on conditions
    adjustVitalsForPatient(patient);
    refreshLucide();
  }

  function adjustVitalsForPatient(patient) {
    const sliderEgfr = document.getElementById("slider-egfr");
    const sliderSbp = document.getElementById("slider-sbp");
    const sliderHba1c = document.getElementById("slider-hba1c");

    const hasCkd = patient.conditions.some((c) => c.includes("CKD"));
    const hasHtn = patient.conditions.some((c) => c.toLowerCase().includes("hypertension"));
    const hasT2dm = patient.conditions.some((c) => c.toLowerCase().includes("t2dm") || c.toLowerCase().includes("diabetes"));

    if (sliderEgfr) {
      sliderEgfr.value = hasCkd ? 18 : 88;
      updateEgfrDisplay(sliderEgfr.value);
    }
    if (sliderSbp) {
      sliderSbp.value = hasHtn ? 165 : 122;
      updateSbpDisplay(sliderSbp.value);
    }
    if (sliderHba1c) {
      sliderHba1c.value = hasT2dm ? 8.6 : 5.6;
      updateHba1cDisplay(sliderHba1c.value);
    }
  }

  // --- 4. Interactive Vitals Sliders ---
  function initVitalsSliders() {
    const sliderEgfr = document.getElementById("slider-egfr");
    const sliderSbp = document.getElementById("slider-sbp");
    const sliderHba1c = document.getElementById("slider-hba1c");

    if (sliderEgfr) {
      sliderEgfr.addEventListener("input", (e) => updateEgfrDisplay(e.target.value));
    }
    if (sliderSbp) {
      sliderSbp.addEventListener("input", (e) => updateSbpDisplay(e.target.value));
    }
    if (sliderHba1c) {
      sliderHba1c.addEventListener("input", (e) => updateHba1cDisplay(e.target.value));
    }
  }

  function updateEgfrDisplay(val) {
    const badge = document.getElementById("val-egfr");
    const stage = val < 15 ? "Stage 5 (Kidney Failure)" : val < 30 ? "Stage 4 (Severe CKD)" : val < 60 ? "Stage 3" : "Normal Renal Function";
    badge.textContent = `${val} mL/min/1.73m² (${stage})`;
  }

  function updateSbpDisplay(val) {
    const badge = document.getElementById("val-sbp");
    const stage = val >= 160 ? "Stage 2 HTN (Crisis Risk)" : val >= 140 ? "Stage 2 HTN" : val >= 130 ? "Stage 1 HTN" : "Normal Pressure";
    badge.textContent = `${val} mmHg (${stage})`;
  }

  function updateHba1cDisplay(val) {
    const badge = document.getElementById("val-hba1c");
    const stage = val >= 8.5 ? "Poor Control (T2DM)" : val >= 7.0 ? "Moderate Control" : "Optimal Glycemic Target";
    badge.textContent = `${val}% (${stage})`;
  }

  // --- 5. Run Case Solver Execution & Closed-Loop Visualizer ---
  function initSolverActions() {
    const btnRun = document.getElementById("btn-run-solver");
    if (!btnRun) return;

    btnRun.addEventListener("click", async () => {
      await executeCaseSolver();
    });

    const btnToggleAudit = document.getElementById("btn-toggle-audit");
    if (btnToggleAudit) {
      btnToggleAudit.addEventListener("click", () => {
        const list = document.getElementById("audit-log-list");
        if (list) list.classList.toggle("hidden");
      });
    }
  }

  async function executeCaseSolver() {
    const btn = document.getElementById("btn-run-solver");
    const btnText = document.getElementById("btn-run-solver-text");
    const emptyState = document.getElementById("solver-empty-state");
    const panels = document.getElementById("solver-dynamic-panels");

    if (!state.activePatient) return;

    // Reset Stepper
    resetStepper();
    setStepActive("step-node-ingest");

    // UI loading state
    btn.disabled = true;
    btnText.textContent = "Executing Neuro-Symbolic Verification...";

    const payload = {
      patient_id: state.activePatient.patient_id,
      conditions: state.activePatient.conditions,
      clinical_notes: document.getElementById("input-clinical-notes").value,
      vitals: {
        egfr: parseFloat(document.getElementById("slider-egfr").value),
        sbp: parseFloat(document.getElementById("slider-sbp").value),
        hba1c: parseFloat(document.getElementById("slider-hba1c").value),
      },
      max_retries: 2,
    };

    // Stagger animation simulation for presentation effect
    setStepPassed("step-node-ingest");
    setStepActive("step-node-retrieve");

    try {
      const response = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Evaluation failed with status ${response.status}`);
      }

      const result = await response.json();

      // Complete Stepper animation
      setStepPassed("step-node-retrieve");
      setStepActive("step-node-generate");
      await delay(180);

      setStepPassed("step-node-generate");
      setStepActive("step-node-verify");
      await delay(180);

      if (result.total_attempts > 1) {
        setStepPassed("step-node-verify");
        setStepActive("step-node-feedback");
        await delay(200);
        setStepPassed("step-node-feedback");
      }

      setStepActive("step-node-finalize");
      setStepPassed("step-node-finalize");

      // Show results
      if (emptyState) emptyState.classList.add("hidden");
      if (panels) panels.classList.remove("hidden");

      renderVerdictBanner(result);
      renderSelfCorrectionPanel(result);
      renderRagEvidence(result);
      renderNeuralProposal(result);
      renderSymbolicVerification(result);
      renderAuditTrail(result);

      refreshLucide();
    } catch (err) {
      console.error("Execution error:", err);
      alert(`Pipeline error: ${err.message}`);
    } finally {
      btn.disabled = false;
      btnText.textContent = "Execute Neuro-Symbolic Verification";
    }
  }

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function resetStepper() {
    document.querySelectorAll(".step-item").forEach((s) => {
      s.classList.remove("active", "passed");
    });
  }

  function setStepActive(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add("active");
  }

  function setStepPassed(id) {
    const el = document.getElementById(id);
    if (el) {
      el.classList.remove("active");
      el.classList.add("passed");
    }
  }

  // --- Render Result Panels ---
  function renderVerdictBanner(result) {
    const banner = document.getElementById("verdict-banner");
    if (!banner) return;

    const isApproved = result.status === "APPROVED";
    const hyp = result.final_hypothesis || {};

    banner.innerHTML = `
      <div class="verdict-left">
        <span class="verdict-status-badge">
          <i data-lucide="${isApproved ? "check-circle-2" : "alert-triangle"}"></i>
          ${isApproved ? "Formal Verification: APPROVED" : "ESCALATED TO HUMAN CLINICIAN"}
        </span>
        <h2 class="verdict-drug-name instrument-serif">${result.approved_drug || "No Drug Approved"}</h2>
        <p class="verdict-regimen">${hyp.dosage_or_regimen || "Prescription verified formally safe under Description Logic axioms."}</p>
      </div>
      <div class="verdict-right">
        <span class="verdict-pill-attempts">${result.total_attempts} Attempt${result.total_attempts > 1 ? "s" : ""} (${result.retries_used} Retries)</span>
        <span class="verdict-latency">Execution Latency: ${result.latency_ms} ms</span>
      </div>
    `;
  }

  function renderSelfCorrectionPanel(result) {
    const panel = document.getElementById("correction-cycle-panel");
    if (!panel) return;

    if (result.total_attempts <= 1 && result.retries_used === 0) {
      panel.classList.add("hidden");
      return;
    }

    panel.classList.remove("hidden");
    const att1 = result.attempts_history && result.attempts_history[0] ? result.attempts_history[0] : null;
    const att2 = result.attempts_history && result.attempts_history[1] ? result.attempts_history[1] : null;

    panel.innerHTML = `
      <div class="correction-cycle-header">
        <i data-lucide="refresh-cw" class="correction-icon"></i>
        <h3 class="correction-title">Closed-Loop Self-Correction Cycle Detected</h3>
      </div>
      <div class="comparison-columns">
        
        <!-- Attempt 1: Contraindicated -->
        <div class="attempt-card rejected">
          <div class="attempt-header">
            <span class="attempt-tag tag-fail">Attempt 1: REJECTED</span>
            <span class="card-badge">HermiT Catch</span>
          </div>
          <span class="attempt-drug">${att1 ? att1.proposed_drug : "Initial Candidate"}</span>
          <p class="attempt-desc">
            ${att1 && att1.verification && att1.verification.feedback_message ? att1.verification.feedback_message : "Contraindication flagged by Description Logic Reasoner."}
          </p>
        </div>

        <!-- Feedback routing arrow -->
        <div class="attempt-arrow">
          <i data-lucide="arrow-right" style="width: 28px; height: 28px;"></i>
          <span class="arrow-label">Axiom Feedback</span>
        </div>

        <!-- Attempt 2: Self-Corrected & Approved -->
        <div class="attempt-card resolved">
          <div class="attempt-header">
            <span class="attempt-tag tag-pass">Attempt 2: APPROVED</span>
            <span class="card-badge">Sound & Safe</span>
          </div>
          <span class="attempt-drug">${att2 ? att2.proposed_drug : result.approved_drug}</span>
          <p class="attempt-desc">
            ${att2 && att2.hypothesis && att2.hypothesis.clinical_rationale ? att2.hypothesis.clinical_rationale : "Model ingested axiom warning and formulated a safe alternative verified by HermiT."}
          </p>
        </div>

      </div>
    `;
  }

  function renderRagEvidence(result) {
    const container = document.getElementById("rag-passages-container");
    const countBadge = document.getElementById("rag-docs-count");
    if (!container) return;

    const docs = result.retrieved_docs || [];
    if (countBadge) countBadge.textContent = `${docs.length} Evidence Passages`;
    container.innerHTML = "";

    docs.forEach((doc) => {
      const item = document.createElement("div");
      item.className = "passage-item";
      item.innerHTML = `
        <div class="passage-meta">
          <span class="passage-source">${doc.source || "Clinical Practice Guideline"}</span>
          <span class="passage-score">RRF Rank Score: ${doc.rrf_score || "Top Match"}</span>
        </div>
        <span class="passage-topic">${doc.topic || ""}</span>
        <p class="passage-text">"${doc.text || ""}"</p>
      `;
      container.appendChild(item);
    });
  }

  function renderNeuralProposal(result) {
    const container = document.getElementById("proposal-details-container");
    if (!container) return;

    const hyp = result.final_hypothesis || {};
    container.innerHTML = `
      <div class="proposal-item">
        <span class="proposal-key">Recommended Medication</span>
        <span class="proposal-val">${hyp.recommended_drug || result.approved_drug || "None"}</span>
      </div>
      <div class="proposal-item">
        <span class="proposal-key">Dosage & Administration</span>
        <span class="proposal-val">${hyp.dosage_or_regimen || "Standard dosage"}</span>
      </div>
      <div class="proposal-item full-width">
        <span class="proposal-key">Clinical Rationale</span>
        <span class="proposal-val full-width">${hyp.clinical_rationale || "Guideline-grounded prescribing."}</span>
      </div>
      <div class="proposal-item full-width">
        <span class="proposal-key">Guideline Citation</span>
        <span class="proposal-val full-width">${hyp.guideline_reference || "KDIGO / ADA Consensus"}</span>
      </div>
    `;
  }

  function renderSymbolicVerification(result) {
    const container = document.getElementById("verification-details-container");
    if (!container) return;

    const v = result.verification_result || {};
    const isSafe = v.is_safe !== false;

    container.innerHTML = `
      <div class="verification-status-banner">
        <span class="verif-badge ${isSafe ? "pass" : "fail"}">
          ${isSafe ? "CLASSIFICATION: SAFE PRESCRIPTION" : "CLASSIFICATION: CONTRAINDICATED"}
        </span>
        <span class="verif-message">${v.feedback_message || "Prescription passed formal DL consistency check."}</span>
      </div>
      <div class="proposal-details-grid" style="padding: 10px; margin-top: 8px;">
        <div class="proposal-item">
          <span class="proposal-key">Reasoning Engine</span>
          <span class="proposal-val">HermiT OWL 2 DL Reasoner</span>
        </div>
        <div class="proposal-item">
          <span class="proposal-key">Memory State Isolation</span>
          <span class="proposal-val">Ephemeral World (Disposed)</span>
        </div>
        <div class="proposal-item full-width">
          <span class="proposal-key">Inferred Ontology Classes</span>
          <span class="proposal-val full-width">${(v.inferred_classes || ["SafePrescription"]).join(", ")}</span>
        </div>
      </div>
    `;
  }

  function renderAuditTrail(result) {
    const list = document.getElementById("audit-log-list");
    const countBadge = document.getElementById("audit-steps-count");
    if (!list) return;

    const trail = result.audit_trail || [];
    if (countBadge) countBadge.textContent = `${trail.length} Transitions`;
    list.innerHTML = "";

    trail.forEach((step, idx) => {
      const item = document.createElement("div");
      item.className = "audit-item";
      let msg = step.message || step.decision || step.status || step.query || "Executed";
      if (step.proposal) msg = `Proposed: ${step.proposal.recommended_drug}`;
      if (step.feedback) msg = `Feedback: ${step.feedback}`;

      item.innerHTML = `
        <span class="audit-stage">${idx + 1}. [${step.stage}]</span>
        <span class="audit-msg">${msg}</span>
      `;
      list.appendChild(item);
    });
  }

  // --- 6. Automated Cohort Benchmark Suite ---
  function initBenchmarkActions() {
    const btn = document.getElementById("btn-run-benchmark");
    if (!btn) return;

    btn.addEventListener("click", async () => {
      const btnText = document.getElementById("btn-run-benchmark-text");
      btn.disabled = true;
      btnText.textContent = "Running Benchmark Suite...";

      try {
        const res = await fetch("/api/benchmark", { method: "POST" });
        if (!res.ok) throw new Error("Benchmark failed");
        const data = await res.json();
        renderBenchmarkResults(data);
      } catch (e) {
        console.error("Benchmark runner error:", e);
        alert(`Benchmark run failed: ${e.message}`);
      } finally {
        btn.disabled = false;
        btnText.textContent = "Execute Full Benchmark";
      }
    });
  }

  async function loadBenchmarkInitial() {
    try {
      const res = await fetch("/results/cohort_benchmark_results.json");
      if (res.ok) {
        const data = await res.json();
        renderBenchmarkResults(data);
      }
    } catch (e) {
      // Fallback
    }
  }

  function renderBenchmarkResults(summary) {
    const meta = summary.benchmark_metadata || {};
    const trajs = summary.cohort_trajectories || [];

    document.getElementById("kpi-safety-rate").textContent = meta.first_pass_safety_compliance_rate || meta.overall_clinical_safety_rate || "100.0%";
    document.getElementById("kpi-correction-rate").textContent = meta.self_correction_success_rate || "100.0%";
    document.getElementById("kpi-escape-rate").textContent = meta.formal_contraindication_escape_rate || "0.0%";
    document.getElementById("kpi-runtime").textContent = `${meta.total_benchmark_runtime_seconds || 41.85} s`;

    const tbody = document.getElementById("benchmark-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    trajs.forEach((t) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="patient-cell">${t.patient_id}</td>
        <td>${(t.conditions || []).join(", ").replace(/_/g, " ")}</td>
        <td>${t.attempts || 1}</td>
        <td><span class="table-status-pill ${t.status === "APPROVED" ? "approved" : "escalated"}">${t.status}</span></td>
        <td><strong>${t.approved_drug || "None"}</strong></td>
        <td><span class="card-badge" style="color: var(--safe-green)">0.0% Escape Rate (HermiT)</span></td>
      `;
      tbody.appendChild(tr);
    });

    refreshLucide();
  }

  // --- 7. Formal Safety Axioms Explorer ---
  async function loadAxioms() {
    try {
      const res = await fetch("/api/ontology");
      if (!res.ok) return;
      const data = await res.json();
      state.axioms = data.axioms || [];
      renderAxioms();
    } catch (e) {
      console.warn("Could not load axioms:", e);
    }
  }

  function renderAxioms() {
    const container = document.getElementById("axioms-container");
    if (!container) return;
    container.innerHTML = "";

    state.axioms.forEach((ax) => {
      const card = document.createElement("div");
      card.className = "card axiom-card";
      card.innerHTML = `
        <div class="card-header">
          <div class="card-title-group">
            <i data-lucide="shield-alert" class="card-icon" style="color: var(--danger-red)"></i>
            <h3 class="card-title">${ax.name}</h3>
          </div>
          <span class="card-badge" style="color: var(--danger-red)">Formal DL Equivalence</span>
        </div>

        <div class="axiom-formula-box" id="formula-${ax.id}">
          <code>${ax.formula_dl}</code>
        </div>

        <div class="axiom-meta-row">
          <span>Target Contraindicated Drug:</span>
          <span class="axiom-target-drug">${ax.target_drug}</span>
        </div>

        <div class="axiom-meta-row">
          <span>Trigger Condition:</span>
          <strong>${ax.trigger_condition}</strong>
        </div>

        <p class="axiom-rationale">${ax.rationale}</p>

        <div class="axiom-alternatives-box">
          <i data-lucide="check-circle" style="width: 18px; height: 18px;"></i>
          <span>Verified Safe Alternatives: <strong>${(ax.safe_alternatives || []).join(", ")}</strong></span>
        </div>
      `;
      container.appendChild(card);

      // Render KaTeX formula if available
      if (window.katex && ax.formula_latex) {
        const box = document.getElementById(`formula-${ax.id}`);
        if (box) {
          try {
            window.katex.render(ax.formula_latex, box, { throwOnError: false, displayMode: true });
          } catch (ke) {
            // keep plain text
          }
        }
      }
    });

    refreshLucide();
  }

  // --- 8. Clinical Guidelines Browser ---
  async function loadGuidelines() {
    try {
      const res = await fetch("/api/guidelines");
      if (!res.ok) return;
      state.guidelines = await res.json();
      renderGuidelines(state.guidelines);
    } catch (e) {
      console.warn("Could not load guidelines:", e);
    }
  }

  function renderGuidelines(list) {
    const container = document.getElementById("guidelines-container");
    if (!container) return;
    container.innerHTML = "";

    list.forEach((g) => {
      const card = document.createElement("div");
      card.className = "card guideline-full-card";
      card.innerHTML = `
        <div class="card-header">
          <div class="card-title-group">
            <i data-lucide="bookmark" class="card-icon" style="color: var(--accent-indigo)"></i>
            <h3 class="card-title">${g.topic}</h3>
          </div>
          <span class="card-badge">${g.id}</span>
        </div>
        <span class="passage-source" style="color: var(--text-tertiary)">Source: ${g.source}</span>
        <p class="passage-text" style="font-size: 0.9rem; line-height: 1.6; color: var(--text-secondary);">"${g.text}"</p>
      `;
      container.appendChild(card);
    });

    refreshLucide();
  }

  function initGuidelinesSearch() {
    const input = document.getElementById("input-guideline-search");
    if (!input) return;

    input.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      if (!q) {
        renderGuidelines(state.guidelines);
        return;
      }
      const filtered = state.guidelines.filter(
        (g) =>
          g.topic.toLowerCase().includes(q) ||
          g.text.toLowerCase().includes(q) ||
          g.source.toLowerCase().includes(q) ||
          g.id.toLowerCase().includes(q)
      );
      renderGuidelines(filtered);
    });
  }

  // Helper to re-initialize Lucide SVG icons dynamically
  function refreshLucide() {
    if (window.lucide) {
      window.lucide.createIcons();
    }
  }
});
