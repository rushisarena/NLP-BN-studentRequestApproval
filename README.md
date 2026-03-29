# NLP-BN-studentRequestApproval
# 🏫 Institutional Leave Approval AI
### Hybrid NLP + Bayesian Network for Probabilistic Leave Decision Modeling

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/pgmpy-0.1.x-orange?style=flat-square" alt="pgmpy"/>
  <img src="https://img.shields.io/badge/NLP-Rule--Based-green?style=flat-square" alt="NLP"/>
  <img src="https://img.shields.io/badge/Type-Hybrid%20AI%20System-purple?style=flat-square" alt="Type"/>
  <img src="https://img.shields.io/badge/Colab-Ready-yellow?style=flat-square&logo=google-colab" alt="Colab"/>
</p>

---

## 📌 Project Overview

This project implements a **hybrid AI system** that combines rule-based Natural Language Processing with a Bayesian Network to predict the probability of leave approval in an academic institution. A student submits a request in plain English — the system extracts intent and credibility from the text, maps them to probabilistic evidence, and infers approval probability through a multi-stakeholder hierarchy.

The project bridges two paradigms:
- **NLP layer** — keyword extraction and rule-based credibility scoring (no trained ML model)
- **Bayesian Network** — CPTs learned from synthetic data via MLE; Variable Elimination for exact inference

---

## 🎯 Problem Statement

Traditional approval systems apply hard rules: `if reason == medical → approve`. Real institutions are probabilistic. A medical claim with no paperwork is not the same as a marriage request with no paperwork.

**Key questions this model answers:**
- Given a free-text leave request, what is the probability of institutional approval?
- How does documentary proof change the outcome for a low-priority reason?
- Which authority (faculty, HoD, principal) is the strongest bottleneck?
- When should the system flag a request for human review?

---

## 🔗 Network Structure

```
SRW → FP
SRW → PR
FP  → HODP
PR  → HODP
HODP → PIP
PR  → PIP
PIP → OB
```

| Node | Full Name | Type | Description |
|------|-----------|------|-------------|
| SRW | Student Request Weight | Root | Strength of the request (from reason_score) |
| FP | Faculty Permission | Intermediate | Faculty grants/denies permission |
| PR | Parent Response | Intermediate | Parent supports/opposes the request |
| HODP | HoD Approval | Intermediate | Head of Department approves |
| PIP | Principal Permission | Intermediate | Principal grants final permission |
| OB | Final Outcome | Leaf / Query | P(OB=1) = approval probability |

**Why this structure?**
- `PR` has **dual influence** — directly on PIP and through HODP, modelling real scenarios where the principal contacts parents independently of the HoD chain
- `SRW` is derived from NLP, not observed directly — it encodes reason priority into the BN
- `OB` is always the **query variable**, never evidence — we compute its probability, never assume it

---

## 🧠 Methodology

### 1. NLP Layer — Priority & Credibility Extraction
The NLP layer is entirely rule-based. No trained model, no external dataset.

```python
# Priority classification — first-match greedy scan
PRIORITY_KEYWORDS = {
    "HIGH":   ["illness", "sick", "hospital", "accident", "emergency", "surgery"],
    "MEDIUM": ["project", "hackathon", "conference", "exam", "internship"],
    "LOW":    ["marriage", "wedding", "function", "festival", "vacation"],
}

# Credibility — starts at 0.5, adjusted by signals
score = 0.5
if has_parent and has_proof:  score += 0.40  # Rule B
if no_proof_phrase:           score -= 0.30  # Rule C
if vague_language:            score -= 0.15  # Rule C
if parents_unaware:           score -= 0.20  # Rule C
```

### 2. Scoring System
Two continuous scores in [0, 1] summarise the request quality:

| Score | Measures | Range |
|-------|----------|-------|
| `reason_score` | Importance/validity of the stated reason | 0.20 – 0.90 |
| `credibility_score` | Trustworthiness based on supporting evidence | 0.00 – 1.00 |

### 3. BN Evidence Mapping (Rules A / B / C)

| Rule | Trigger | Effect |
|------|---------|--------|
| **Rule A** | LOW priority AND credibility < 0.45 | FP=0, PR=0 forced → rejection bias |
| **Rule B** | Parent approval AND proof both detected | cs += 0.40; LOW+cs≥0.60 → FP flips to 1 |
| **Rule C** | No-proof / vague / contradiction signals | cs −= 0.15 to −0.30 per signal (stack) |

### 4. Synthetic Data Generation
```python
# Causal generative process — 4,000 samples
SRW  = Bernoulli(0.55)
FP   = Bernoulli(0.72 if SRW==1 else 0.30)
PR   = Bernoulli(0.65 if SRW==1 else 0.25)
HODP = Bernoulli({(1,1):0.85, (1,0):0.45, (0,1):0.35, (0,0):0.10}[(FP,PR)])
PIP  = Bernoulli({(1,1):0.88, (1,0):0.52, (0,1):0.28, (0,0):0.05}[(HODP,PR)])
OB   = Bernoulli(0.90 if PIP==1 else 0.08)
```

### 5. CPT Learning & Inference
```python
model.fit(data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)
result = inference.query(variables=["OB"], evidence={"SRW":1, "FP":1, "PR":1})
approval_prob = float(result.values[1])   # P(OB=1)
```

---

## 📊 Key Results

| Query / Case | Approx. Probability | Notes |
|---|---|---|
| Medical emergency + parent letter + certificate | ~88–93% | Rule B, SRW=1 FP=1 PR=1 |
| Marriage + no proof (Rule A fires) | ~8–15% | FP=0 PR=0 forced |
| Marriage + strong proof (Rule B rescue) | ~55–70% | FP flips to 1; SRW=0 dampens |
| National hackathon + proof + parent support | ~70–78% | MEDIUM priority; SRW=1 |
| Vague, no proof, parents unaware | ~4–10% | Rule A+C stack; cs≈0.0 |

### Score Thresholds Summary

```
reason_score ≥ 0.5          → SRW = 1 (strong request)
credibility_score ≥ 0.65    → PR  = 1 (trusted)
credibility_score ∈ [0.4, 0.65) → PR = 1 (marginal — CPT dampens)
credibility_score < 0.4     → PR  = 0 (untrusted)
priority = HIGH             → FP  = 1 always
priority = LOW, cs < 0.45   → FP  = 0 (Rule A)
priority = LOW, cs ≥ 0.60   → FP  = 1 (Rule B rescue)
```

---

## 📂 Repository Structure

```
leave-approval-ai/
│
├── README.md                  ← You are here
├── leave_approval_system.py   ← Full pipeline (NLP + BN + inference)
│
└── docs/
    ├── Leave_Approval_AI_Guide.pdf          ← Module walkthroughs
    └── Leave_Approval_AI_Complete_Guide.pdf ← Review questions + answers
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Upload `leave_approval_system.py` to Colab
2. Run the install cell:
```bash
!pip install pgmpy pandas numpy -q
```
3. Runtime → Run all

### Option 2: Local
```bash
pip install pgmpy pandas numpy
python leave_approval_system.py
```

### Evaluate a Request
```python
from leave_approval_system import build_and_fit_model, evaluate_request, print_report

inference_engine = build_and_fit_model()

text = "I had a road accident. I have a medical certificate and my parents sent a letter."
result = evaluate_request(text, inference_engine)
print_report(text, result)

# Approval Probability : 91.3%
# Priority             : HIGH  (reason_score=0.9)
# Credibility          : HIGH  (score=0.90)
# Override (Rule B)    : YES
```

---

## 🔬 Module Reference

| Function | Purpose |
|----------|---------|
| `extract_nlp_features(text)` | NLP layer — returns priority, scores, notes |
| `classify_priority(text)` | Keyword scan → priority tier + reason_score |
| `assess_credibility(text)` | Rules A/B/C → credibility_score + notes |
| `map_to_bn_evidence(features)` | Scores → discrete BN evidence {SRW, FP, PR} |
| `generate_synthetic_data()` | 4,000 synthetic training cases |
| `build_and_fit_model()` | Constructs BN, fits CPTs, returns VE engine |
| `evaluate_request(text, inf)` | Full pipeline → result dict |
| `print_report(text, result)` | Formatted decision report |

---

## ⚠️ Limitations

1. **Keyword-only NLP** — Cannot handle paraphrasing or novel phrasing not in the dictionaries. Upgrading to a fine-tuned classifier would improve recall.
2. **Synthetic data** — CPTs reflect designer priors, not real institutional records. Replace with historical data when available.
3. **Binary variables only** — Partial approvals (e.g. "approved for 2 of 5 days") are not modelled.
4. **Static model** — No memory of prior requests. A Dynamic BN could track repeated leave patterns.
5. **MLE sensitivity** — Sparse parent combinations may have noisy CPT estimates. A Bayesian Estimator with Dirichlet priors would be more robust.

---

## 🔭 Future Improvements

| Improvement | Technique | Benefit |
|-------------|-----------|---------|
| Upgrade NLP | Fine-tuned DistilBERT / Anthropic API | Handle paraphrasing and novel phrasing |
| Real training data | Anonymised institutional records | Accurate CPTs from ground truth |
| Robust CPT estimation | Bayesian Estimator (Dirichlet prior) | Better for sparse combinations |
| Continuous BN | Gaussian / hybrid nodes | Remove discretisation information loss |
| Structure learning | Hill-Climbing + BIC score | Discover dependencies from data |
| Interactive UI | Streamlit / Gradio | User-facing probability explorer |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pgmpy` | Bayesian Network definition, fitting, inference |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations, clipping |

---

## 📄 License

MIT License — free to use, modify, and cite.

---

## 👤 Author

**[Rushikesh Y. Dhote]**  
B.Tech — [Industrial IoT]  
[SVPCET, nagpur.]  
rushikeshstudissue@gmail.com  
[LinkedIn](https://www.linkedin.com/in/rushikesh-dhote-bb53383b9) | [GitHub](https://github.com/rushisarena)


---

*Developed as part of coursework in Intelligent Systems / Probabilistic AI, demonstrating hybrid NLP and Bayesian reasoning applied to institutional decision-making under uncertainty.*
