# Choosing a Maintenance Policy Under Failure Uncertainty

## Decision Intelligence Case Study  
**Simulation, risk-aware optimisation, and policy comparison**

---

## 1. Context & Motivation

Maintenance decisions for critical assets are rarely about *predicting* failures perfectly.  
They are about **choosing a policy** that balances:

- expected cost  
- operational risk  
- downtime exposure  
- uncertainty in failure and repair processes  

This project studies a **maintenance policy selection problem under uncertainty**, using Monte Carlo simulation and risk-adjusted optimisation to support defensible decisions.

The goal is not to build a predictive maintenance model, but to answer:

> *“Given uncertainty, which maintenance policy should we choose — and why?”*

---

## 2. Decision Problem

A company operates a critical asset over a fixed planning horizon.

Failures are:
- stochastic (random failure times)  
- costly (repair cost, downtime, penalties)  

Maintenance actions:
- reduce failure risk  
- incur direct cost and downtime  

The decision-maker must choose **one maintenance policy**.

---

## 3. Policies Evaluated

### **RTF — Run to Failure**
- No preventive actions  
- Corrective maintenance only after failure  
- Lowest preventive cost, highest risk  

---

### **TBM — Time-Based Maintenance**
- Preventive maintenance at fixed calendar intervals  
- Deterministic schedule  
- Stable but potentially over-conservative  

---

### **CBM — Condition-Based Maintenance**
- Periodic inspections  
- Maintenance triggered when a condition proxy exceeds a threshold  

In this simplified model:
- **Condition = age since last reset**  
- Threshold defines how conservative the policy is  

CBM interpolates naturally between:
- TBM-like behaviour (low threshold)  
- RTF-like behaviour (high threshold)  

---

## 4. Modelling Approach

### System Dynamics
- Renewal-process model  
- After repair or preventive action, the asset is “as good as new”  

### Uncertainty Models
- **Failure time**: Weibull distribution  
- **Repair duration**: Lognormal distribution  

These choices capture:
- increasing failure risk with age  
- right-tailed downtime behaviour  

---

## 5. Evaluation Methodology

### Monte Carlo Simulation
Each policy is simulated thousands of times to estimate:
- expected cost  
- downtime distribution  
- probability of at least one failure  
- tail-risk metrics (p95)  

### Risk-Adjusted Objective
Policies are compared using:

J = mean_cost + λ · P(failure ≥ 1)

Where:
- λ expresses risk aversion  
- higher λ penalises exposure to failure  

This allows **explicit trade-offs between cost and risk**, instead of hiding risk inside averages.

---

## 6. Optimisation Strategy

### Why Grid Search
- Low-dimensional decision variables  
- Noisy Monte Carlo objective  
- Fully explainable results  

A two-stage approach is used:
1. **Coarse grid** → identify promising region  
2. **Refined grid** → locate optimum accurately  

### Optimised Parameters
- **TBM**: preventive interval (days)  
- **CBM**: condition threshold (days since last reset)  

---

## 7. Key Results & Insights

### CBM Behaviour
CBM exhibits **regime changes**:
- Below a certain threshold, every inspection triggers maintenance  
- Above it, maintenance becomes rare and failures dominate  

This creates **step-like objective curves**, which are:
- expected  
- meaningful  
- diagnostic of correct policy logic  

### Interpretation
CBM does not smoothly outperform TBM or RTF — it **changes behaviour discretely** based on the threshold.

This highlights:
- why optimisation is needed  
- why “average cost” alone is insufficient  
- how risk preferences drive policy choice  

---

## 8. What This Case Demonstrates

- Decision-making under uncertainty (not prediction)  
- Simulation-based policy evaluation  
- Risk-aware optimisation  
- Clear separation between:
  - assumptions  
  - uncertainty  
  - decisions  
  - outcomes  
- Practical Decision Intelligence thinking  

---

## 9. Limitations & Extensions

This is a **deliberately simplified v1 model**.

Not included (yet):
- degradation states  
- imperfect inspections  
- resource constraints  
- multi-asset interactions  

These omissions are intentional to keep:
- interpretation clear  
- results explainable  
- decisions traceable  

---

## 10. Takeaway

> **The value of maintenance analytics lies not in predicting failures, but in choosing policies that balance cost and risk under uncertainty.**

This project demonstrates how simulation and optimisation can support such decisions in a transparent and defensible way.
