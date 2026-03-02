
# Climate optimizer based on szleb and szleb-gym  — Lightweight MPCRL for Greenhouse Climate Control (QDLDL/QP)

## Summary

`szleb_mpc_qdldl_optimizer` is a **lightweight Model Predictive Control** library for greenhouse / grow-room climate optimization. It aims to keep **key plant vital parameters** (e.g., **inside temperature Tin** and **inside relative humidity RHin**) within **species-specific optimal bands**, while minimizing **actuation energy**.

The library is updated to replace the heavy **CasADi + IPOPT nonlinear program (NLP)** solve  with a **convex Quadratic Program (QP)** solve using an **OSQP-style operator-splitting method**  backed by the **QDLDL** C implementation of **LDLᵀ factorization** . This makes the control loop much more suitable for real-time and edge deployments.

---

## Introduction

Modern controlled-environment agriculture (CEA) requires balancing:

* **Plant comfort**: keep Tin/RHin in optimal ranges (tight quality and yield constraints).
* **Energy**: heaters, fans, and ventilation are costly and must be used sparingly.
* **Adaptation**: real rooms drift, crops change, and models are imperfect.

This library uses:

1. **Online system identification** (RLS) to learn a linear predictive model of the grow room:
   
   $x_{k+1}=A x_k + B u_k + E z_k + b$

   where (x=[Tin, RHin]), (u) are actuator decisions, and (z) are exogenous drivers (e.g., Tout/RHout, solar, LAI). 

2. **Fast QP-based MPC** (rather than NLP) to compute actions that track optimal bands with minimal energy. The OSQP formulation is:
   
   $\min \tfrac12 x^T P x + q^T x \quad \text{s.t. } l \le Ax \le u$


3. A **C-backed sparse linear algebra kernel** using QDLDL (`QDLDL_etree`, `QDLDL_factor`, `QDLDL_solve`) for repeated KKT solves , enabling factorization reuse and fast iteration.

---

## Objectives

* **Primary:** Keep Tin/RHin inside plant-optimal bands (constraint satisfaction).
* **Secondary:** Minimize energy consumption from actuators (heater/fan/vents).
* **Tertiary:** Learn and adapt online with minimal overhead (RLS model update).
* **Engineering goal:** Enable real-time optimization: OSQP-style methods are designed to reuse a quasi-definite factorization and support warm-starting and factorization caching. 

[Diagram](images/RL_szleb_mpc_diagram.png)
---

## Formalism

### 1) Online predictive model (RLS)

We learn a linear model using Recursive Least Squares (RLS):

* Feature vector: ($\phi=[x,u,z,1]$) 
* Parameter matrix: (\Theta\in\mathbb{R}^{n_x\times(n_x+n_u+n_z+1)}) 
* Prediction and update follow standard RLS recursion. 

### 2) MPC as a convex QP (OSQP form)

At each step, MPC solves a QP of the form:


$\min \tfrac12 w^T P w + q^T w \quad \text{s.t. } l \le A w \le u$

This is well-matched to your learned linear dynamics and quadratic penalties (energy, smoothing, slack penalties).

The OSQP paper highlights why this is practical for embedded/real-time:

* The method repeatedly solves a **quasi-definite** linear system with (almost) the **same coefficient matrix** 
* Supports **factorization caching** and **warm starting** to accelerate repeated solves 
* The open-source C implementation has a “small footprint” and is designed for embedded applications 

### 3) QDLDL (C) linear algebra backend

QDLDL is a C library providing LDLᵀ factorization for quasi-definite sparse systems. The typical flow is:

1. elimination tree: `QDLDL_etree`
2. factorization: `QDLDL_factor`
3. solve: `QDLDL_solve` 

This exactly matches the low-level solver needs of OSQP-style methods and is why QDLDL is used inside the OSQP ecosystem. 

### 4) Why not CasADi + IPOPT (previous approach)

The original implementation formulated MPC as a nonlinear program and solved it with IPOPT via CasADi: 
That approach is powerful, but it has much higher runtime overhead for repeated real-time solves, especially when the model is already linear/quadratic (as with RLS).

---

## Installation

### 1) Install QDLDL (required, C library)

Build and install QDLDL (shared library + headers). Example:

```bash
git clone https://github.com/osqp/qdldl.git
cd qdldl
mkdir -p build && cd build
cmake .. -DQDLDL_BUILD_SHARED_LIB=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
sudo cmake --install .
sudo ldconfig
```

QDLDL usage follows the standard API pattern shown in the QDLDL reference/example (etree → factor → solve). 

### 2) Install `szleb_mpc_qdldl_optimizer`

From your project root (inside your venv):

```bash
pip install -e .
```

---

## Quick start (training + evaluation)

### Training

Run your training entrypoint (example):

```bash
python train_szleb_mpcrl_v2.py
```

The training loop learns:

* an online linear model (RLS) 
* policy behavior via MPC actions, while logging “slack” diagnostics (sT0, sRH0) 

### Evaluation / reporting

Use your evaluation script (e.g., `evaluate_szleb_mpcrl_report_v4.py`) to generate plots and summary metrics.

---

## Library tree structure

```text
szleb_mpc_optimizer/
├── train_szleb_mpcrl_v2.py                 # training entrypoint
└── szleb_mpcrl/
    ├── __init__.py                         # lazy imports
    ├── agent.py                            # MPCRL agent (MPC + RLS + freeze for eval)
    ├── rls.py                              # online linear model learning: x_{k+1}=Ax+Bu+Ez+b :contentReference[oaicite:20]{index=20}
    ├── mpc.py                              # builds QP and calls solver (QDLDL-backed)
    ├── costs.py                            # weights, band violations, energy proxy
    ├── envs.py                             # Gym/Gymnasium wrappers + state extraction :contentReference[oaicite:21]{index=21}
    └── solvers/
        ├── qdldl_ctypes.py                  # ctypes binding to libqdldl.so (C backend)
        ├── osqp_qdldl.py                    # OSQP-style ADMM/QP solver using QDLDL
        └── csc_utils.py                     # sparse/CSC helpers (upper-tri extraction)
```

---

## Results and expected behavior

### What you should see in the reports

Your report plots typically include:

* **Inside parameters vs optimal bands** (Tin/RHin)
* **Outside drivers** (Tout, RHout)
* **Actuator decisions and ON/OFF status** (heater/fan/vents)
* **Energy split**
* **RLS one-step prediction error** (e.g., MAE/RMSE)
* **Constraint satisfaction rate** 

A common failure mode (seen previously) is “do-nothing control” where triggers and energy stay flat while Tin/RHin drift outside bands . The QP/QDLDL update is designed to address this by:

* making solves fast enough to act every step,
* enabling stable constraint handling with slacks,
* and leveraging the RLS linear model that matches QP structure.

### Performance notes (from the attached OSQP paper)

The OSQP approach is designed for **high accuracy** QP solutions using operator splitting with a reusable quasi-definite factorization  and is reported to be typically faster than many interior-point methods on benchmark classes, especially with warm-start/caching. 

[Diagram](images/best_window_000.png)
---

## References

* **OSQP (operator splitting QP solver):** Stellato et al., *OSQP: an operator splitting solver for quadratic programs*, Mathematical Programming Computation (2020). 

