---
layout: page
title: "Input-Adaptive Hard-Exit DARTS"
permalink: /papers/input-adaptive-early-exit-darts/
description: "A corrected Input-Adaptive DARTS formulation with true hard early-exit inference that actually reduces computation."
---

# Input-Adaptive Hard-Exit DARTS

> This document presents a full mathematical formulation of Input-Adaptive Early-Exit DARTS and clarifies the key correction required for **actual inference-time compute savings**.

Consider a supervised learning problem with input samples `x in X` and labels `y in Y`. Let:

- `w` denote network weights,
- `alpha` denote global architecture parameters,
- `theta` denote parameters of an input-dependent controller,
- `g_theta(.)` denote the controller function.

Unlike standard DARTS, which learns one architecture shared by all samples, this framework allows both architecture selection and inference depth to adapt per input.

## 1.1 Input-Adaptive Architecture

For a given input sample `x`, define sample-dependent architecture perturbation:

```text
Delta alpha(x) = g_theta(x)                                                   (1)
```

The effective architecture parameters become:

```text
alpha'(x) = alpha + Delta alpha(x)                                            (2)
```

Here, `alpha` is a global architecture prior and `Delta alpha(x)` provides sample-specific adaptation.

Assume `K` candidate operations:

```text
O = {o_1, o_2, ..., o_K}                                                      (3)
```

Following DARTS relaxation, for edge `(j, i)` the operation probability is:

```text
p_(j,k)^i(x) = exp(alpha'_(j,k)^i(x)) / sum_{m=1}^K exp(alpha'_(j,m)^i(x))   (4)
```

Node representation:

```text
h_i = sum_{j<i} sum_{k=1}^K p_(j,k)^i(x) o_k(h_j)                             (5)
```

When `Delta alpha(x) = 0`, this reduces to standard DARTS architecture mixing.

## 1.2 Intermediate Predictions

Let each candidate exit node `i` include an auxiliary prediction head:

```text
y_i = f_i(h_i)                                                                 (6)
```

Hence, each intermediate depth can produce a valid prediction.

## 1.3 Exit Modeling: Soft Training Form

Define exit logits over `N` candidate exits:

```text
beta(x) = [beta_1(x), beta_2(x), ..., beta_N(x)]                              (7)
```

A softmax can produce normalized exit weights:

```text
q_i(x) = exp(beta_i(x)) / sum_{m=1}^N exp(beta_m(x)),  i=1,...,N              (8)
```

and the training-time blended output:

```text
y_hat = sum_{i=1}^N q_i(x) y_i                                                 (9)
```

This is differentiable and useful for optimization.

## 1.4 Why Softmax Exit Alone Does Not Guarantee Compute Savings

If the model computes all nodes first and only then forms `q_i(x)` and `argmax_i q_i(x)`, full-depth compute has already been spent.

So the statement:

```text
e(x) = argmax_i q_i(x)                                                        (10)
```

only yields compute reduction if `q_i` is available causally during forward execution (without evaluating deeper nodes first).

This is the core correction needed for experiment-time speedups.

## 1.5 Causal Exit Parameterization for Real Hard Exit

To enable true early termination, define node-local stop logits using current state only:

```text
beta_i = u_i(h_i),
s_i = sigma(beta_i) in (0, 1)                                                 (11)
```

where `s_i` is the probability of stopping at node `i` conditioned on reaching `i`.

The implied probability of exiting exactly at node `i` is:

```text
q_1 = s_1                                                                      (12)
q_i = s_i * prod_{m=1}^{i-1} (1 - s_m),     i=2,...,N-1                       (13)
q_N = prod_{m=1}^{N-1} (1 - s_m)                                               (14)
```

This forms a valid distribution (`sum_i q_i = 1`) and is causal by construction.

## 1.6 Computation Cost Model

Let:

```text
C_i                                                                            (15)
```

denote cumulative computation up to exit node `i` (FLOPs, latency, memory, or energy).

Expected compute under exit distribution:

```text
F(x) = sum_{i=1}^N q_i(x) C_i                                                  (16)
```

Because `q_i` is differentiable during training, compute can be optimized jointly with accuracy.

## 1.7 Architecture Regularization

To prevent over-fragmented per-sample architectures, use L2 regularization on perturbation:

```text
R(x) = ||Delta alpha(x)||_2^2                                                  (17)
```

This keeps sample-specific architectures close to the global prior.

## 1.8 Learning Objective

Prediction loss:

```text
L_acc = CE(y, y_hat)                                                           (18)
```

Combined objective:

```text
L = L_acc + lambda F(x) + mu R(x)                                              (19)
```

Substituting `F(x)` and `R(x)`:

```text
L = CE(y, y_hat) + lambda sum_{i=1}^N q_i(x) C_i + mu ||Delta alpha(x)||_2^2  (20)
```

where:

- `lambda` controls accuracy-vs-compute trade-off,
- `mu` controls strength of architecture regularization.

Overall optimization problem:

```text
min_{w, alpha, theta} E_{(x,y)} [
    CE(y, y_hat) + lambda sum_{i=1}^N q_i(x) C_i + mu ||Delta alpha(x)||_2^2
]                                                                               (21)
```

## 1.9 True Hard-Exit Inference Rule

During inference, use sequential stopping instead of post-hoc argmax over full-depth outputs.

Given threshold `tau`:

```text
for i = 1..N:
    compute h_i and y_i
    beta_i = u_i(h_i)
    s_i = sigmoid(beta_i)
    if s_i >= tau or i == N:
        return y_i
```

Equivalent exit index:

```text
e(x) = min { i : s_i >= tau }  (or e(x)=N if no earlier trigger)              (22)
```

Then only nodes `1..e(x)` are executed, so realized per-sample compute is exactly `C_{e(x)}`.

This is the mechanism that delivers real inference-time efficiency.

## 1.10 Relation to Standard DARTS

If the controller outputs no perturbation:

```text
Delta alpha(x) = 0                                                             (23)
```

then:

```text
alpha'(x) = alpha                                                              (24)
```

If compute is not penalized:

```text
lambda = 0                                                                     (25)
```

the objective becomes:

```text
L = CE(y, y_hat)                                                               (26)
```

and optimization reduces to:

```text
min_{w, alpha} E_{(x,y)} [ CE(y, y_hat) ]                                      (27)
```

which corresponds to standard DARTS behavior (without sample-adaptive perturbation and without compute-aware exit pressure).

## 1.11 Practical Experimental Protocol

To avoid mismatch between optimization and deployment, use:

1. **Training phase**
   - Optimize with soft expected objective (`q_i`, `F(x)`) for stable gradients.
2. **Validation phase**
   - Sweep `tau` (or policy parameters) to obtain accuracy/compute Pareto points.
3. **Inference reporting**
   - Use strict hard stopping policy from Section 1.9.
   - Report mean compute `E[C_{e(x)}]`, tail/worst-case compute `C_N`, and accuracy.

This ensures that claimed computational gains are physically realized during execution.

## 1.12 Accuracy-First Optimization Strategy

To keep the method highly optimized while preserving accuracy, use a staged objective schedule instead of optimizing all pressures equally from the first epoch.

Recommended schedule:

1. **Warm-up for accuracy**
   - Train with `lambda = 0` for initial epochs to stabilize representations and avoid premature shallow exits.
2. **Progressive efficiency pressure**
   - Increase `lambda` gradually (linear or cosine ramp) so the model learns compute-efficient exits without collapsing accuracy.
3. **Controlled architecture perturbation**
   - Start with moderate `mu`, then tune so `Delta alpha(x)` improves specialization without noisy over-adaptation.

This schedule typically produces better final accuracy at the same compute budget than applying strong compute pressure from the start.

## 1.13 Industry-Standard Training and Inference Recipe

For reproducible, production-grade behavior, use the following defaults.

### Training-time standards

- Use mixed precision and gradient scaling for throughput while preserving numerical stability.
- Apply gradient clipping (especially for controller and architecture parameters).
- Use exponential moving average (EMA) of weights for more stable validation metrics.
- Keep a strict train/validation split for architecture-related tuning (`alpha`, `theta`, `tau`).
- Track per-exit accuracy (`Acc_i`) and average realized compute (`E[C_{e(x)}]`) every validation cycle.

### Inference-time standards

- Use deterministic hard-exit policy from Section 1.9.
- Set an explicit final-exit fallback (`i = N`) so no sample fails to produce output.
- Calibrate confidence scores on a held-out set before deployment.
- Optionally enforce a minimum depth for difficult classes if early exits hurt tail accuracy.

## 1.14 Robust Hard-Exit Decision Rule

For strong practical accuracy, combine stop probability and predictive confidence:

```text
for i = 1..N:
    compute h_i, y_i
    s_i = sigmoid(u_i(h_i))
    c_i = max softmax(y_i)
    if (s_i >= tau_s and c_i >= tau_c) or i == N:
        return y_i
```

This dual-threshold rule is often more robust than stop-probability alone because it avoids uncertain early predictions.

## 1.15 Metrics Required for Claiming "Works in Practice"

To satisfy industry-style evaluation, report all of the following:

- Top-1/Top-5 accuracy at fixed compute budgets.
- Accuracy-compute Pareto frontier over `tau` settings.
- Mean, P90, and worst-case compute/latency.
- Exit-depth histogram (fraction of samples exiting at each node).
- Calibration quality (ECE or reliability curves), especially for early exits.

Without these metrics, efficiency claims can hide unacceptable accuracy regression on hard samples.

## 1.16 Recommended Hyperparameter Defaults

Starting points (to be tuned per dataset/model scale):

- `lambda`: start near `0`, ramp to a small target range (for example `1e-3` to `1e-1` after normalization of `C_i`).
- `mu`: small but non-zero (for example `1e-4` to `1e-2`) to prevent unstable architecture drift.
- `tau`: selected on validation to meet a required minimum accuracy constraint first, then minimize compute.

In deployment, treat accuracy as a hard constraint and compute reduction as a secondary objective:

```text
maximize   compute_saving
subject to accuracy >= accuracy_target
```

This policy aligns with real production requirements and keeps the method both optimized and reliable.
