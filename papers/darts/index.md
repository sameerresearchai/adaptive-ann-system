---
layout: page
title: "DARTS: Differentiable Architecture Search"
permalink: /papers/darts/
description: "Accurate explainer of DARTS with core equations, optimization details, and practical caveats."
---

# DARTS: Differentiable Architecture Search

> Source paper: Liu, Simonyan, Yang (2018), arXiv:1806.09055

DARTS (Differentiable ARchiTecture Search) is a foundational NAS method that replaces expensive discrete search (RL/evolution) with gradient-based optimization over a continuous relaxation of architecture choices.

## Why DARTS mattered

Earlier NAS approaches achieved strong performance but needed massive compute:

- NASNet (RL): ~2000 GPU-days
- AmoebaNet (evolution): ~3150 GPU-days

DARTS reduced architecture search to a few GPU-days while remaining competitive.

## Search Space (Cell as DAG)

DARTS searches for a *cell* represented as a directed acyclic graph (DAG):

- Node `x^(i)`: latent representation
- Edge `(i, j)`: operation applied to predecessor node
- Cell output: reduction over intermediate nodes (typically concatenation)

Intermediate node computation:

```text
x^(j) = sum_{i<j} o^(i,j)(x^(i))
```

## Continuous Relaxation

Let `O` be candidate operations (e.g., separable conv, pooling, identity, zero).
Instead of one discrete op per edge, DARTS uses a soft mixture:

```text
o_bar^(i,j)(x) = sum_{o in O} [exp(alpha_o^(i,j)) / sum_{o' in O} exp(alpha_o'^(i,j))] * o(x)
```

At the end of search, DARTS discretizes by selecting strongest operations.

```text
o^(i,j) = argmax_{o in O} alpha_o^(i,j)
```

## Bilevel Optimization Core

DARTS optimizes architecture parameters `alpha` and network weights `w` with a bilevel objective:

```text
min_alpha L_val(w*(alpha), alpha)
s.t. w*(alpha) = argmin_w L_train(w, alpha)
```

Interpretation:

- `w` should fit training data
- `alpha` should be selected by validation performance
- This separation reduces architecture overfitting

## Practical Gradient Approximation

Exact architecture gradients are expensive due to inner optimization for `w*(alpha)`.
DARTS uses one-step unrolling:

```text
w' = w - xi * grad_w L_train(w, alpha)
```

Second-order architecture gradient (unrolled):

```text
grad_alpha L_val(w', alpha) - xi * grad_alpha(grad_w L_train(w, alpha)^T * grad_w' L_val(w', alpha))
```

Finite-difference approximation for Hessian-vector part:

```text
grad_alpha(grad_w L_train(w, alpha)^T * grad_w' L_val(w', alpha))
  ~= [grad_alpha L_train(w+, alpha) - grad_alpha L_train(w-, alpha)] / (2 * epsilon)
```

with:

```text
w+/- = w +/- epsilon * grad_w' L_val(w', alpha)
```

## Deriving Discrete Cells

For each intermediate node, keep top-`k` strongest non-zero incoming operations:

- Convolutional cells: `k = 2`
- Recurrent cells: `k = 1`

Operation strength:

```text
strength(o) = exp(alpha_o^(i,j)) / sum_{o' in O} exp(alpha_o'^(i,j))
```

## Reported Results (Original Paper)

### CIFAR-10 (Convolutional search)

- DARTS (second-order): **2.76%** test error, **3.3M** params, **4 GPU-days** search
- Competitive with methods using orders-of-magnitude more search compute

### Penn Treebank (Recurrent search)

- DARTS (second-order): **55.7** test perplexity, **23M** params, **~1 GPU-day** search

## What is still state-of-the-art today?

DARTS is historically foundational, but not the final word in NAS.

- It is still a standard reference for differentiable NAS.
- Modern practice often uses improved DARTS variants to address known instabilities (e.g., skip-connection collapse in some search spaces).
- So the state-of-the-art position is: **DARTS remains foundational and influential; newer variants can be stronger in robustness.**

## Key Takeaways

- DARTS made NAS differentiable and efficient.
- Bilevel optimization (train vs validation separation) is the central idea.
- First-order is faster; second-order is generally more accurate.
- Search-space design remains critically important.

## References

- Paper: https://arxiv.org/abs/1806.09055
- Code: https://github.com/quark0/darts
