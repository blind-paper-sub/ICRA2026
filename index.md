---
layout: default
title: Anonymous Project Page
---

# Anonymous Project Page

This page accompanies an anonymous research submission.

## SIMTT Implementation

The implementation also includes parallel optimizations. Training proceeds
through three distinct phases:

- **Rollout Phase:** Each rank performs rollouts using its own teacher, without
  communicating with the other ranks.

- **Policy Update and Decoder Alignment Phase:** Each rank computes the policy
  update gradient for its own teacher independently and in parallel, again
  without any communication with the others (Policy Update Phase). An
  `all_reduce` operation is then performed across all ranks so that each rank
  obtains the average PPO update gradient applied to decoder `D` (Decoder
  Alignment Phase). A single optimizer step is then performed: `D` is updated
  using the averaged gradients, while each teacher encoder `E_Ti` is updated
  using its own local gradients.

- **Student Alignment Phase:** After each rank computes the student alignment
  gradient relative to its own teacher, an `all_reduce` operation is performed
  across all ranks to average the gradients. The averaged gradient is then
  backpropagated by all ranks through student encoder `E_S`, resulting in a
  synchronized update across all ranks.

![Parallel multi-GPU implementation of SIMTT](images/parallel-implementation.png)

*Parallel SIMTT implementation. Each rank updates its local teacher; decoder
and student gradients are aggregated across ranks using All-Reduce.*

## Action Variance

Following the reasoning of Messikommer et al.[^messikommer], we associate the
action outputs of each teacher and the student with multivariate Gaussian
distributions of dimension `d`.

For teacher `i`, the state-dependent mean is `μ_Ti(s_t)` and the
state-independent diagonal covariance is `Σ_Ti`. The student mean is
`μ_S(s_t)`. Each teacher covariance `Σ_Ti` is a learned parameter updated
alongside the corresponding teacher encoder `E_Ti` during PPO updates.

The student encoder `E_S`, however, is used purely as a *deterministic* policy
and has no variance of its own. The student covariance `Σ_S` is introduced only
to define the Gaussian surrogate used in the KL divergence. When evaluating
the divergence for teacher `i`, we set `Σ_S` equal to `Σ_Ti` for that teacher.

Under these assumptions, define the mean difference and KL divergence as:

```text
Δμ_i(s_t) = μ_Ti(s_t) − μ_S(s_t)

D_KL,i(s_t) = ½ · Δμ_i(s_t)ᵀ · Σ_Ti⁻¹ · Δμ_i(s_t)
```

This is the expression actually used in the rollout penalty and in the
auxiliary PPO loss, computed teacher-by-teacher.

[^messikommer]: N. Messikommer, J. Xing, E. Aljalbout, and D. Scaramuzza,
    “Student-Informed Teacher Training,” *International Conference on Learning
    Representations (ICLR)*, 2025.
