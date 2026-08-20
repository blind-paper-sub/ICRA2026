---
layout: default
title: Anonymous Project Page
---

# Anonymous Project Page

This page accompanies an anonymous research submission.

### SIMTT Implementation Details

The implementation also includes parallel optimizations. Training proceeds
through three distinct phases:

- **Rollout Phase:** Each rank performs rollouts using its own teacher, without
  communicating with the other ranks.

- **Policy Update and Decoder Alignment Phase:** Each rank computes the policy
  update gradient for its own teacher independently and in parallel, again
  without any communication with the others (Policy Update Phase). An
  `all_reduce` operation is then performed across all ranks so that each rank
  obtains the average PPO update gradient applied to decoder $$D$$ (Decoder
  Alignment Phase). A single optimizer step is then performed: $$D$$ is updated
  using the averaged gradients, while each teacher encoder $$E_{T_i}$$ is updated
  using its own local gradients.

- **Student Alignment Phase:** After each rank computes the student alignment
  gradient relative to its own teacher, an `all_reduce` operation is performed
  across all ranks to average the gradients. The averaged gradient is then
  backpropagated by all ranks through student encoder $$E_S$$, resulting in a
  synchronized update across all ranks.

![Parallel multi-GPU implementation of SIMTT](images/parallel-implementation.png)

*Parallel SIMTT implementation. Each rank updates its local teacher; decoder
and student gradients are aggregated across ranks using All-Reduce.*

## Action Variance

Following the reasoning of Messikommer et al.[^messikommer], we associate the
action outputs of each teacher and of the student with multivariate Gaussians of
dimension $$d$$, with state-dependent means
$$\boldsymbol{\mu}_{T_i}(s_t)$$, $$\boldsymbol{\mu}_S(s_t)$$ and state-independent
diagonal covariances $$\boldsymbol{\Sigma}_{T_i}$$,
$$\boldsymbol{\Sigma}_S$$, which is a learned parameter, updated alongside the
corresponding teacher encoder $$E_{T_i}$$ during the PPO updates. The student
encoder $$E_S$$, on the other hand, is used purely as a *deterministic* policy and
has no variance of its own: the student covariance $$\boldsymbol{\Sigma}_S$$ is
introduced only to define the Gaussian surrogate used in the KL-divergence.
When evaluating the KL for teacher $$T_i$$, we set
$$\boldsymbol{\Sigma}_S \equiv \boldsymbol{\Sigma}_{T_i}$$ on a per-teacher basis.
With these hypothesis, the KL-divergence becomes:

$$
D_{KL}^{(i)}(s_t)
= \tfrac{1}{2}
\bigl(\boldsymbol{\mu}_{T_i}(s_t)
- \boldsymbol{\mu}_S(s_t)\bigr)^{\!\top}
\boldsymbol{\Sigma}_{T_i}^{-1}
\bigl(\boldsymbol{\mu}_{T_i}(s_t)
- \boldsymbol{\mu}_S(s_t)\bigr).
$$

This is the expression actually used in the rollout penalty and in the
auxiliary PPO loss, computed teacher-by-teacher.

[^messikommer]: N. Messikommer, J. Xing, E. Aljalbout, and D. Scaramuzza,
    “Student-Informed Teacher Training,” *International Conference on Learning
    Representations (ICLR)*, 2025.
