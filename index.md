---
layout: default
title: Enabling Wheeled-Legged Robot Parkour via Reinforcement Practice and Lessons Learned
---

# Enabling Wheeled-Legged Robot Parkour via Reinforcement Practice and Lessons Learned


## SIMTT Parallel Implementation Details

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

## SIMTT Action Variance

Following the reasoning of Messikommer et al.[^messikommer], we associate the
action outputs of each teacher and of the student with multivariate Gaussians of
dimension $$d$$, with state-dependent means
$$\boldsymbol{\mu}_{T_i}(s_t)$$, $$\boldsymbol{\mu}_S(s_t)$$ and state-independent
diagonal covariances $$\boldsymbol{\Sigma}_{T_i}$$,
$$\boldsymbol{\Sigma}_S$$. In this general setting, the KL-divergence between
the $$i$$-th teacher and the student admits the following closed form:

$$
D_{KL}^{(i)}(s_t)
= \tfrac{1}{2}\!\biggl[
\log\frac{|\boldsymbol{\Sigma}_S|}
         {|\boldsymbol{\Sigma}_{T_i}|}
- d
+ \mathrm{Tr}\!\left(
    \boldsymbol{\Sigma}_S^{-1}
    \boldsymbol{\Sigma}_{T_i}
  \right)
+ \bigl(\boldsymbol{\mu}_{T_i}(s_t)
         - \boldsymbol{\mu}_S(s_t)\bigr)^{\!\top}
  \boldsymbol{\Sigma}_S^{-1}
  \bigl(\boldsymbol{\mu}_{T_i}(s_t)
        - \boldsymbol{\mu}_S(s_t)\bigr)
\biggr].
$$

Each teacher $$T_i$$ is a separate expert with its own RL trajectories and its
own characteristic exploration scale. Accordingly, each
$$\boldsymbol{\Sigma}_{T_i}$$ is represented by a separate learned parameter,
which is updated alongside the corresponding teacher encoder $$E_{T_i}$$ during
the PPO updates. The student encoder $$E_S$$, on the other hand, is used purely
as a *deterministic* policy and has no variance of its own: the student
covariance $$\boldsymbol{\Sigma}_S$$ is introduced only to define the Gaussian
surrogate used in the KL-divergence. When evaluating the KL for teacher $$T_i$$,
we set $$\boldsymbol{\Sigma}_S \equiv \boldsymbol{\Sigma}_{T_i}$$ on a
per-teacher basis. The log-determinant, dimensionality, and trace terms then
cancel exactly, leaving

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
auxiliary PPO loss, computed teacher-by-teacher. The intuition is that the loss
grows whenever a teacher is confident in its actions, corresponding to a small
$$\boldsymbol{\Sigma}_{T_i}$$, while its mean still differs significantly from
that of the student encoder $$E_S$$. During the PPO update, the gradient aligns
the teacher mean with the student mean through $$E_{T_i}$$ and increases the
teacher's variance in action dimensions where substantial disagreement remains.
The resulting broader action distribution encourages exploration during the
teacher's rollouts and increases the likelihood of discovering behaviors that
$$E_S$$ can learn. Crucially, since each teacher owns a separate variance
parameter, this exploration pressure is adapted independently to each task,
rather than being averaged across all of them as it would be with a single
shared $$\boldsymbol{\Sigma}_T$$.

## SITT/SIMTT Rollout Subsample Ratio

The **rollout subsample ratio** is the fraction of parallel-environment
trajectories from each fully collected rollout that are replayed during each
student-alignment epoch. A ratio of `1.0` uses every trajectory, whereas `0.5`
samples half of them at random. Rollout collection and the teacher PPO update
still use all environments; subsampling only reduces the repeated alignment
workload, with the goal of saving training time while retaining comparable
performance. In our final configuration, `0.5` selects 2048 of the 4096
trajectories per rank and alignment epoch.

This remains a comparatively large alignment budget. In the original SITT
paper's Isaac Gym manipulation experiment (Appendix A.1.3, p. 15), only 128 of
4096 parallel environments rendered student images for paired alignment at each
timestep (3.125%).[^messikommer] Those streams used a 16-step horizon and fed a
rolling buffer of 100,000 samples. Each of our alignment epochs instead replays
2048 collected environment streams per rank—16 times the original number of
image-generating streams. These counts describe data scale, not identical
operations: the original subset controls paired-observation rendering into a
FIFO buffer, whereas our ratio controls post-collection replay from a fully
collected rollout. The comparison is therefore not a like-for-like compute
benchmark.

**Observed time savings.** We compared `rollout_subsample_ratio=0.5` with `1.0`
using two runs per setting. Total iteration time is collection time plus
learning time:

| Method and timing window | Ratio `1.0` | Ratio `0.5` | Saved per iteration |
|:--|--:|--:|--:|
| SIMTT, mean after iteration 100 | 8.263 s | 7.223 s | 1.040 s (12.6%) |
| SIMTT, final 1000 iterations | 8.187 s | 7.649 s | 0.538 s (6.6%) |
| SITT, mean after iteration 100 | 8.317 s | 6.651 s | 1.666 s (20.0%) |
| SITT, final 1000 iterations | 8.170 s | 6.507 s | 1.663 s (20.4%) |

The alignment time itself was 30.9–37.8% lower for SIMTT and 45.8–46.0% lower
for SITT.

**Training-performance check.** To compare the policies at the same training
progress, we averaged the common iteration window from 8500 through 9500,
inclusive, over the two runs in each setting. Each cell reports ratio
`1.0` → `0.5`:

| Method | Mean reward | Episode length | Timeout rate |
|:--|--:|--:|--:|
| SIMTT | 115.50 → 119.50 | 459.66 → 460.79 | 0.8762 → 0.8869 |
| SITT | 93.74 → 94.69 | 383.91 → 387.53 | 0.8151 → 0.8157 |

The averaged logged metrics remained comparable at ratio `0.5`; in particular,
mean reward did not decrease for either method. The final SIMTT curriculum level
was `8.0` in both settings, while mean SITT curriculum mastery was `0.917` in
both. The evidence therefore supports no observed loss in training performance
alongside the lower iteration time. It does not establish strict statistical
equivalence or matched deployment performance: there are only two runs per
setting, they ran on different cluster hosts, SITT has high between-run
variance, and no matching ratio-`0.5` deployment benchmark was recovered.

[^messikommer]: N. Messikommer, J. Xing, E. Aljalbout, and D. Scaramuzza,
    [“Student-Informed Teacher Training”](https://proceedings.iclr.cc/paper_files/paper/2025/file/a8223b0ad64007423ffb308b0dd92298-Paper-Conference.pdf),
    *International Conference on Learning Representations (ICLR)*, 2025.
