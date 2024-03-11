---
title: Noisy MCMC - The Pseudo-Marginal and Monte Carlo within Metropolis Algorithms
subtitle: I explore so-called "noisy" MCMC algorithms, where exact density evaluations are replaced by a noisy approximation.
layout: default
date: 2023-12-31
keywords: MCMC, Metropolis-Hastings, inverse-problem, pseudo-marginal
published: false
---

{% katexmm %}
The Metropolis-Hastings (MH) Markov Chain Monte Carlo (MCMC) method is a general
purpose algorithm, in theory able to produce samples distributed according to
any generic probability density $\pi(\mathbf{x})$. However, the algorithm
requires (unnormalized) density evaluations at each iteration, which in certain
settings are not
{% endkatexmm %}

## Setup and notation
{% katexmm %}
We consider drawing samples from a target probability distribution $\mu$ supported
on a state space $\mathcal{X} \subseteq \mathbb{R}^D$ with Borel sigma algebra
$\mathcal{A}$. Let $\pi: \mathcal{X} \to [0,\infty]$ denote the Lebesgue density
of $\mu$, i.e.
$$
\mu(A) = \int_A \mu(d\mathbf{x}) = \int_A \pi(\mathbf{x}) d\mathbf{x}, \qquad \forall A \in \mathcal{A}.
$$
Let $Q$ denote the proposal kernel for the MH algorithm, and $q$ its Lebesgue density.
An iteration of the basic MH algorithm proceeds as follows:
1. Sample $\mathbf{y} \sim Q(\mathbf{x},\cdot)$ and $u \sim \mathcal{U}(0,1)$ independently.
2. Compute the MH acceptance probability:
\begin{align}
\alpha(\mathbf{x},\mathbf{y})
= \min\left(1, \frac{\pi(\mathbf{y})q(\mathbf{y},\mathbf{x})}{\pi(\mathbf{x})q(\mathbf{x},\mathbf{y})}\right)
\end{align}
3. If $u \leq \alpha(\mathbf{x},\mathbf{y})$, set next state to $\mathbf{y}$; else
set to $\mathbf{x}$.

From this we see clearly that each step of the algorithm requires computation of the
(unnormalized) posterior density, in order to compute the acceptance probability.
In the context of Bayesian inverse problems, this means that the expensive forward
model must be run at each iteration. In particular, the model must be run even
if the proposal $\mathbf{y}$ is terrible and will inevitably be rejected.
This seems wasteful, begging the question of whether we might be able to cut down
on the number of required density evaluations while still maintaining the correct
MCMC convergence properties.
{% endkatexmm %}
