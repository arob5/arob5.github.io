---
title: Combining Samples From Non-mixing MCMC Chains
subtitle:
layout: default
date: 2024-12-12
keywords:
published: true
---

{% katexmm %}
Consider a typical Bayesian model consisting of a joint data-parameter
distribution $p(y,\theta)$. The goal is to characterize the posterior
distribution $\pi(\theta) := p(\theta|y=y_{\text{obs}})$, where
$y_{\text{obs}}$ is the fixed data realization. In this post, we consider the
setting where $\pi$ may contain many local modes. There are specialty MCMC
algorithms (e.g., using tempering approaches) designed to be able to hop
between modes, but in general sampling from such distributions
is very challenging. An alternative perspective on this problem is to abandon
the hope of designing an algorithm that will generate exact posterior samples
and instead be content with "exploring" the most important regions of the
parameter space. Multiple chains of a standard MCMC algorithm can be run from
different initial conditions, with the hope that collectively the chains will
identify the dominant modes of the posterior, even though individually each may
be stuck in a single mode. This is the perspective taken in
{% cite nonmixingMCMC %}. The authors note the observation that it is often not
difficult to achieve good within-chain mixing adapted to a local region of
the posterior, even when it is near impossible to achieve such mixing with
respect to the whole distribution. In this post, I summarize a few different
approaches used in the literature that adopt this perspective. This is certainly
not a comprehensive review, and I will likely add to this post if I come across
new ideas.

# Setup
Consider the typical problem of estimating the expectation of $\phi(\theta)$,
where $\theta \sim \pi$, for some function $\phi(\cdot)$ (e.g., estimating a
posterior mean or variance). This expectation is given by
$$
\pi(\phi)
:= \mathbb{E}[\phi(\theta)|y=y_{\text{obs}}]
= \int \phi(\theta) \pi(d\theta). \tag{1}
$$
Suppose we run $M$ MCMC chains and obtain samples
$\theta_m := \{\theta_m^{(k)}\}_{k=1}^{K_m}$ for each of the chains
$m = 1, \dots, M$. Here, $K_m$ is the number of samples from the $m^{\text{th}}$
chain and $\theta_m^{(k)}$ is the $k^{\text{th}}$ draw from the $m^{\text{th}}$
chain. We assume that each set of samples $\theta_m$ is well-mixed with respect
to a *local* region of the posterior, and that any burn-in has already been
dropped. We can think of each $\theta_m$ as a summary of one of the dominant
modes of the distribution, but of course in reality it may be that we
have completely missed some modes or multiple chains end up in the same mode.
In any case, the question is now how to use the sets of samples
$\theta_1, \dots, \theta_M$ to estimate (1). The problem is that even if the
chains are well-mixed with respect to each mode, it is not obvious how
to estimate the relative weights of the modes.

# Some Basic Ideas
Perhaps the simplest approach is to simply weight all of the chains equally;
i.e., compute the sample mean for each chain separately and then take an
average of all the chain means. The gives the estimator
$$
\hat{\pi}(\phi) := \frac{1}{M} \sum_{m=1}^{M}
\frac{1}{K_m} \sum_{k=1}^{K_m} \phi(\theta_k^{(m)}). \tag{2}
$$

A natural generalization is to introduce weights $w_1, \dots, w_M$ that
may differ for each chain, yielding
$$
\hat{\pi}(\phi) := \sum_{m=1}^{M}
\frac{w_m}{K_m} \sum_{k=1}^{K_m} \phi(\theta_k^{(m)}). \tag{3}
$$
Note that (2) is a special case of (3) with $w_m := M^{-1}$. The question is
how to choose these weights. In running the MCMC algorithm, we obtain
information about the relative importance of different regions via the
evaluation of the posterior density at each iteration. Let's denote by
$\ell_m := \{\ell_m^{(k)}\}_{k=1}^{K_m}$ and
$\rho_m := \{\rho_m^{(k)}\}_{k=1}^{K_m}$ the
log-likelihood and log (unnormalized) posterior density evaluations from
chain $m$. A natural choice for weights might then be
$w_m \propto \exp\{\bar{\ell}_m\}$ or
$w_m \propto \exp\{\bar{\rho}_m\}$, where $\bar{\ell}_m$ and
$\bar{\rho}_m$ denote the sample means of $\ell_m$ and $\rho_m$,
respectively; i.e.,
$$
\bar{\ell}_m
:= \frac{1}{K_m} \sum_{k=1}^{K_m} \ell_m^{(k)} \tag{4}
$$

If one of the chains got stuck in a local mode in a
negligible region of parameter space, then the log likelihood
(or posterior density) evaluations in that region will be very
small, so $w_m \approx 0$.

# A Simple Heuristic
We now explore a slight generalization of the above idea that was
introduced in {% cite Pritchard2000 %}. The context in the paper
is actually quite different; the heuristic is used to select the
number of clusters in a clustering model. However, the idea nicely applies
to the problem of combining nonmixing MCMC chains. The idea boils down to
treating each chain as corresponding to a different "model" and set each weight
$w_m$ to an estimate of the corresponding marginal likelihood under that
"model". In reality, the only model here is $p(\theta,y)$ and the marginal
likelihood is
$$
p(y_{\text{obs}})
:= p(y=y_{\text{obs}})
= \int p(y=y_{\text{obs}}, \theta) \pi(d\theta) \tag{5}
$$
However, we will think of there being a marginal
likelihood $p(y_{\text{obs}}|\mathcal{M}=m)$ corresponding to each of the $M$
chains. The idea is now to produce an estimate of
$p(y_{\text{obs}}|\mathcal{M}=m)$ using the log-likelihood evaluations
$\ell_m$ and then set the weight $w_m$ to this estimate. Define the log of
this quantity as
$$
\mathcal{L}_m := \log p(y_{\text{obs}}|\mathcal{M}=m), \tag{6}
$$
which we emphasize is random as a function of $\theta \sim \pi$ (see (5)).
Assuming the individual chains are well-mixed, we can view $\ell_m$ as a
set of (correlated) samples from $\mathcal{L}_m$. Computing a Monte Carlo
estimate of $\mathcal{L}_m$ and then exponentiating the result would produce
$w_m \propto \exp\{\bar{\ell}_m\}$, the same weight mentioned in the previous
section. The paper {% cite Pritchard2000 %} considers an alternative
heuristic: assume that $\mathcal{L}_m$ is Gaussian distributed. We estimate
the mean $\bar{\ell}_m$ as in (4), but now also estimate the variance
$$
\hat{\sigma}^2_m :=
\frac{1}{K_m - 1} \sum_{k=1}^{K_m} \left(\ell^{(k)}_m - \bar{\ell}_m \right)^2. \tag{7}
$$
and approximate the distribution of $\mathcal{L}_m$ as
$$
\hat{\mathcal{L}}_m \sim \mathcal{N}\left(\bar{\ell}_m, \hat{\sigma}^2_m\right). \tag{8}
$$
The chain weights are then set to
$$
w_m
\propto \mathbb{E}\left[\exp\left\{\hat{\mathcal{L}}_m\right\}\right]
= \exp\left\{\bar{\ell}_m + \frac{1}{2}\hat{\sigma}^2_m \right\}, \tag{9}
$$
where we have used the expression for the expectation of a log-normal. As
opposed to just using $\bar{\ell}_m$, the weight (9) also takes into account
the variability of the samples in each chain. In a sense, $\bar{\ell}_m$
captures the average height of the mode, while $\hat{\sigma}^2_m$ captures
some notion of its width. This is completely heuristic, but at the very least
it seems reasonable to take into account both the height of the modes, as well
as the volume they occupy in parameter space, in estimating their relative
importance.

# Other resources
- A blog [post](https://yulingyao.com/blog/2019/stacking/) by one of the
authors of {% cite nonmixingMCMC %}.


{% endkatexmm %}
