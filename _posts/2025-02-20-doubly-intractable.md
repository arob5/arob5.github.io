---
title: Doubly Intractable  Posterior Inference
subtitle: Doubly intractable MCMC, auxiliary variable methods, and the exchange algorithm.
layout: default
date: 2025-02-20
keywords: probability, statistics
published: false
---

{% katexmm %}
A typical Bayesian model consists of a joint probability distribution over
the parameter $u$ and data $y$ of the form
$$
p(u,y) = \pi_0(u)L(u) \tag{1}
$$
where $\pi_0(u)$ is the prior density on $u$ and $L(u) = p(y|u)$ the likelihood.
The posterior distribution is then given by
$$
\pi(u) := p(u|y) = \frac{1}{Z}\pi_0(u)L(u) \tag{2}
$$
where $Z$ is a normalizing constant (independent of $u$) that we are not
typically able to compute. Fortunately, common algorithms for posterior
inference such as Markov chain Monte Carlo (MCMC) and variational inference (VI)
only require pointwise evaluations of the *unnormalized* posterior density
$\pi_0(u)L(u)$.

In this post, we consider a class of Bayesian models that adds an additional
difficulty, rendering these standard inference algorithms infeasible.
In particular, we assume a likelihood of the form
$$
L(u) = \frac{f(y; u)}{C(u)}, \tag{3}
$$
such that we can evaluate $f(y;u)$ but not the normalizing function $C(u)$.
We refer to this as a *function* as it depends on the parameter $u$. The
posterior density in this setting becomes
$$
\pi(u) = \frac{1}{ZC(u)}\pi_0(u)f(y;u). \tag{4}
$$
Such setups are known as *doubly intractable* owing to the two quantities
we are unable to compute in (4): $Z$ and $C(u)$. While the former does not pose
a problem for typical inference algorithms, the latter does since it depends on
$u$.


{% endkatexmm %}
