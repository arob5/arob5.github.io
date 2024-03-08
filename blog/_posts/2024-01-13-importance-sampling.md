---
title: Importance Sampling
subtitle: I describe the Monte Carlo method of importance sampling and derive some of its properties.
layout: default
date: 2024-01-13
keywords: Comp-Stats, Monte-Carlo, importance-sampling
published: false
---

## The Basic Idea
{% katexmm %}
Suppose we want to estimate the expectation
\begin{align}
P(\phi) := \mathbb{E}_{P}\left[\phi(X)\right] \tag{1}
= \int \phi(x) p(x) dx
\end{align}

where $\phi$ is some scalar-valued function and the random variable $X \sim P$
has a probability density function $p(x)$. Assuming this integral cannot be
computed in closed-form but that we can draw samples from $P$, a popular approach
to approximate (1) is to apply Monte Carlo methods. In simple Monte Carlo,
we simulate iid draws from $P$ and then approximate (1) with the typical
sample mean estimator:
\begin{align}
\hat{P}(\phi) := \frac{1}{N} \sum_{n=1}^{N} \phi(X_n), \qquad X_n \overset{\text{iid}}{\sim} P.
\end{align}
This estimator is unbiased $\mathbb{E}\left[\hat{P}(\phi)\right] = P(\phi)$ and
is justified by the law of large numbers and central limit theorems. However,
in some situations (examples to come) it may be impossible or undesirable to
draw samples from $P$. *Importance sampling* offers a solution to this problem by
instead simulating from some other distribution *Q*, and then re-weighting the
samples to correct for the fact that we sampled from the wrong distribution.
To see how this works, let's try to re-write the expectation \tag{1} with
respect to $Q$ instead of $P$. Letting $q(x)$ denote the density of $Q$, we have
\begin{align}
P(\phi)
= \mathbb{E}_P \left[\phi(X)\right]
= \int \phi(x) p(x) dx
= \int \frac{\phi(x)p(x)}{q(x)} q(x) dx
= \mathbb{E}_Q \left[\frac{\phi(X)p(X)}{q(X)} \right]
= Q\left(\phi p/q\right) \tag{2}
\end{align}
The first and last inequalities are purely to offer two different notations for
writing the same thing. The key step is in the middle, where we divide and
multiply by $q(x)$ in order to re-write the expectation with respect to $Q$.
Assuming this was all valid, we could then use the alternative Monte Carlo
estimator to estimate the expectation of interest:

\begin{align}
\hat{P}\_Q(\phi) := \frac{1}{N} \sum_{n=1}^{N} \frac{\phi(X_n)p(X_n)}{q(X_n)},
\qquad X_n \overset{\text{iid}}{\sim} Q.
\end{align}
The fact that the estimator $\hat{P}_Q(\phi)$ is unbiased follows directly from
the equality (2). Notice that in order to implement this procedure we must be
able to
1. draw iid samples from $Q$
2. evaluate the densities $p(x)$ and $q(x)$.
{% endkatexmm %}

## Being a Bit More Rigorous
