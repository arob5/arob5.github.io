---
title: Importance Sampling
subtitle: I describe the Monte Carlo method of importance sampling and derive some of its properties.
layout: default
date: 2025-02-28
keywords: Comp-Stats, Monte-Carlo, importance-sampling
published: true
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
\hat{P}(\phi) := \frac{1}{N} \sum_{n=1}^{N} \phi(X_n), \qquad X_n \overset{\text{iid}}{\sim} P. \tag{1}
\end{align}
This estimator is unbiased $\mathbb{E}\left[\hat{P}(\phi)\right] = P(\phi)$ and
is justified by the law of large numbers and central limit theorems. However,
in some situations it may be impossible or undesirable to
draw samples from $P$. *Importance sampling* offers a solution to this problem by
instead simulating from some other distribution *Q*, and then re-weighting the
samples to correct for the fact that we sampled from the wrong distribution.
To see how this works, let's try to rewrite the expectation in (1) with
respect to $Q$ instead of $P$. Letting $q(x)$ denote the density of $Q$, we have
$$
P(\phi)
= \mathbb{E}_P \left[\phi(X)\right]
= \int \phi(x) p(x) dx
= \int \frac{\phi(x)p(x)}{q(x)} q(x) dx
= \mathbb{E}_Q \left[\frac{\phi(X)p(X)}{q(X)} \right]
= Q\left(\phi p/q\right) \tag{2}
$$
The first and last inequalities are purely to offer two different notations for
writing the same thing. The key step is in the middle, where we divide and
multiply by $q(x)$ in order to rewrite the expectation with respect to $Q$.
Assuming this was all valid, we could then use the alternative Monte Carlo
estimator to estimate the expectation of interest:

$$
\hat{P}\_Q(\phi) := \frac{1}{N} \sum_{n=1}^{N} \frac{\phi(X_n)p(X_n)}{q(X_n)},
\qquad X_n \overset{\text{iid}}{\sim} Q. \tag{3}
$$
The fact that the estimator $\hat{P}_Q(\phi)$ is unbiased follows directly from
the equality (2). Notice that in order to implement this procedure we must be
able to
1. draw iid samples from $Q$
2. compute pointwise evaluations of the densities $p(x)$ and $q(x)$.

For now, we have swept many details under the rug. We will shortly provide
more rigor, but to provide a bit of a roadmap let's make some initial notes:

**Support issues.** Notice that we need to be careful about dividing by
zero in (2). Loosely speaking, we cannot use a distribution $Q$ such that
$q(x)=0$ when $\phi(x)p(x) \neq 0$. This will hold in particular when
the support of $P$ is contained in the support of $Q$.

**Choosing Q.** Intuitively, it seems that choices of $Q$ that are "similar"
to $P$ would work best. After all, if $Q=P$ then we are just directly
sampling from $P$. There are two directions one could take in designing a good
$Q$: (i.) try to choose the $Q$ that works the best for a specific function
$\phi$; (ii.) try to choose the $Q$ that works best on average for any
choice of $\phi$. Typically, we are interested in estimating multiple
expectations (e.g., computing means and variances). Thus, the latter approach
is typically preferred. Note that $Q$ is typically called the *importance*
or *proposal* distribution.

**Importance Weights.** If we define the function
$$
w(x) := \frac{p(x)}{q(x)}, \tag{4}
$$
then we can rewrite (3) as
$$
\hat{P}_Q(\phi) := \frac{1}{N} \sum_{n=1}^{N} \phi(X_n)w(X_n),
\qquad X_n \overset{\text{iid}}{\sim} Q. \tag{5}
$$
The $w(X_1), \dots, w(X_N)$ are typically called the *importance weights*.
The IS estimate (5) can now be seen to be a weighted mean, where the weights
adjust for the fact that we are sampling from $Q$ instead of $P$. We will see
that the variance of the importance weights plays a key role in determining
the success of the algorithm.
{% endkatexmm %}

## Measure Theoretic Setup
{% katexmm %}
In this section we cast IS in a measure-theoretic framework.
From this point of view, IS can simply be seen as a change of measure.

Let $(\mathcal{X}, \mathcal{B})$ be a measurable space, and $P$ a probability
measure on this space. Let $Q$ be a second probability measure such that
$P$ is absolutely continuous with respect to $Q$, written $P \ll Q$. This means
that
$$
Q(B)=0 \implies P(B)=0, \qquad \forall B \in \mathcal{B}. \tag{6}
$$
This says that $G$ must have mass wherever $P$ has mass; $G$ dominates $P$.
Assuming both of these measures are $\sigma$-finite then the
[Radon-Nikodym](https://en.wikipedia.org/wiki/Radon%E2%80%93Nikodym_theorem)
theorem implies the existence of a $\mathcal{B}$-measurable function
$\frac{dP}{dG}: \mathcal{X} \to [0,\infty)$ satisfying
$$
P(B) = \int_B \frac{dP}{dQ}(x) Q(dx), \qquad \forall B \in \mathcal{B}. \tag{7}
$$
The function $\frac{dP}{dQ}$ is typically called the *Radon-Nikodym derivative*
(or sometimes more loosely, the *density*) of $P$ with respect to $G$.

Notice that (7) immediately yields the analog of (2) for the particular function $\phi(x) := 1_B(x)$, where $1_B(x)$ is the indicator (i.e., characteristic)
function for a measurable set $B$. Indeed,
$$
\mathbb{E}_P[1_B(X)] = P(B) = \int_B \frac{dP}{dQ}(x) Q(dx)
= \int 1_B(x) \frac{dP}{dQ}(x) Q(dx)
= \mathbb{E}_Q\left[1_B(X) \frac{dP}{dQ}(X) \right], \tag{8}
$$
so we see that the expectation of $1_B(X)$ with respect to $X \sim P$ has
been rewritten as an expectation with respect to $Q$; this is a
*change-of-measure*. We see that the Radon-Nikodym derivative $dP/dQ(x)$
adopts the role of the IS weight function $w(x)$ defined in (4). It is a
density describing how to re-weight $Q$ to align with $P$.



{% endkatexmm %}

# Example: Bayesian Inference
