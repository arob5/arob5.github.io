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
\hat{P}_Q(\phi) := \frac{1}{N} \sum_{n=1}^{N} \frac{\phi(X_n)p(X_n)}{q(X_n)},
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
$q(x)=0$ when $\phi(x)p(x) \neq 0$. For $Q$ to be valid for different choices
of $\phi$, then it must satisfy $q(x) > 0$ whenever $p(x) > 0$; i.e.,
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
The $w(X_1), \dots, w(X_N)$ are typically called the *importance weights*, or
sometimes the *likelihood ratio*.
The IS estimate (5) can now be seen to be a weighted mean, where the weights
adjust for the fact that we are sampling from $Q$ instead of $P$. We will see
that the variance of the importance weights plays a key role in determining
the success of the algorithm.
{% endkatexmm %}

## Measure Theoretic Framework
{% katexmm %}
In this section we cast IS in a measure-theoretic framework.
From this point of view, IS can simply be seen as a change of measure.

### Setup
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
*change-of-measure*. We also see that the Radon-Nikodym derivative $dP/dQ(x)$
adopts the role of the IS weight function $w(x)$ defined in (4). It is a
density describing how to re-weight $Q$ to align with $P$.

### Importance sampling justification
We saw in (8) that the main IS equality fell right out of the Radon-Nikodym
theorem for the choice $\phi(x) = 1_B(x)$. To establish the main IS result we
must generalize this to any $\mathcal{B}$-measurable function
$\phi: \mathcal{X} \to \mathbb{R}$. The derivation is straightforward, and
I will make use of the
[Dirac measure](https://en.wikipedia.org/wiki/Dirac_measure)
$\delta_x(B) := 1_B(x)$. Note that we can rewrite (7) using the Dirac measure as
$$
P(B) = \int_B \frac{dP}{dQ}(y) Q(dy)
= \int 1_B(y) \frac{dP}{dQ}(y) Q(dy)
= \int \delta_y(B) \frac{dP}{dQ}(y) Q(dy), \tag{9}
$$
and thus
$$
P(dx) = \int \delta_y(dx) \frac{dP}{dQ}(y) Q(dy). \tag{10}
$$
We therefore have,
\begin{align}
\mathbb{E}_P[\phi(X)]
= \int \phi(x) P(dx)
&= \int \phi(x) \left[\int \delta_y(dx) \frac{dP}{dQ}(y) Q(dy)\right] \newline
&= \int \int \phi(x) \frac{dP}{dQ}(y)\delta_y(dx)Q(dy) \newline
&= \int \frac{dP}{dQ}(y) \left[\phi(x)\delta_y(dx) \right] Q(dy) \newline
&=\int \frac{dP}{dQ}(y) \phi(y) Q(dy) \newline
&= \mathbb{E}_Q\left[\phi(X)\frac{dP}{dQ}(X)\right]. \tag{11}
\end{align}

I'm not providing full justification for all of the steps here, but the
second, third, and fourth equalities can be justified via Fubini's theorem. The penultimate step follows from the fact the integral of a function with respect
to $\delta_y$ simply yields the function evaluated at $y$. An equivalent way
to write this result is
$$
P(\phi) = Q(\phi dP/dQ). \tag{12}
$$

### Common Dominating Measure
To wrap up the more formal specification, we connect the
measure-theoretic statement (11) back to the typical density formulation
(2). The latter statement is really a special case where both $P$ and $Q$
admit densities with respect to a common dominating measure $\lambda$.
In common applications, $\lambda$ is either the Lebesgue or counting measure,
corresponding to the typical continuous and discrete settings.
In addition to the assumption $P \ll Q$, we now add the additional
assumption $Q \ll \lambda$ which implies
$$
P \ll Q \ll \lambda. \tag{13}
$$

With this additional assumption, we can simplify (11) as
\begin{align}
P(\phi)
&= \int \phi(y) \frac{dP}{dQ}(y) Q(dy) \newline
&= \int \phi(y) \frac{dP}{dQ}(y) \frac{dQ}{d\lambda}(y) \lambda(dy) \newline
&= \int \phi(y) \left(\frac{dP}{dQ}\frac{dQ}{d\lambda}\right)(y) \lambda(dy), \tag{14}
\end{align}

where we have simply first applied the assumption $P \ll Q$ and then the
assumption $Q \ll \lambda$. Note that we have now written the expectation
$P(\phi)$ with respect to the dominating measure $\lambda$. The final line in
the above derivation implies that
$$
\frac{dP}{d\lambda} = \frac{dP}{dQ}\frac{dQ}{d\lambda}, \tag{15}
$$
so
$$
\frac{dP}{dQ} = \frac{dP/d\lambda}{dQ/d\lambda},
\qquad \text{when} \frac{dQ}{d\lambda} \neq 0. \tag{16}
$$
Let's denote the Radon-Nikodym derivatives in the numerator and denominator
of (16) by $p(x) := dP/d\lambda(x)$ and $q(x) := dQ/d\lambda(x)$. When
$\lambda$ is the Lebesgue measure then $p(x)$ and $q(x)$ are precisely what
we colloquially refer to as "densities". Continuing from (14), we have
\begin{align}
P(\phi)
&= \int \phi(y)\frac{dP}{dQ}(y)q(y) \lambda(dy) \newline
&= \int \phi(y) \frac{p(y)}{q(y)} q(y) \lambda(dy) \newline
&= Q(\phi p/q), \tag{17}
\end{align}
which is precisely the statement (2) when $\lambda$ is the Lebesgue
density. Throughout the rest of this post, we will work with $p(x)$ and
$q(x)$ and write integrals with respect to $dx$, consistent with the Lebesgue
density case. However, note that one can think of $p(x)$ and $q(x)$ as
densities with respect to some other dominating measure $\lambda$.
{% endkatexmm %}

# Basic Properties


# Unnormalized Densities
{% katexmm %}
In many settings (e.g., Bayesian inference) we only know the target density
up to a normalizing constant; i.e,
$$
p(x) = \frac{\tilde{p}(x)}{Z_p}, \tag{18}
$$
where $Z_p$ is intractable. Similarly, we might want to work with a proposal
density $q(x)$ that is also only known up to a normalizing constant:
$$
q(x) = \frac{\tilde{q}(x)}{Z_q}. \tag{19}
$$
It is natural to ask if the IS is still applicable in this setting. Notice that
the IS estimate from (3) becomes
$$
\hat{P}_Q(\phi)
= \frac{1}{N} \sum_{n=1}^{N} \frac{\phi(X_n)p(X_n)}{q(X_n)},
= \frac{Z_q}{Z_p} \frac{1}{N} \sum_{n=1}^{N} \frac{\phi(X_n)\tilde{p}(X_n)}{\tilde{q}(X_n)}, \tag{20}
$$
which we cannot compute without knowing $Z_p$ and $Z_q$ (unless $Z_p=Z_q$).
Thus, the basic
method presented above does not apply to this setting. We thus seek to extend
the IS estimate to handle the unnormalized case. We first define the
extension, and then provide two different interpretations.

<blockquote>
  <p><strong>Self-Normalized Importance Sampler.</strong> <br>
  The self-normalized importance sampling (SNIS) estimator for the expectation
  $P(\phi)$ with target density $p(x)=\tilde{p}(x)/Z_p$ and proposal density
  $q(x)=\tilde{q}(x)/Z_q$ is given by
  $$
  \bar{P}_Q(\phi) := \sum_{n=1}^{N} \phi(X_n)\bar{w}(X_n), \qquad
  X_n \overset{\text{iid}}{\sim} Q, \tag{21}
  $$
  where
  \begin{align}
  &\bar{w}(X_n) := \frac{\tilde{w}(X_n)}{\sum_{n=1}^{N} \tilde{w}(X_n)},
  &&\tilde{w}(X_n) := \frac{\tilde{p}(X_n)}{\tilde{q}(X_n)}. \tag{22}
  \end{align}
  </p>
</blockquote>

Notice that the $\tilde{w}(X_n)$ are the weights computed with the unnormalized
densities. These weights are then normalized to sum to one, yielding the
normalized weights $\bar{w}(X_n)$ in (21).

## Perspectives on the SNIS Estimator
We now provide a couple of different interpretations of (21).

**Ratio Estimator**. One way we could derive (21) is to consider a ratio
estimator
$$
\mathbb{E}_P[\phi(X)]
= \frac{\mathbb{E}_P[\phi(X)]}{1}
\approx \frac{\hat{P}_Q(\phi)}{\hat{P}_Q(1)}, \tag{23}
$$
where the $1$ in $\hat{P}_Q(1)$ refers to the function that is identically
$1$; $\phi(x) \equiv 1$. We have written the expectation $P(\phi)$ as a ratio
and plugged in IS estimates for the numerator and denominator. The motivation
for this is that the intractable normalizing constants cancel in the ratio:

\begin{align}
\frac{\hat{P}\_Q(\phi)}{\hat{P}\_Q(1)} = \frac{\frac{1}{N} \sum_{n=1}^{N} \frac{\phi(X_n)p(X_n)}{q(X_n)}}{\frac{1}{N} \sum_{n=1}^{N} \frac{p(X_n)}{q(X_n)}} \tag{24}
\end{align}

**Estimating the normalizing constant.** We can also view (21) as a modification
of (20), whereby an IS estimator is inserted in place of $Z_q/Z_p$.
Indeed, notice that
\begin{align}
\frac{1}{N} \sum_{n=1}^{N} \tilde{w}(X_n)
= \frac{Z_q}{Z_p} \frac{1}{N} \sum_{n=1}^{N} \frac{p(X_n)}{q(X_n)}
\overset{N \to \infty}{\to} \frac{Z_q}{Z_p}. \tag{25}
\end{align}


{% endkatexmm %}

# Approximating the Target Distribution with Dirac Deltas


# Example: Bayesian Inference
