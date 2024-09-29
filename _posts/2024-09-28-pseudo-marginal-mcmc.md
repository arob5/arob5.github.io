---
title: Pseudo-Marginal MCMC
subtitle: Motivated by Bayesian inference, I introduce the pseudo-marginal approach to MCMC and then discuss why is works from a more generic perspective.
layout: default
date: 2024-09-28
keywords: Probability-Theory, MCMC
published: true
---

Pseudo-marginal Markov chain Monte Carlo (MCMC) is a variant of the Metropolis-Hastings
algorithm that works without the ability to evaluate the unnormalized target
density, so long as an unbiased sample of this density can be obtained for any
input. In this post, we motivate the algorithm by considering a problem of
Bayesian inference where the likelihood function is intractable. We then take
a step back to understand why the algorithm works, and discuss the method from
a more generic and rigorous viewpoint.

# Pseudo-Marginal MCMC for Bayesian Inference
{% katexmm %}
We start by considering a standard problem of Bayesian inference for a parameter
of interest $u \in \mathcal{U}$. Given a prior density $\pi_0(u)$ and
likelihood function $L(u)$, the unnormalized posterior density is then obtained
as the product of these two quantities:
$$
\pi(u) := \pi_0(u) L(u). \tag{1}
$$
With the ability to evaluate this unnormalized density, MCMC algorithms can
be applied to obtain samples from the posterior distribution. However, suppose
we face a situation where $L(u)$ is intractable in the sense that it does not
admit an analytic expression that can be computed for any $u$. Suppose, though,
that we can draw an unbiased sample of the quantity $L(u)$ for any input
$u$; that is,
\begin{align}
&\ell \sim P(u, \cdot), &&\mathbb{E}[\ell] = L(u), \tag{2}
\end{align}
where $P(u,\cdot)$ is a probability measure on the sample space $[0, \infty)$
for each $u \in \mathcal{U}$ (formally, we can think of $P$ as a Markov kernel).
It turns out that this is sufficient to define an MCMC algorithm with target
distribution equal $u$'s posterior. The algorithm that accomplishes
this is referred as *pseudo-marginal MCMC*. A single step of this algorithm
is detailed below.

<blockquote>
  <p><strong>Pseudo-Marginal MCMC.</strong>
  Let $u$ be the current state of the algorithm, with $\ell \sim P(u,\cdot)$
  the associated unbiased likelihood sample. Let $Q$ denote the proposal kernel.
  The next state is then determined as follows. <br>

  1. Propose a new state:
  $$
  \tilde{u} \sim Q(u, \cdot) \tag{3}
  $$
  2. Draw an unbiased likelihood sample at the proposed state:
  $$
  \tilde{\ell} \sim P(\tilde{u}, \cdot) \tag{4}
  $$
  3. With probability
  $$
  \alpha(u,\ell; \tilde{u},\tilde{\ell}) := \min\left(1, \frac{\pi_0(\tilde{u})\tilde{\ell}q(\tilde{u},u)}{\pi_0(u)\ell q(u,\tilde{u})} \right), \tag{5}
  $$
  set the new state to $\tilde{u}$. Else set it to the current state $u$.
  </p>
</blockquote>

Notice that the acceptance probability (5) is the typical Metropolis-Hastings
acceptance probability but with the unbiased likelihood samples $\ell$ and
$\tilde{\ell}$ inserted in place of $L(u)$ and $L(\tilde{u})$, respectively.
The claim is that this algorithm defines a Markov chain with invariant distribution
$\pi$. To see why this is true, the trick is to view the above algorithm as
a Metropolis-Hastings scheme operating on the extended state vector
$(u, \ell)$. In showing this, I will assume $P(u,\cdot)$ and $Q(u,\cdot)$
admit densities $p(u,\cdot)$ and $q(u,\cdot)$ with respect to the same base
measure for which $\pi$ is a density (typically, the Lebesgue or counting measure.)
Now, to view the above algorithm with respect to the extended state space,
start by noticing that (3) and (4) can be interpreted as a joint proposal
$$
(\tilde{u},\tilde{\ell}) \sim \overline{Q}(u,\ell; \cdot, \cdot), \tag{6}
$$
with $\overline{Q}$ a Markov kernel on the product space
$\mathcal{U} \times [0,\infty)$ with density
$$
\overline{q}(u,\ell; \tilde{u},\tilde{\ell}) := q(u,\tilde{u})p(\tilde{u},\tilde{\ell}). \tag{7}
$$
Notice that $\overline{Q}(u,\ell; \cdot, \cdot)$ is independent of $\ell$.
It now remains to write the acceptance probability (5) in a form that can be
interpreted with respect to the extended state space. To this end, consider
\begin{align}
\frac{\pi_0(\tilde{u})\tilde{\ell}q(\tilde{u},u)}{\pi_0(u)\ell q(u,\tilde{u})}
&= \frac{\pi_0(\tilde{u})\tilde{\ell}}{\pi_0(u)\ell}
\cdot \frac{q(\tilde{u},u)p(u,\ell)}{q(u,\tilde{u})p(\tilde{u},\tilde{\ell})}
\cdot \frac{p(\tilde{u},\tilde{\ell})}{p(u,\ell)} \newline
&= \frac{\pi_0(\tilde{u})\tilde{\ell}p(\tilde{u},\tilde{\ell})}{\pi_0(u)\ell p(\tilde{u},\tilde{\ell})}
\cdot \frac{\overline{q}(\tilde{u},\tilde{\ell};u,\ell)}{\overline{q}(u,\ell;\tilde{u},\tilde{\ell})}. \tag{8}
\end{align}
The second term is the proposal density ratio with respect to extended proposal
$\overline{q}$. Thus, the function appearing in the numerator and denominator
of the first term must be the (unnormalized) density targeted by this
Metropolis-Hastings scheme. In other words, the invariant distribution implied
by the above algorithm has unnormalized density
$$
\overline{\pi}(u,\ell) := \pi_0(u)p(u,\ell)\ell. \tag{9}
$$
Notice that $\pi_0(u)\ell$ is the unnormalized density (1) with the sample $\ell$
inserted in place of $L(u)$. This is multiplied by the weight $p(u,\ell)$, which
encodes the probability of sampling $\ell$ at the input $u$. Our proof of the
algorithm's correctness is concluded by noting that $\overline{\pi}$ admits
$\pi$ as a marginal distribution; indeed,
\begin{align}
\int \overline{\pi}(u,\ell)d\ell
&= \int \pi_0(u)p(u,\ell)\ell d\ell
= \pi_0(u) \int \ell \cdot p(u,\ell) d\ell
= \pi_0(u) \mathbb{E}[\ell|u]
= \pi_0(u) L(u), \tag{10}
\end{align}
following from the unbiasedness of the likelihood sample. This means that, in
theory, we can run the above algorithm to obtain joint samples
$(u,\ell) \sim \overline{\pi}$, and then simply extract the $u$ portion of
these pairs to obtain the desired draws $u \sim \pi$.   


{% endkatexmm %}
