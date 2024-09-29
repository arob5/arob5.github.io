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
these pairs to obtain the desired draws $u \sim \pi$. One last thing to note
is that we don't actually need to be able to evaluate the density $p(u,\ell)$
appearing in (8); we see in the acceptance probability (5) that we need only
be able to sample from $P(u,\cdot)$. As usual, we need to be able to evaluate
the density $q(u,\tilde{u})$.

# A More Generic and Formal Perspective
The above idea of course extends beyond the Bayesian example. In this section,
we discuss the pseudo-marginal algorithm from a more generic perspective, and
fill in some of the measure-theoretic details. Let's assume
$\Pi$ is some generic target distribution on a measurable space
$(\mathcal{U}, \mathcal{B}(\mathcal{U}))$. We write $\mathcal{B}(\mathcal{U})$
to denote the Borel $\sigma$-algebra; that is, the $\sigma$-algebra generated
by the open sets of $\mathcal{U}$. We assume $\Pi$ admits a density (i.e.,
Radon-Nikodym derivative) $\pi$ with respect to some reference measure $\nu$.
The density $\pi$ need not be normalized. All densities considered throughout
this section will be with respect to the same reference measure $\nu$.
As before, we consider $\pi(u)$ intractable, but assume we can draw samples from
an unbiased estimator. We could define $P(u,\cdot)$ as before such that samples
drawn from $P(u,\cdot)$ are unbiased with respect to $\pi(u)$. However, note that
this is equivalent to considering samples $w \sim P(u,\cdot)$ with expectation
$1$, such that $w \cdot \pi(u)$ is unbiased for $\pi(u)$. This seems to be
a roundabout way to go around this, but for the purposes of analysis it turns
out to be convenient. This is the definition used in some of the "noisy MCMC"
literature (see, e.g., Medina-Aguayo et al, 2018). Thus, let's go with this
definition and define the Markov kernel $P: \mathcal{U} \to [0,1]$ such that
(1) $P(u,\cdot)$ is a probability measure on $(\mathcal{W},\mathcal{B}(\mathcal{W}))$
for each $u \in \mathcal{U}$, where $\mathcal{W} \subseteq [0,\infty)$; and (2)
$P$ produces weights with unit expectation:
\begin{align}
&w \sim P(u,\cdot), &&\mathbb{E}_{P_u}[w] = 1. \tag{11}
\end{align}
We use $P_u$ as shorthand for $P(u,\cdot)$ in the subscript. We again emphasize
that the sample $w$ from (11) implies that $w\pi(u)$ is an unbiased estimate
of $\pi(u)$. The pseudo-marginal algorithm proceeds exactly as before. We state
it again below to emphasize the new notation.

<blockquote>
  <p><strong>Pseudo-Marginal MCMC.</strong> <br>
  1. Propose a new state:
  $$
  \tilde{u} \sim Q(u, \cdot) \tag{12}
  $$
  2. Draw an unbiased weight sample at the proposed state:
  $$
  \tilde{w} \sim P(\tilde{u}, \cdot) \tag{13}
  $$
  3. With probability
  $$
  \alpha(u,w; \tilde{u},\tilde{w}) := \min\left(1, \frac{\pi(\tilde{u})\tilde{w}q(\tilde{u},u)}{\pi(u)w q(u,\tilde{u})} \right), \tag{14}
  $$
  set the new state to $\tilde{u}$. Else set it to the current state $u$.
  </p>
</blockquote>
Of course, we can't evaluate $\pi(u)$ in (14), but we have just defined
things this was for its theoretical benefits. In practice, we can think of drawing
a sample to directly approximate $\pi(u)$. Similar to before, we can think about this
algorithm as targeting an invariant distribution on the product space
$(\mathcal{U} \times \mathcal{W}, \mathcal{B}(\mathcal{U}) \times \mathcal{B}(\mathcal{W}))$.
The steps (12) and (13) represent a draw from the proposal kernel
$\overline{Q}: \mathcal{U} \times \mathcal{W} \to [0,1]$ defined by
$$
\overline{Q}(u,w; U,W) := \int_{U} P(\tilde{u},W)Q(u,d\tilde{u}), \tag{15}
$$
for $U \in \mathcal{B}(\mathcal{U})$ and $W \in \mathcal{B}(\mathcal{W})$.

{% endkatexmm %}

# References
1. The pseudo-marginal approach for efficient Monte Carlo computations (Andrieu and Roberts, 2009)
2. Convergence properties of pseudo-marginal Markov chain Monte Carlo
algorithms (Andrieu and Vihola, 2015)
3. Stability of Noisy Metropolis-Hastings (Medina-Aguayo et al, 2018)
