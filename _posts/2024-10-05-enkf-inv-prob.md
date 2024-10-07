---
title: Ensemble Kalman Methodology for Inverse Problems
subtitle: Particle-based Kalman methods for the approximate solution of Bayesian inverse problems.
layout: default
date: 2024-10-05
keywords: UQ
published: true
---

In this post we provide a brief introduction to how ideas related to the
[Ensemble Kalman Filter](https://arob5.github.io/blog/2024/07/30/enkf/) (EnKF)
can be applied to approximate the posterior
distribution of a Bayesian inverse problem. While the EnKF is traditionally
applied to time-varying dynamical systems, we show how to introduce artificial
dynamics into an otherwise static inverse problem in order to port the
EnKF methodology to the inverse problem setting. After defining the inverse
problem formulation, we start simply by demonstrating how combining joint Gaussian
assumptions with Monte Carlo approximations can provide a means of approximate
Bayesian inference. We then generalize this idea by considering a sequence of
such joint Gaussian approximations, which then provides a clear link to the
EnKF. These methods are characterized by a *finite* sequence of updates that map
the prior distribution to an (approximate) posterior distribution. We then
generalize further by considering a slight adjustment to the EnKF update rules
such that the resulting algorithms consist of an *infinite* sequence of
updates. Practical algorithms are defined by truncating after a finite number
of steps. Algorithms falling into this class have come to be known as
*ensemble Kalman inversion (EKI)*, and have gained much recent interest.

# Setup: Bayesian Inverse Problems
{% katexmm %}
In this post, we consider the inverse problem formulation
\begin{align}
y &= \mathcal{G}(u) + \epsilon \tag{1} \newline
u &\sim \pi_0 \newline
\epsilon &\sim \mathcal{N}(0, \Sigma),
\end{align}
whereby the task is to recover a parameter $u \in \mathcal{U}$ from noisy
observations $y \in \mathbb{R}^p$ resulting from the mapping of $u$ through
a forward model $\mathcal{G}: \mathcal{U} \to \mathbb{R}^p$. The Bayesian
approach to inverse problems treats $u$ as a random variable, such that the
statistical model (1) defines a joint distribution over the random vector
$(u,y)$. The solution of the Bayesian inverse problem is then defined to be
the conditional distribution $u|y$; i.e., the *posterior distribution*. The
model (1) defines the joint distribution by separately defining the two components
in the product
$$
p(u,y) = p(y|u)p(u). \tag{2}
$$
We note that the model (1) implicitly assumes a priori independence between the
noise $\epsilon$ and parameter $u$.
The first term $p(y|u)$ in (2), when viewed as a
function of $u$ is called the *likelihood*. For convenience later on, we will
introduce notation for (up to a constant) the negative log-likelihood
$$
\Phi(u) := -\log p(u|y) + C, \tag{3}
$$
which is sometimes also called the *potential*. Throughout most of this post,
we restrict to the choice of a Gaussian likelihood (as in (1)), in which case
\begin{align}
-\log p(y|u)
&= -\log \mathcal{N}(\mathcal{G}(u),\Sigma)
= -\frac{1}{2}\log\det(2\pi \Sigma) - \frac{1}{2}\lVert y - \mathcal{G}(u)\rVert^2_{\Sigma} \tag{4}
\end{align}
and
$$
\Phi(u) = \frac{1}{2}\lVert y - \mathcal{G}(u)\rVert^2_{\Sigma}. \tag{5}
$$
As above, we will use the notation
$$
\lVert x \rVert^2_A := \langle x, x\rangle_A := x^\top A^{-1}x \tag{5}
$$
to denote the Euclidean norm weighted by a positive definite matrix $A$. The
second term in (2), $p(u)$, is called the *prior*. In this post, we will restrict
to the setting $\mathcal{U} \subseteq \mathbb{R}^d$ and assume that the prior
distribution is given by a density $\pi_0(u)$. We denote the resulting posterior
density (the density describing the distribution $u|y$) by $\pi(u)$.
The methods discussed below
can be generalized to settings where $\mathcal{U}$ may be function space; e.g.,
the prior distribution may be given by a Gaussian process.  
{% endkatexmm %}

# Joint Gaussian Conditioning
{% katexmm %}

## Joint Gaussian Assumption
We now begin by considering approximation of the posterior $u|y$ by way of
a certain Gaussian approximation. In particular, we will assume that $(u,y)$
are jointly Gaussian distributed, in which case standard Gaussian conditioning
identities can be implied to yield an approximation of $u|y$. Given that
conditionals of Gaussians are also Gaussian, this approach produces a Gaussian
approximation to the posterior $u|y$. To avoid notational confusion between
exact and approximate distributions, we will denote by $(\hat{u}, \hat{y})$
the random variables defining the joint Gaussian approximation. The Gaussian
assumption thus takes the form
\begin{align}
\begin{bmatrix} \hat{u} \newline \hat{y} \end{bmatrix}
\sim
\mathcal{N}\left(
\begin{bmatrix} \overline{m} \newline \overline{y} \end{bmatrix},
\begin{bmatrix} \hat{C} & \hat{C}^{uy} \newline  
                \hat{C}^{yu} & \hat{C}^{y} \end{bmatrix}
\right) \tag{6}
\end{align}
where the moments $\overline{m}$, $\overline{y}$, $\hat{C}$,
$\hat{C}^y$, and $\hat{C}^{uy}$ defining this approximation are quantities that
we must specify. Note that if the forward model $\mathcal{G}$ is linear and
the prior distribution $\pi_0$ is Gaussian, then the joint Gaussian approximation
(6) is actually exact, and the moments can be computed in closed-form. In other
words, with the moments properly defined,
$(\hat{u},\hat{y}) \overset{d}{=} (u,y)$ and therefore
$(\hat{u}|\hat{y}) \overset{d}{=} (u|y)$; that is, the posterior approximation
is exact. This special case is typically referred to as the *linear Gaussian*
setting, which I discuss in depth in this
[this](https://arob5.github.io/blog/2024/07/03/lin-Gauss/) post. When
$\mathcal{G}$ is nonlinear and/or $\pi_0$ is non-Gaussian, then (6) will truly
be an approximation and the above equalities will not hold.

## Gaussian Conditional Moments
Regardless of whether or not we are truly in the linear Gaussian setting, let
us suppose that we have constructed the joint distribution (6). Using standard
facts about Gaussian distributions, we know the conditional is also Gaussian
$$
\hat{u}|[\hat{y}=y] \sim \mathcal{N}(\hat{m}_*, \hat{C}_*) \tag{7}
$$
with moments given by

\begin{align}
\hat{m}_* &= \overline{m} + \hat{C}^{uy}[\hat{C}^y]^{-1}(y - \overline{y}) \tag{8} \newline
\hat{C}_* &= \hat{C} - \hat{C}^{uy}[\hat{C}^y]^{-1}\hat{C}^{yu}. \tag{9}
\end{align}

## Gaussian Conditional Simulation
As an alternative to explicitly computing the conditional moments (8) and (9),
we can consider a Monte Carlo representation of $\hat{u}|\hat{y}$. The conditional
distribution can be directly simulated (without computing (8) and (9)) using the
fact
$$
(\hat{u}|[\hat{y}=y]) \overset{d}{=} \hat{u} + \hat{C}^{uy}[\hat{C}^y]^{-1}(y-\hat{y}), \tag{10}
$$
which can be quickly verified by computing the mean and covariance of each
side. Note that the randomness in the righthand side is inherited from
the random variables $\hat{u}$ and $\hat{y}$, while $y$ here is non-random
(observed). This fact, known as *Matheron's Rule*, provides the basis for the
following algorithm to draw independent samples from the conditional
distribution $\hat{u}|[\hat{y}=y]$.

<blockquote>
  <p><strong>Gaussian Conditional Simulation via Matheron's Rule.</strong> <br>
  Independent samples $\hat{u}_*^{(1)}, \dots, \hat{u}_*^{(J)}$ can be simulated from the
  distribution $\hat{u}|[\hat{y}=y]$ by repeating the following procedure
  for each $j=1,\dots,J$:

  1. Draw independent samples $\hat{u}^{(j)}$ and $\hat{y}^{(j)}$ from the marginal
  distributions of $\hat{u}$ and $\hat{y}$, respectively. That is,
  \begin{align}
  &\hat{u}^{(j)} \sim \mathcal{N}(\overline{m},\hat{C})
  &&\hat{y}^{(j)} \sim \mathcal{N}(\overline{y},\hat{C}^y). \tag{11}
  \end{align}
  2. Return
  $$
  \hat{u}^{(j)}_* := \hat{u}^{(j)} + \hat{C}^{uy}[\hat{C}^y]^{-1}(y-\hat{y}^{(j)}). \tag{12}
  $$
  </p>
</blockquote>

{% endkatexmm %}
