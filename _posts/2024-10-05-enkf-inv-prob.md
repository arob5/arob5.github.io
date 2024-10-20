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
such joint Gaussian approximations, providing a clear link to the
EnKF. These methods are characterized by a *finite* sequence of updates that map
the prior distribution to an (approximate) posterior distribution. We then
generalize further by considering a slight adjustment to the EnKF update rules
such that the resulting algorithms consist of an *infinite* sequence of
updates. Practical algorithms are defined by truncating after a finite number
of steps. Algorithms falling into this class have come to be known as
*ensemble Kalman inversion (EKI)*, and have gained much recent interest.
We conclude by briefly highlighting connections to measure transport.

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
the conditional distribution of $u|y$; i.e., the *posterior distribution*. The
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
which is sometimes also called the *potential* or *model-data misfit*.
Throughout most of this post,
we restrict to the choice of a Gaussian likelihood (as in (1)), in which case
\begin{align}
-\log p(y|u)
&= -\log \mathcal{N}(y|\mathcal{G}(u),\Sigma) \newline
&= \frac{1}{2}\log\det(2\pi \Sigma) + \frac{1}{2}\lVert y - \mathcal{G}(u)\rVert^2\_{\Sigma} \tag{4}
\end{align}
and
\begin{align}
&\Phi(u) = \frac{1}{2}\lVert y - \mathcal{G}(u)\rVert^2\_{\Sigma}, &&C = \frac{1}{2}\log\det(2\pi \Sigma). \tag{5}
\end{align}
As above, we will use the notation
$$
\lVert x \rVert^2_A := \langle x, x\rangle_A := x^\top A^{-1}x \tag{5}
$$
to denote the Euclidean norm weighted by the inverse of a positive definite matrix
$A$. The second term in (2), $p(u)$, is called the *prior*. In this post, we will
restrict to the setting $\mathcal{U} \subseteq \mathbb{R}^d$ and assume that the prior
distribution is given by a density $\pi_0(u)$. We denote the resulting posterior
density (the density describing the distribution of $u|y$) by $\pi(u)$.
The methods discussed below
can be generalized to settings where $\mathcal{U}$ may be a function space; e.g.,
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
approximation to the posterior $u|y$. However, borrowing an idea from EnKF
methodology, we will consider a slight modification with the ability to produce
non-Gaussian approximations. To avoid notational confusion between
exact and approximate distributions, we will denote by $(\hat{u}, \hat{y})$
the random variables defining the joint Gaussian approximation. The Gaussian
assumption thus takes the form
\begin{align}
\begin{bmatrix} \hat{u} \newline \hat{y} \end{bmatrix}
\sim
\mathcal{N}\left(
\begin{bmatrix} \overline{u} \newline \overline{y} \end{bmatrix},
\begin{bmatrix} \hat{C} & \hat{C}^{uy} \newline  
                \hat{C}^{yu} & \hat{C}^{y} \end{bmatrix}
\right) \tag{6}
\end{align}
where the moments $\overline{u}$, $\overline{y}$, $\hat{C}$,
$\hat{C}^y$, and $\hat{C}^{uy}$ defining this approximation are quantities that
we must specify. We use the notation $\hat{C}^{yu} := \hat{C}^{uy}$.
Note that if the forward model $\mathcal{G}$ is linear and
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

In the following subsections, we briefly review some properties of joint Gaussians
and their conditional distributions. We work with the joint distribution (6),
assuming the means and covariances are known. With the necessary background
established, we then discuss practical algorithms for estimating these moments
and producing approximations of $u|y$.
{% endkatexmm %}

## Gaussian Conditional Moments
{% katexmm %}
Regardless of whether or not we are truly in the linear Gaussian setting, let
us suppose that we have constructed the joint distribution (6). Using standard
facts about Gaussian distributions, we know the conditional is also Gaussian
$$
\hat{u}|[\hat{y}=y] \sim \mathcal{N}(\hat{m}_*, \hat{C}_*), \tag{7}
$$
with moments given by

$$
\hat{m}_* = \overline{u} + \hat{C}^{uy}[\hat{C}^y]^{-1}(y - \overline{y}) \tag{8}
$$
$$
\hat{C}_* = \hat{C} - \hat{C}^{uy}[\hat{C}^y]^{-1}\hat{C}^{yu}. \tag{9}
$$
{% endkatexmm %}

## Gaussian Conditional Simulation
{% katexmm %}
As an alternative to explicitly computing the conditional moments (8) and (9),
we can consider a Monte Carlo representation of $\hat{u}|\hat{y}$. The conditional
distribution can be directly simulated (without computing (8) and (9)) using the
fact
$$
(\hat{u}|[\hat{y}=y]) \overset{d}{=} \hat{u} + \hat{C}^{uy}[\hat{C}^y]^{-1}(y-\hat{y}), \tag{10}
$$
which can be quickly verified by computing the mean and covariance of each
side. Note that the randomness in the righthand side is inherited from
the random variables $\hat{u}$ and $\hat{y}$, while $y$ here is viewed as
a specific realization of the data (and is thus non-random).
The result (10), known as *Matheron's Rule*, provides the basis for the
following algorithm to draw independent samples from the conditional
distribution $\hat{u}|[\hat{y}=y]$.

<blockquote>
  <p><strong>Gaussian Conditional Simulation via Matheron's Rule.</strong> <br>
  Independent samples $\hat{u}_*^{(1)}, \dots, \hat{u}_*^{(J)}$ can be simulated from the
  distribution $\hat{u}|[\hat{y}=y]$ by repeating the following procedure
  for each $j=1,\dots,J$: <br> <br>

  1. Draw independent samples $\hat{u}^{(j)}$ and $\hat{y}^{(j)}$ from the marginal
  distributions of $\hat{u}$ and $\hat{y}$, respectively. That is,
  \begin{align}
  &\hat{u}^{(j)} \sim \mathcal{N}(\overline{u},\hat{C})
  &&\hat{y}^{(j)} \sim \mathcal{N}(\overline{y},\hat{C}^y). \tag{11}
  \end{align}
  2. Return
  $$
  \hat{u}^{(j)}_* := \hat{u}^{(j)} + \hat{C}^{uy}[\hat{C}^y]^{-1}(y-\hat{y}^{(j)}). \tag{12}
  $$
  </p>
</blockquote>
{% endkatexmm %}

## A Monte Carlo Approach
{% katexmm %}
Up to this point, we have not discussed the choice of the moments defining the
joint Gaussian approximation (6). We now provide these definitions, leading to
concrete algorithms for posterior approximation. We will adopt a Monte Carlo
strategy by sampling independently from the *true* joint distribution $(u,y)$
defined by (1); i.e., $(u^{(j)}, y^{(j)}) \sim p(u,y)$. We next compute the
empirical means and covariances of these random draws and insert them into
the joint Gaussian approximation (6). This subsection focuses on the estimation
of these moments, which we will then follow by utilizing the above Gaussian
conditioning results to derive various approximations of $u|y$.

To be explicit, the first step in our approach requires generating the prior
*ensemble*
$$
\{(u^{(j)}, \epsilon^{(j)})\}, \qquad j = 1, \dots, J \tag{13}
$$
constructed by sampling according to model (1); i.e.,
\begin{align}
&u^{(j)} \sim \pi_0, &&\epsilon^{(j)} \sim \mathcal{N}(0, \Sigma). \tag{14}
\end{align}
We now consider estimating the first two moments of this joint distribution.
Starting with the $u$ marginal, we define the sample estimates
\begin{align}
\overline{u} &:= \frac{1}{J}\sum_{j=1}^{J} u^{(j)} \tag{15} \newline
\hat{C} &:= \frac{1}{J-1}\sum_{j=1}^{J} (u^{(j)}-\overline{u})(u^{(j)}-\overline{u})^\top. \tag{16}
\end{align}
Alternatively, if the prior $\pi_0$ takes the form of a well-known distribution,
then we can simply set $\overline{u}$ and/or $\hat{C}$ to the known moments
of this distribution. We can likewise consider such mean and covariance estimates
for the $\hat{y}$ portion of (6), defined with respect to the ensemble
$$
\{y^{(j)}\}_{j=1}^{J}, \qquad\qquad y^{(j)} := \mathcal{G}(u^{(j)}) + \epsilon^{(j)}. \tag{17}
$$
However, we can simplify matters a bit by performing part of the calculations
analytically, owing to the simple additive Gaussian error structure. Noting that
under (1) we have
\begin{align}
\mathbb{E}[y]
&= \mathbb{E}[\mathcal{G}(u) + \epsilon]
= \mathbb{E}[\mathcal{G}(u)] \tag{18} \newline
\text{Cov}[y]
&= \text{Cov}[\mathcal{G}(u) + \epsilon]
= \text{Cov}[\mathcal{G}(u)] + \text{Cov}[\epsilon]
= \text{Cov}[\mathcal{G}(u)] + \Sigma, \tag{19}
\end{align}
we can focus our efforts on substituting sample-based estimates for the first
term in both (18) and (19). Doing so yields,
\begin{align}
\overline{y} &:= \frac{1}{J} \sum_{j=1}^{J} \mathcal{G}(u^{(j)}) \tag{20} \newline
\hat{C}^y &:= \frac{1}{J-1} \sum_{j=1}^{J} (\mathcal{G}(u^{(j)})-\overline{y})(\mathcal{G}(u^{(j)})-\overline{y})^\top + \Sigma. \tag{21}
\end{align}
We similarly define the last remaining quantity $\hat{C}^{uy}$ by noting that
$$
\text{Cov}[u,y]
= \text{Cov}[u,\mathcal{G}(u)+\epsilon]
= \text{Cov}[u,\mathcal{G}(u)] + \text{Cov}[u,\epsilon]
= \text{Cov}[u,\mathcal{G}(u)]. \tag{22}
$$
We thus consider the estimator
$$
\hat{C}^{uy} := \frac{1}{J-1} \sum_{j=1}^{J} (u^{(j)}-\overline{u})(\mathcal{G}(u^{(j)})-\overline{y})^\top. \tag{23}
$$

## Gaussian Approximations of the Posterior
At this point, we have obtained the joint approximation (6) derived by computing
the sample moments as described in the previous subsection. The question is now
how best to use this joint approximation in approximating the true posterior.
The most straightforward approach is to simply define our approximate posterior
to be $\hat{u}|[\hat{y}=y]$, the Gaussian conditional. This conditional is
available in closed-form and is defined in equations (7), (8), and (9). Suppose,
for whatever reason, we are only interested in a sample-based representation
of this Gaussian conditional. In this case, we can apply the algorithm summarized
in (11) and (12) based on Matheron's Rule. Explicitly, we construct an
approximate posterior ensemble $\{\hat{u}_*^{(j)}\}_{j=1}^{J^\prime}$ via the
update equation in (12),
$$
\hat{u}^{(j)}_* := \hat{u}^{(j)} + \hat{C}^{uy}[\hat{C}^y]^{-1}(y-\hat{y}^{(j)}), \tag{24}
$$
for $j = 1, \dots, J^\prime$. At this point we should be quite clear about the
definition of each term in (24). The mean and covariance estimates appearing in
this expression are all derived from the ensemble
$\{(u^{(j)}, \epsilon^{(j)})\}_{j=1}^{J}$ sampled from the *true* joint distribution
implied by (1). On the other hand, $\{\hat{u}^{(j)}, \hat{y}^{(j)}\}_{j=1}^{J^\prime}$
are samples from the *approximate* joint distribution defined in (6). This
is simply an exact implementation of Matheron's rule under the joint distribution
(6); i.e., the resulting ensemble $\{\hat{u}^{(j)}_*\}_{j=1}^{J^\prime}$ are
iid samples from $\hat{u}|[\hat{y}=y]$. If it seems odd to dwell on this point,
we note that we are primarily doing so to provide motivation for the alternative
method discussed below.

## A Slight Tweak to Matheron's Rule: Beyond Gaussianity
As discussed above, a direct application of Matheron's rule to (6) results in
an ensemble of samples from a Gaussian distribution, which we interpret as
a sample-based approximation to the true posterior. We now consider constructing
the ensemble $\{\hat{u}_*^{(j)}\}_{j=1}^{J}$ via the slightly different
update rule
$$
\hat{u}^{(j)}_* := u^{(j)} + \hat{C}^{uy}[\hat{C}^y]^{-1}(y-y^{(j)}), \tag{25}
$$
for $j = 1, \dots, J$. This is precisely the update equation used in the analysis
(i.e., update) step of the EnKF, so we refer to (25) as the *EnKF update*.
The only difference from (24) is that we have replaced
the samples $(\hat{u}^{(j)},\hat{y}^{(j)})$ from (6) with the samples
$(u^{(j)}, y^{(j)})$ from the true joint distribution, as defined in (14) and
(17). In other words, we now utilize the samples from the initial ensemble (the
same samples used to compute the mean and covariance estimates). Thus, equation
(25) can be viewed as an update to the initial ensemble,
$$
\{u^{(j)}\}_{j=1}^{J} \mapsto \{\hat{u}^{(j)}_*\}_{j=1}^{J}. \tag{26}
$$
The update (25) is somewhat heuristic, and it is difficult to assess the properties
of the updated ensemble. Given that the initial ensemble will not in general
constitute samples from a Gaussian (since $p(u,y)$ is in general non-Gaussian),
then the updated ensemble will also generally be non-Gaussian distributed.
If the inverse problem (1) is linear Gaussian, then the updates (24) and (25)
are essentially equivalent. Beyond this special case, the hope is that the use
of the true prior samples in the (25) will better approximate
non-Gaussianity in the posterior distribution, as opposed to the Gaussian
approximation implied by (24). The EnKF update can alternatively be justified
from an optimization perspective, which we won't discuss here; see my
[post](https://arob5.github.io/blog/2024/07/30/enkf/) on the EnKF for details.
We conclude this section by summarizing the final algorithm.
<blockquote>
  <p><strong>Posterior Approximation via EnKF Update.</strong> <br><br>
  1. Generate the initial ensembles $\{u^{(j)}\}_{j=1}^{J}$ and $\{\mathcal{G}(u^{(j)})\}_{j=1}^{J}$, where $u^{(j)} \sim \pi_0$. <br>
  2. Compute the sample estimates $\overline{u}$, $\overline{y}$,
  $\hat{C}$, $\hat{C}^y$, and $\hat{C}^{uy}$ as defined in (20)-(23). <br>
  3. Return the updated ensemble $\{\hat{u}^{(j)}_*\}_{j=1}^{J}$ by applying the
  EnKF update  
  $$
  \hat{u}^{(j)}_* := u^{(j)} + \hat{C}^{uy}[\hat{C}^y]^{-1}(y-\hat{y}^{(j)}). \tag{27}
  $$
  </p>
</blockquote>

{% endkatexmm %}
