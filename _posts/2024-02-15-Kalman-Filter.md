---
title: The Kalman Filter
subtitle: I discuss the Kalman filter from both the probabilistic and optimization perspectives, and provide multiple derivations of the Kalman update.
layout: default
date: 2024-02-15
keywords: Bayes, Filtering, Data-Assim
published: true
---

Originally introduced in R.E. Kalman's seminal 1960 paper, the Kalman filter has
found a myriad of applications in fields as disparate as robotics and economics.
In short, the Kalman filter (KF) gives the closed-form solutions to the filtering
problem (see my previous
[post](https://arob5.github.io/blog/2024/01/29/Bayesian-filtering/)) defining the
filtering problem) in the linear-Gaussian setting; that is, the
deterministic dynamic model and observaton operator are each linear and subject
to additive Gaussian noise, and the initial condition is also Gaussian.
As in other models with Gaussian noise, the KF
can alternatively be viewed as an optimization algorithm, which is optimal
in the sense of minimizing the quadratic loss. Indeed, this optimization
perspective was the one originally considered by Kalman in the original paper.
In this post I begin by deriving the KF equations in the Bayesian statistical
setting in two different ways. The two derivations produce two different sets
of equations, which I show are equivalent through an application of the Woodbury
matrix identity. I then proceed to the optimization perspective, which is
interesting in its own right as well as providing motivation for more complex
algorithms used in the nonlinear setting.   

## The Model and Problem
{% katexmm %}
### State Space Model
The setting for the KF is the discrete-time, linear Gaussian state space model
\begin{align}
v_{k+1} &= Gv_k + \eta_{k+1} && \eta_{k+1} \sim \mathcal{N}(0, Q) \tag{1} \newline
y_{k+1} &= Hv_{k+1} + \epsilon_{k+1}, && \epsilon_{k+1} \sim \mathcal{N}(0, R) \newline
v_0 &\sim \mathcal{N}(m_0, C_0) \newline
\\{\epsilon_k\\} &\perp \\{\eta_k\\} \perp v_0
\end{align}
with states $v_k \in \mathbb{R}^d$, observations $y_k \in \mathbb{R}^n$,
forward dynamics operator $G \in \mathbb{R}^{d \times d}$, and observation
operator $H \in \mathbb{R}^{n \times d}$. The final line concisely encodes
the key conditional independence assumptions. We will use the notation
$Y_k = \{y_1, \dots, y_k\}$ to denote the collection of observations through
time $k$.

### The Filtering Problem
Let $\mu_k$ denote the conditional distribution $v_k|Y_k$, which we call the
*filtering distribution* at time $k$, which
admits a Lebesgue density $\pi_k(v_k) := p(v_k|Y_k)$. With this notation established
we can state the overarching goal, which is two-fold:
1. Characterize the filtering distribution $\mu_k$, and
2. Recursively update this characterization as the time index is stepped forward.

What it means to "characterize" a distribution can vary, and often entails some
sort of approximation to the true distribution. However,
the linear Gaussian setting is special in that the filtering distributions
are available in closed-form. The derivation of the KF thus entails performing
analytical calculations to derive the closed-form distribution $\mu_{k+1}$
as a function $\mu_k$.

## The Kalman Filter Equations
As we will shortly show, the filtering distributions in the linear Gaussian
setting are themselves Gaussian, and we will denote their densities by
$\pi_k(v_k) = \mathcal{N}(v_k|m_k, C_k)$. The problem of determining the update  
$\pi_k(v_k) \to \pi_{k+1}(v_{k+1})$ thus simplifies to that of establishing
recursions for the filtering mean and covariance; i.e., $m_k \to m_{k+1}$ and
$C_k \to C_{k+1}$. The following propositions summarize the main result, providing
these recursions in two different (but equivalent) forms. These filtering equations
are broken into two stages, following the *forecast* and *analysis* steps
that I discussed in [this](https://arob5.github.io/blog/2024/01/29/Bayesian-filtering/)
post. Derivations of these equations are given in the subsequent sections.

<blockquote>
  <p><strong>Proposition.</strong>
  Given the state space model (1), the forecast and filtering
  distributions are both Gaussian
  \begin{align}
  &v_{k+1}|Y_{k} \sim \mathcal{N}(\hat{m}_{k+1}, \hat{C}_{k+1}),
  &&v_{k+1}|Y_{k+1} \sim \mathcal{N}(m_{k+1}, C_{k+1})
  \end{align}
  with the forecast mean and covariance given by
  \begin{align}
  \hat{C}_{k+1} &= G C_k G^\top + Q \tag{2} \newline
  \hat{m}_{k+1} &= Gm_k.
  \end{align}
  The recursions for the filtering mean and covariance are given below in two
  equivalent forms.
  </p>
</blockquote>

<blockquote>
  <p><strong>Filtering Equations: State Space.</strong>
  The mean and covariance of the filtering distribution can be written as
  \begin{align}
  C_{k+1} &= \left(H^\top R^{-1} H + \hat{C}^{-1}_{k+1}\right)^{-1} \tag{3} \newline
  m_{k+1} &= C_{k+1}\left(H^\top R^{-1}y_{k+1} + \hat{C}^{-1}_{k+1} \hat{m}_{k+1} \right).
  \end{align}
  We refer to this as the state space representation due to the fact that the primary
  linear algebra computations (most notably, the matrix inversion) are performed in
  $\mathbb{R}^d$.
  </p>
</blockquote>

<blockquote>
  <p><strong>Filtering Equations: Data Space.</strong>
  The mean and covariance can alternatively be written as
  \begin{align}
  C_{k+1} &= \left(I - K_{k+1}H\right)\hat{C}_{k+1} \tag{4} \newline
  m_{k+1} &= \hat{m}_{k+1} + K_{k+1}(y_{k+1} - H\hat{m}_{k+1}),
  \end{align}
  where
  $$
  K_{k+1} := \hat{C}_{k+1}H^\top \left(H\hat{C}_{k+1}H^\top + R\right)^{-1}.
  $$
  We refer to this as the data space representation since the primary computations
  are performed in $\mathbb{R}^n$.
  </p>
</blockquote>


### Investigating the Formulas
Before diving into the derivations, we note some initial properties on the KF
formulas (3). We begin by noting that the forecast mean update
$m_k \mapsto \hat{m}_{k+1}$ is linear, and the analysis update
$\hat{m}_{k+1} \mapsto m_{k+1}$ is affine; thus, the composition of the two
steps defining the map $m_k \mapsto m_{k+1}$ is also affine. We similarly see
that the covariance forecast update $C_k \mapsto \hat{C}_{k+1}$ is affine,
while the analysis update $\hat{C}_{k+1} \mapsto C_{k+1}$ is nonlinear (due
to the inverse). Viewed in terms of the precision matrix, the map
$\hat{C}^{-1}_{k+1} \mapsto C^{-1}_{k+1}$ is affine. The forecast formulas
are quite straightforward, as they simply represent the forward propagation
of uncertainty through the linear dynamics model (see the derivation in the
next section). The analysis formulas are a bit more interesting. The mean
update can be viewed as a weighted average of the data $y_{k+1}$ and the
prior mean $\hat{m}_{k+1}$, with the weights determined by observation and
forecast precision matrices, relative to the covariance of the filtering
distribution. In the
one-dimensional case where the covariance matrices are scalars, then the weights
reduce to the fraction of these precisions over the total precision of the
filtering distribution (the data precision is also affected by $H$, but the
interpretation is basically the same). The analysis precision matrix can
similarly be viewed as a weighted average of the data and forecast precisions.
Importantly, notice
that the covariance analysis update does not depend on the data $y_{k+1}$; the
uncertainty in the filtering distribution is not at all affected by the
sequence of observations. This may seem odd at first, but this property is
shared by all linear Gaussian models, frequentist and Bayesian alike
(recall that in a basic linear regression model with Gaussian noise the
standard error of the coefficient estimator similarly is not a function of the
observed response).

TODO: note that can generalize to time-varying Q and R, complexity, pos-def,
weighted sum.

## Bayesian Derivation
We begin with an approach aligns directly with the derivations of the generic
forecast and analysis updates derived in the previous
[post](https://arob5.github.io/blog/2024/01/29/Bayesian-filtering/). These generic updates
are typically analytically intractable, but in the linear Gaussian setting
all of the calculations go through neatly in closed-form. The derivation
proceeds inductively; note that the base case is provided by the initial condition
$v_0 \sim \mathcal{N}(m_0, C_0)$. The inductive hypothesis assumes that
$v_k|Y_k \sim \mathcal{N}(m_k, C_k)$ and it remains for us to show that
$v_{k+1}|Y_{k+1} \sim \mathcal{N}(m_{k+1}, C_{k+1})$ with the mean and covariance
formulas as given in (1).

### Forecast
As an intermediate step we start by obtaining the forecast distribution
$v_{k+1}|Y_k$. This distribution becomes clear when we consider that $v_{k+1}$
is given by
\begin{align}
v_{k+1} &= Gv_k + \eta_{k+1} \newline
v_k|Y_k &\sim \mathcal{N}(m_k, C_k) \newline
\eta_{k+1} &\sim \mathcal{N}(0, Q),
\end{align}
where $v_k|Y_k$ and $\eta_{k+1}$ are independent. Hence, conditional on $Y_k$,
$Gv_k$ is also Gaussian and so is the sum $Gv_k + \eta_{k+1}$. We thus have
$v_{k+1}|Y_k \sim \mathcal{N}(\hat{m}_{k+1}, \hat{C}_{k+1})$ with mean
and covariance given by
\begin{align}
\hat{m}\_{k+1} &= \mathbb{E}[v\_{k+1}|Y_k] = G\mathbb{E}[v_k|Y_k] + \mathbb{E}[\eta\_{k+1}|Y_k] = Gm_k \newline
\hat{C}\_{k+1} &= \text{Cov}\left[v\_{k+1}|Y_k \right] = G\text{Cov}\left[v_k|Y_k \right]G^\top +
\text{Cov}\left[\eta\_{k+1}|Y_k \right] = GC_kG^\top + Q,
\end{align}
using the linearity of $G$ and the independence of $v_k$ and $\eta_{k+1}$, conditional
on $Y_k$. This verifies the equations in (2).

### Analysis
We recall that the analysis step of the filtering update corresponds to an
application of Bayes' rule,

\begin{align}
\pi_{k+1}(v_{k+1})
&\propto p(y_{k+1}|v_{k+1}) \hat{\pi}\_{k+1}(v_{k+1}) \newline
&= \mathcal{N}(y_{k+1}|Hv_{k+1}, R)\mathcal{N}(v_{k+1}|\hat{m}\_{k+1}, \hat{C}\_{k+1}),    
\end{align}
and thus deriving $\pi_{k+1}(v_k)$ amounts to deriving the posterior distribution
in a linear Gaussian regression model. To do so, we write out the Gaussian densities,
suppressing terms without $v_{k+1}$ dependence, and do a bit of algebra.
\begin{align}
\pi_{k+1}(v_{k+1})
&\propto \mathcal{N}(y_{k+1}|Hv_{k+1}, R)\mathcal{N}(v_{k+1}|\hat{m}\_{k+1}, \hat{C}\_{k+1}) \newline
&\propto \exp\left(-\frac{1}{2}(y_{k+1}-Hv_{k+1})^\top R^{-1}(y_{k+1}-Hv_{k+1})\right)
\exp\left(-\frac{1}{2}(v_{k+1}-\hat{m}\_{k+1})^\top \hat{C}\_{k+1}^{-1}(v_{k+1}-\hat{m}\_{k+1})\right) \newline
&\propto \exp\left(-\frac{1}{2}\left[v_{k+1}^\top(H^\top R^{-1}H + \hat{C}\_{k+1}^{-1})v_{k+1}
-2v_{k+1}^\top(H^\top R^{-1}y_{k+1} + \hat{C}\_{k+1}^{-1}\hat{m}\_{k+1})\right] \right)
\end{align}
Since $\pi_{k+1}(v_{k+1})$ is proportional to the exponential of an expression
which is quadratic in $v_{k+1}$, we immediately know that it must be Gaussian.
To find the mean $m_{k+1}$ and covariance $C_{k+1}$ of this Gaussian, we set
the term in square brackets equal to

\begin{align}
(v_{k+1} - m_{k+1})^\top C_{k+1}^{-1}(v_{k+1} - m_{k+1})
&= v_{k+1}^\top C_{k+1}^{-1}v_{k+1} - 2v_{k+1}^\top C_{k+1}^{-1}m_{k+1} +
m_{k+1}^\top C_{k+1}^{-1} m_{k+1}
\end{align}

and equate like terms. Doing so yields
\begin{align}
C_{k+1} &= \left(H^\top R^{-1}H + \hat{C}\_{k+1}^{-1}\right)^{-1} \newline
m_{k+1} &= C_{k+1}\left(H^\top R^{-1}y_{k+1} + \hat{C}\_{k+1}^{-1}\hat{m}_{k+1}\right),
\end{align}
which completes the derivation of (3). In another
[post](https://arob5.github.io/blog/2024/07/03/lin-Gauss/) I derive the posterior
distribution for generic linear Gaussian inverse problems; we could have skipped
the above derivations by simply applying this result.

## A Second Derivation of the Analysis Step: The Joint Gaussian Approach  
We now explore a second derivation of the KF equations, which yield the second
form of the equations presented in (4). The key observation here
is that the joint distribution of $(v_{k+1},y_{k+1})|Y_k$ is Gaussian
\begin{align}
\begin{pmatrix} v_{k+1} \newline y_{k+1} \end{pmatrix}\bigg|Y_k \tag{5}
\sim \mathcal{N}\left(\begin{pmatrix} \hat{m}\_{k+1} \newline H\hat{m}_{k+1} \end{pmatrix},
  \begin{pmatrix} \hat{C}\_{k+1} & \hat{C}^{vy}\_{k+1} \newline
  \hat{C}^{yv}\_{k+1} & \hat{C}^y\_{k+1} \end{pmatrix} \right),
\end{align}
where we have denoted $\hat{C}^{vy}_{k+1} = \text{Cov}[v_{k+1},y_{k+1}|Y_k]$,
$\hat{C}^{yv}_{k+1} = \left(\hat{C}^{vy}_{k+1}\right)^\top$, and
$\hat{C}^y_{k+1} = \text{Cov}[y_{k+1}|Y_k]$ (see the appendix for the proof that
this joint distribution really is Gaussian). We have already derived the
expressions for $\hat{m}_{k+1}$ and $\hat{C}_{k+1}$ in the previous section
so we need only focus on $\hat{C}^{vy}_{k+1}$ and $\hat{C}^y_{k+1}$ here.
Utilizing the conditional independence assumptions in the state space model we
have,

\begin{align}
\hat{C}^{vy}\_{k+1} = \text{Cov}[v_{k+1},y_{k+1}|Y_k]
&= \text{Cov}[v_{k+1},Hv_{k+1} + \epsilon_{k+1}|Y_k] \newline
&= \text{Cov}[v_{k+1},Hv_{k+1}|Y_k] + \text{Cov}[v_{k+1},\epsilon_{k+1}|Y_k] \newline
&= \text{Cov}[v_{k+1},v_{k+1}|Y_k]H^\top \newline
&= \hat{C}_{k+1}H^\top
\end{align}

and

\begin{align}
\hat{C}^{y}\_{k+1} = \text{Cov}[y_{k+1},y_{k+1}|Y_k]
&= \text{Cov}[Hv_{k+1} + \epsilon_{k+1},Hv_{k+1} + \epsilon_{k+1}|Y_k] \newline
&= \text{Cov}[Hv_{k+1},Hv_{k+1}|Y_k] + 2\text{Cov}[Hv_{k+1},\epsilon_{k+1}|Y_k] +
\text{Cov}[\epsilon_{k+1}|Y_k] \newline
&= H \hat{C}_{k+1} H^\top + R.
\end{align}

With the specification of this joint distribution complete, we note that the
filtering distribution we care about,
$v_{k+1}|Y_{k+1} = v_{k+1}|y_{k+1},Y_k$ is a conditional
distribution of (5); in particular, it is the conditional distribution
resulting from conditioning on the final $n$ dimensions. Applying the
closed-form [equation](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
for Gaussian conditionals, we conclude that
$v_{k+1}|Y_{k+1} \sim \mathcal{N}(m_{k+1}, C_{k+1})$, where

\begin{align}
m_{k+1} &= \hat{m}\_{k+1} + \hat{C}^{vy}\_{k+1}\left[\hat{C}^y\_{k+1}\right]^{-1}(y_{k+1} - H\hat{m}\_{k+1}) \tag{6} \newline
C_{k+1} &= \hat{C}_{k+1} - \hat{C}^{vy}\_{k+1}\left[\hat{C}^y\_{k+1}\right]^{-1}\hat{C}^{yv}.
\end{align}

Typically, these equations are written in terms of the $d \times n$
**Kalman gain** matrix

\begin{align}
K\_{k+1}
&:= \hat{C}^{vy}\_{k+1} \left[\hat{C}^y\_{k+1}\right]^{-1}
= \hat{C}\_{k+1}H^\top \left(H\hat{C}\_{k+1}H^\top + R\right)^{-1}, \tag{7}
\end{align}

which gives

\begin{align}
m\_{k+1} &= \hat{m}\_{k+1} + K\_{k+1}(y\_{k+1} - H\hat{m}\_{k+1}) \tag{8} \newline
C\_{k+1} &= \hat{C}\_{k+1} - K\_{k+1}\hat{C}^{yv}\_{k+1}
= \left(I - K\_{k+1}H\right)\hat{C}_{k+1},
\end{align}
precisely the updates given in (4).
Note that inserting the formulas for $\hat{m}_{k+1}$ and $\hat{C}_{k+1}$ provides the
equations defining the complete maps
$m_k \mapsto m_{k+1}$ and $C_k \mapsto C_{k+1}$.
{% endkatexmm %}

## Understanding the Kalman Gain
<blockquote>
  <p><strong>Proposition: Different Expressions for the Kalman Gain.</strong>
  The Kalman gain
  $$
  K_{k+1} := \hat{C}_{k+1}H^\top \left(H\hat{C}_{k+1}H^\top + R\right)^{-1} \tag{9}
  $$
  admits the equivalent expressions
  $$
  K_{k+1} = \hat{C}^{vy}_{k+1}\left(\hat{C}^{y}_{k+1} \right)^{-1} \tag{10}
  $$
  and
  $$
  K_{k+1} = C_{k+1}H^\top R^{-1}. \tag{11}
  $$
  </p>
</blockquote>
The equivalence between (9) and (10) was already established in (7), and the
equivalence of (11) is proved in the appendix.


## The Optimization Perspective
{% katexmm %}
The preceding derivations adopt a Bayesian perspective, with $m_k$ and $C_k$
interpreted as the mean and covariance of the posterior distribution
$p(v_k | Y_k)$. We now investigate an alternative view in which $m_k$ is
interpreted as the solution to a regularized optimization problem. In this section
we will only concern ourselves with point estimates of the states, with the
covariances playing the role of weights in the objective functions. Throughout
this section we will utilize the following notation for the inner product and
norm weighted by a positive definite matrix $C$:
\begin{align}
&\langle u, v \rangle_{C} := \langle C^{-1}u, v\rangle,
&& \lVert u \rVert_{C} := \lVert C^{-1/2}u \rVert_2 \tag{9}
\end{align}
with the inner product and norm on the righthand side being of the standard
Euclidean variety. We will also let $\mathcal{R}(C)$ denote the *range*
(i.e., *column space*) of a matrix $C$.

<blockquote>
  <p><strong>Tikhonov Regularized Optimization Perspective.</strong>
  The filtering mean $m_k$ solves the optimization problem
  $$
  m_k
  := \text{argmin}_{v \in \mathbb{R}^d} \left\{\frac{1}{2} \lVert y_k - Hv \rVert^2_{R} + \frac{1}{2} \lVert v - \hat{m}_k \rVert^2_{\hat{C}_k}\right\}. \tag{10}
  $$
  </p>
</blockquote>
The first term encourages agreement with the observation, while the second
promotes agreement with the forecast mean. These two objectives are weighted
by the observation covariance $R$ and forecast covariance $\hat{C}_k$,
respectively. These covariances encode the uncertainty in the observations and
model forecasts. The first term in the objective is a typical $L_2$ regression
objective, while the second can be viewed as a regularization term of the
Tikhonov variety (also referred to as ridge regression). The proof of (10)
is immediate upon noticing that
(10) defines the maximum a posteriori (MAP) estimate of $\pi_{k}$. We already
know that $\pi_{k}$ is Gaussian and hence its mean agrees with its mode.
Therefore, the mean $m_k$ is indeed the optimizer. To make the connection more
explicit, observe that
$$
\pi_k(v)
\propto \mathcal{N}(y_k|Hv, R)\mathcal{N}(v|\hat{m}_k, \hat{C}_k)
\propto \exp\left\{-\frac{1}{2}\lVert y_k - Hv\rVert^2_{R} \right\}
\exp\left\{-\frac{1}{2}\lVert v - \hat{m}_k \rVert^2_{\hat{C}_k} \right\}.
$$
Negating the above expression and taking the logarithm gives the objective
function in (10). Note that the normalizing constants in the two
Gaussian densities being multiplied do not depend on $v$ and are thus absorbed
in the proportionality sign. For more details on this perspective, see my
(post)[https://arob5.github.io/blog/2024/07/03/lin-Gauss/] on linear Gaussian
inverse problems. For a more sophisticated optimization perspective that allows
$\hat{C}_{k+1}$ to be singular,
see (this)[https://arob5.github.io/blog/2024/12/03/ls-psd-cov/] post.
{% endkatexmm %}

## References
1. Inverse Problems and Data Assimilation (Stuart, Taeb, and Sanz-Alonso)

## Appendix

### Extra details for the Joint Gaussian Derivation
{% katexmm %}
We verify that the distribution $(v_{k+1},y_{k+1})|Y_k$ is actually joint
Gaussian, as claimed. Everything in this section will be conditional on
$Y_k$ so when I write $v_k$ I am referring to the random vector with distribution
$\mathcal{N}(m_k, C_k)$. To establish the claim, we will show that
\begin{align}
\begin{pmatrix} v_{k+1} \newline y_{k+1} \end{pmatrix} = a + Bz
\end{align}
for some non-random vector $a$, non-random matrix $B$, and $z \sim \mathcal{N}(0, I)$.
This is in fact one of the ways that the joint Gaussian distribution can be defined.
Proceeding with the proof, we recall that $y_{k+1}$ and $v_{k+1}$ are defined by
\begin{align}
v_{k+1} &= Gv_k + \eta_{k+1} && \eta_{k+1} \sim \mathcal{N}(0, Q) \tag{1} \newline
y_{k+1} &= Hv_{k+1} + \epsilon_{k+1}, && \epsilon_{k+1} \sim \mathcal{N}(0, R),
\end{align}
so the goal here will be to re-write the righthand side so that the only random
quantities are iid standard normal random variables. To this end, consider
\begin{align}
v_{k+1}
&= Gv_k + \eta_{k+1} \newline
&= G(m_k + C_k^{1/2}z_v) + Q^{1/2}z_{\eta}
\end{align}
where $z_v \sim \mathcal{N}(0, I_{d})$ and
$z_{\eta} \sim \mathcal{N}(0, I_{d})$. We note that the term $v_{k+1}$
appears also in the observation equation, so we re-use the above derivaton
to obtain
\begin{align}
y_{k+1}
&= Hv_{k+1} + \epsilon_{k+1} \newline
&= H\left[G(m_k + C_k^{1/2}z_v) + Q^{1/2}z_{\eta}\right] + R^{1/2}z_{\epsilon}
\end{align}
where $z_\epsilon \sim \mathcal{N}(0, I_n)$. Crucially, we note that the
conditional independence assumptions of the state space model imply that
$z_v$, $z_\eta$, and $z_\epsilon$ are all pairwise independent, which implies that
they can be concatenated to obtain
$z := (z_v, z_\eta, z_\epsilon)^\top \sim \mathcal{N}(0, I_{2d+n})$.
We have therefore found that
\begin{align}
\begin{pmatrix} v_{k+1} \newline y_{k+1} \end{pmatrix}
&= \begin{pmatrix} Gm_k \newline HGm_k \end{pmatrix} +
\begin{pmatrix} GC_k^{1/2} & Q^{1/2} & 0 \newline
HGC_k^{1/2} & HQ^{1/2} & R^{1/2} \end{pmatrix}
\begin{pmatrix} z_v \newline z_\eta \newline z_{\epsilon} \end{pmatrix},
\end{align}
which is of the required form $a + Bz$, with $B \in \mathbb{R}^{(d+n)\times(2d+n)}$
Therefore, $(v_{k+1},y_{k+1})|Y_k$ is indeed joint Gaussian distributed.
It is also interesting to note that, conditional on $Y_k$, the joint distribution
of $(v_{k+1}, y_{k+1})$ depends on $2d+n$ sources of independent Gaussian noise.
All of the interesting correlations and complexities stem from linearly combining
these Gaussian variables.

### The Woodbury Matrix Identity
We state without proof the [Woodbury identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
here, which is useful in converting between the state space and data space
KF formulations.

<blockquote>
  <p><strong>Woodbury Matrix Identity.</strong>
  Consider matrices $C \in \mathbb{R}^{n \times n}$, $A \in \mathbb{R}^{d \times d}$,
  $U \in \mathbb{R}^{d \times n}$ and $V \in \mathbb{R}^{n \times d}$ such that
  $C$, $A$, and $(A^{-1} + VC^{-1}U)$ are invertible. Then
  $$
  (C + UAV)^{-1} = C^{-1} - C^{-1}U\left(A^{-1} + VC^{-1}U \right)^{-1} VC^{-1}.
  $$
  </p>
</blockquote>

### Proof: Alternative Formula for the Kalman Gain
In this section we verify formula (11), which claims that the Kalman gain (7)
can also be written as
$$
K_{k+1} = C_{k+1}H^\top R^{-1}.
$$
This fact follows from the identity
$$
H^\top R^{-1}\left(H\hat{C}_{k+1}H^\top + R \right)
= \left(\hat{C}^{-1}_{k+1} + H^\top R^{-1}H \right)\hat{C}_{k+1}H^\top,
$$
which can be verified by simply distributing the terms on each side of the
equality. Under the KF
assumptions, both terms in parentheses above are invertible; indeed,
(1) $\left(H\hat{C}_{k+1}H^\top + R \right)$ is positive definite since $R$
is positive definite; and (2) $\left(\hat{C}^{-1}_{k+1} + H^\top R^{-1}H \right)$
is positive definite since $\hat{C}_{k+1}$ (and hence its inverse) is positive
definite. An alternative sufficient condition for the latter expression to be
positive definite is for $H$ to have full column rank. With the invertibility
established, we obtain
$$
\left(\hat{C}^{-1}_{k+1} + H^\top R^{-1}H \right)^{-1} H^\top R^{-1}
= \hat{C}_{k+1}H^\top \left(H\hat{C}_{k+1}H^\top + R \right)^{-1},
$$
where we recognize the righthand side as $K_{k+1}$. Plugging
$C_{k+1} = \left(\hat{C}^{-1}_{k+1} + H^\top R^{-1}H \right)^{-1}$ into the
lefthand side completes the proof. $\qquad \blacksquare$

### Proof: Equivalence of State Space and Data Space KF Updates
We showed above that different derivations of the mean and covariance KF recursions
result in different update formulae, one of which requires a matrix inversion in
the $d$-dimensional state space and the other requires a matrix inversion in
the $n$-dimensional data space. In this section, we prove the equivalence of these
two representations by leveraging the Woodbury matrix identity and the alternative
expression for the Kalman gain, both stated in the appendix above.

Starting with the state space equations (3), we apply the Woodbury identity
to the covariance update formula by setting (in the Woodbury notation used
above) $C := \hat{C}^{-1}_{k+1}$, $A := R^{-1}$, $U := H^\top$, and $V := H$.
Doing so yields
$$
\left(H^\top R^{-1} H + \hat{C}^{-1}_{k+1}\right)^{-1}
= \hat{C}_{k+1} - \hat{C}_{k+1} H^\top \left(R + H \hat{C}_{k+1}H^\top \right)^{-1} H \hat{C}_{k+1}.
$$
Recognizing the Kalman gain $K_{k+1} = \hat{C}_{k+1} H^\top \left(R + H \hat{C}_{k+1}H^\top \right)^{-1}$ we see that the righthand side is equal to
$(I - K_{k+1}H)\hat{C}_{k+1}$, which is the covariance update given in (8).
We now proceed with the mean update formula. Again starting with the state space
formulation (3), we have

\begin{align}
m\_{k+1}
&= C\_{k+1}\left(H^\top R^{-1}y\_{k+1} + \hat{C}^{-1}\_{k+1} \hat{m}\_{k+1} \right) \newline
&= \left(C\_{k+1}H^\top R^{-1}  \right)y\_{k+1} + C\_{k+1}\hat{C}^{-1}\_{k+1} \hat{m}\_{k+1} \newline
&= K\_{k+1}y\_{k+1} + (I - K\_{k+1}H)\hat{C}\_{k+1} \hat{C}\_{k+1}^{-1}\hat{m}\_{k+1} \newline
&= K\_{k+1}y\_{k+1} + (I - K\_{k+1}H) \hat{m}\_{k+1} \newline
&= \hat{m}\_{k+1} + K\_{k+1}(y\_{k+1} - H\hat{m}\_{k+1}),
\end{align}

where the second equality uses the alternative expression for the Kalman gain
(11) and the third plugs in the state space expression for $C_{k+1}$. We recognize
the final expression as the mean update in (4), as desired. $\qquad \blacksquare$
{% endkatexmm %}

# References
1. B. M. Bell and F. W. Cathey. The iterated Kalman filter update as a Gauss- Newton method. IEEE Transactions on Automatic Control, 38(2):294–297, 1993.
