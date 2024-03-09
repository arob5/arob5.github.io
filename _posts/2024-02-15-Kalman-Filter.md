---
title: The Kalman Filter - A Few Different Perspectives
subtitle: I discuss the Kalman filter from both the Bayesian and optimization perspectives.
layout: default
date: 2024-02-15
keywords: Bayes, Filtering, Data-Assim
published: true
---

Originally introduced in R.E. Kalman's seminal 1960 paper, the Kalman filter has
found a myriad of applications in fields as disparate as robotics and economics.
In short, the Kalman filter (KF) gives the closed-form solutions to the filtering
problem (see my previous post) in the linear-Gaussian setting; that is, the
deterministic dynamic model and observaton operator are each linear and subject
to additive Gaussian noise, and the initial condition is also Gaussian.
As in other models with Gaussian noise, the KF
can alternatively be viewed as an optimization algorithm, which is optimal
in the sense of minimizing the quadratic loss. Indeed, this optimization
perspective was the one originally considered by Kalman in the original paper.
In this post I begin by deriving the KF equations in the Bayesian statistical
setting in two different ways. The two derivations produce two different sets
of equations, which I show are equivalent through an application of the Woodbury
matrix identity. I then proceed to the optimization perspective, again offering
two different derivations, one rooted in calculus and the other--the projection
based approach used in the 1960 paper--in linear algebra.


TODOs:
1. Add section that just views this as a sequential implementation of linear regression.
2. Add section on interpreting the KF equations.
3. Add section on computational complexity.

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
sort of Monte Carlo or closed-form approximation to the true distribution. However,
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
$C_k \to C_{k+1}$. The following proposition summarizes the main result, providing
these recursions in two different (but equivalent) forms. These filtering equations
are broken into two stages, following the *forecast* and *analysis* steps
that I discussed in a previous post. Derivations of these
equations are given in the subsequent sections.

**Proposition.** Given the state space model (1), the forecast and filtering
distributions are both Gaussian

\begin{align}
&\text{Forecast Distribution: } v_k|Y_{k-1} \sim \mathcal{N}(\hat{m}_k, \hat{C}_k) \newline
&\text{Filtering Distribution: } v_k|Y_k \sim \mathcal{N}(m_k, C_k)
\end{align}

and the mean and covariance for these distributions satisfy the following
recursions.

\begin{align}
\hat{C}\_{k+1} &= G C_k G^\top + Q \newline
\hat{m}\_{k+1} &= Gm_k \newline
C\_{k+1} &= \left(H^\top R^{-1} H + \hat{C}^{-1}\_{k+1}\right)^{-1} \newline
m\_{k+1} &= C\_{k+1}\left(H^\top R^{-1}y\_{k+1} + \hat{C}^{-1}\_{k+1} \hat{m}\_{k+1} \right)
\end{align}

TODO: note that can generalize to time-varying Q and R

## Bayesian Derivation
We begin with an approach aligns directly with the derivations of the generic
forecast and analysis updates derived in the previous post. These generic updates
are typically analytically intractable, but in the current linear Gaussian setting
all of the calculations go through neatly in closed-form. The derivation
proceeds inductively; note that the base case is provided by the initial condition
$v_0 \sim \mathcal{N}(m_0, C_0)$. The inductive hypothesis assumes that
$v_k|Y_k \sim \mathcal{N}(m_k, C_k)$ and it remains for us to show that
$v_{k+1}|Y_{k+1} \sim \mathcal{N}(m_{k+1}, C_{k+1})$ with the mean and covariance
formulas as given in the proposition.

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
on $Y_k$.

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
which completes the derivation.


## A Second Bayesian Derivation: The Joint Gaussian Approach  
We now explore a second derivation of the KF equations, which yield the second
form of the equations presented in the proposition. The key observation here
is that the joint distribution of $(v_{k+1},y_{k+1})|Y_k$ is Gaussian
\begin{align}
\begin{pmatrix} v_{k+1} \newline y_{k+1} \end{pmatrix}\bigg|Y_k \tag{2}
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

With the specification of this distribution complete, we note that the
filtering distribution we care about,
$v_{k+1}|Y_{k+1} = v_{k+1}|y_{k+1},Y_k$ is a conditional
distribution of (2); in particular, it is the conditional distribution
resulting from conditioning on the final $n$ dimensions. Applying the
closed-form [equation](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
for Gaussian conditionals, we conclude that
$v_{k+1}|Y_{k+1} \sim \mathcal{N}(m_{k+1}, C_{k+1})$, where

\begin{align}
m_{k+1} &= \hat{m}\_{k+1} + \hat{C}^{vy}\_{k+1}\left[\hat{C}^y\_{k+1}\right]^{-1}(y_{k+1} - H\hat{m}\_{k+1}) \newline
C_{k+1} &= \hat{C}_{k+1} - \hat{C}^{vy}\_{k+1}\left[\hat{C}^y\_{k+1}\right]^{-1}\hat{C}^{yv}.
\end{align}

Typically, these equations are written in terms of the $d \times n$
**Kalman gain** matrix
\begin{align}
K_{k+1} = \hat{C}^{vy}_{k+1} \left[\hat{C}^y\_{k+1}\right]^{-1}
\end{align}
which gives

\begin{align}
m_{k+1} &= \hat{m}\_{k+1} + K_{k+1}(y_{k+1} - H\hat{m}\_{k+1}) \newline
C_{k+1} &= \hat{C}\_{k+1} - K\_{k+1}\hat{C}^{yv}.
\end{align}

Inserting the formulas for $\hat{m}_{k+1}$ and $\hat{C}_{k+1}$ provides the
equations defining the complete maps
$m_k \mapsto m_{k+1}$ and $C_k \mapsto C_{k+1}$.

TODO: re-write the equations in the weighted average form (see Stuart book). 

{% endkatexmm %}

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

### Showing the Equivalence of the Two KF Formulas

{% endkatexmm %}
