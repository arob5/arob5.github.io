---
title: Approximating Nonlinear Functions of Gaussians, Part I - Linearized Kalman Filter Extensions
subtitle: I discuss the generic problem of approximating the distribution resulting from a non-linear transformation of a Gaussian random variable, and then show how this leads to extensions of the Kalman filter which yield approximate filtering algorithms in the non-linear setting.
layout: default
date: 2024-02-15
keywords: Filtering, State-Space, Hidden-Markov-Model, Bayes, Data-Assimilation
published: false
---

{% katexmm %}
In a previous [post](https://arob5.github.io/blog/2024/02/15/Kalman-Filter/),
I discussed the Kalman filter (KF), which provides the
closed-form mean and covariance recursions characterizing the Gaussian filtering
distributions for linear Gaussian hidden Markov models. In this post, we retain
the Gaussian noise assumption, but generalize to nonlinear dynamics and
observation operators. Our primary focus will concern the additive noise model
\begin{align}
v_{k+1} &= g(v_k) + \eta_{k+1} && \eta_{k+1} \sim \mathcal{N}(0, Q) \tag{1} \newline
y_{k+1} &= h(v_{k+1}) + \epsilon_{k+1}, && \epsilon_{k+1} \sim \mathcal{N}(0, R) \newline
v_0 &\sim \mathcal{N}(m_0, C_0) \newline
\\{\epsilon_k\\} &\perp \\{\eta_k\\} \perp v_0
\end{align}
but we will also touch on the case where the noise is also subject to nonlinear
mapping; i.e.,
\begin{align}
v_{k+1} &= g(v_k, \eta_{k+1}) && \eta_{k+1} \sim \mathcal{N}(0, Q) \tag{2} \newline
y_{k+1} &= h(v_{k+1}, \epsilon_{k+1}) && \epsilon_{k+1} \sim \mathcal{N}(0, R) \newline
v_0 &\sim \mu_0.
\end{align}
Note even this more general formulation is still a special case of the generic
Bayesian filtering problem given that we are restricting to the setting with
Gaussian noise and a Gaussian initial condition.

Let $Y_k := \{y_1, \dots, y_k\}$ denote the set up observations up through time
step $k$.
We seek to characterize the filtering distributions
$v_k|Y_k$, and update them in an online fashion as more data
arrives. I will denote the density (or more generally, Radon-Nikodym derivative)
of $v_k|Y_k$ by $\pi_k(v_k) = p(v_k|Y_k)$. In the linear Gaussian setting these
distributions are Gaussian, and
can be computed analytically. The introduction of nonlinearity renders the
filtering distributions non-Gaussian and not analytically tractable in general,
motivating the need for approximations. Certain methods, such as particle
filters, are designed to handle the situation where the departure from Gaussianity
is severe. The methods discussed in this post, however, are applicable when
Gaussian approximations are still reasonable. Given this, the algorithms discussed
here all proceed by approximating the current filtering distribution by a
Gaussian $v_k|Y_k \sim \mathcal{N}(m_k, C_k)$. The algorithms then differ in
the approximations they employ to deal with the nonlinear functions $h$ and $g$
in order to arrive at a Gaussian approximation of the subsequent filtering
distribution $v_{k+1}|Y_{k+1} \sim \mathcal{N}(m_{k+1}, C_{k+1})$.

There are many methods that fit into this general Gaussian approximation
framework. In this post we focus on methods rooted in *linearization* of the
nonlinear functions. My plan is to follow this up with a post on the
unscented Kalman filter (which utilizes a quadrature-based approximation) and
then a bunch of posts on the ensemble Kalman filter (which opts for a Monte
Carlo approach). Although the motivation stems from the filtering problem, I
will focus mostly on the underlying fundamental problem here: approximating the
distribution resulting from propagating a Gaussian random variable through a
nonlinear map. The next section illustrates how this problem arises in the
filtering context.
{% endkatexmm %}

## Motivating the Generic Problem
{% katexmm %}
We recall from the
[post](https://arob5.github.io/blog/2024/01/29/Bayesian-filtering/)
on Bayesian filtering that the map $\pi_k \to \pi_{k+1}$
naturally decomposes into two steps: the forecast and analysis steps. We assume
here the Gaussian approximation $v := v_k|Y_k \sim \mathcal{N}(m_k, C_k)$ has been
invoked and consider the approximations that will be required in propagating
this distribution through the map $\pi_k \to \pi_{k+1}$.

### Forecast
The forecast distribution $\hat{\pi}_{k+1}(v_{k+1}) := p(v_{k+1}|Y_k)$ is
the distribution implied by feeding
$\pi_k$ through the stochastic dynamics model. In the additive noise case (1),
we observe that this comes down to approximating the distribution of the random
variable
$$
g(v) + \epsilon, \qquad v \sim \mathcal{N}(m_k, C_k), \epsilon \sim \mathcal{N}(0, Q).
$$
However, since we will be invoking a Gaussian approximation
$g(v) \sim \mathcal{N}(\hat{m}_{k+1}, \hat{C}_{k+1})$, in addition to the assumption
that $v$ and $\epsilon$ are independent, then we see that the $\epsilon$ will
not be a problem. Due to the independence, once we obtain the approximation
$g(v) \sim \mathcal{N}(\hat{m}_{k+1}, \hat{C}_{k+1})$, then the approximation
$g(v) + \epsilon \sim \mathcal{N}(\hat{m}_{k+1}, \hat{C}_{k+1} + Q)$ is
immediate.

In the non-additive noise case, the problem similarly comes down to approximating
the distribution
$$
g(v, \epsilon), \qquad v \sim \mathcal{N}(m_k, C_k), \epsilon \sim \mathcal{N}(0, Q).
$$
Again, due to the independence assumptions we have that $(v, \epsilon)$ are
jointly distributed
\begin{align}
\begin{pmatrix} v \newline \epsilon \end{pmatrix}
\sim \mathcal{N}\left(\begin{pmatrix} m_k \newline 0 \end{pmatrix},
  \begin{pmatrix} C_k & 0 \newline 0 & Q \end{pmatrix} \right).
\end{align}
Thus, this situation also reduces to approximating the distribution
of a nonlinear map of a Gaussian; in this case, $g(\tilde{v})$, where
$\tilde{v} := (v, \epsilon)$.

### Analysis
Let's now suppose that we have the forecast approximation
$\hat{v} := v_{k+1}|Y_k \sim \mathcal{N}(\hat{m}_{k+1}, \hat{C}_{k+1})$ in hand.
The map $\hat{\pi}_{k+1} \mapsto \pi_{k+1}$ from forecast to filtering
distribution is defined by the action of conditioning on the data $y_{k+1}$.
In the additive noise case (1), this entails the following application of Bayes'
theorem,
\begin{align}
\pi_{k+1}(v_{k+1})
&\propto \mathcal{N}(y_{k+1}|h(v_{k+1}), R)\mathcal{N}(v_{k+1}|\hat{m}\_{k+1}, \hat{C}\_{k+1}).
\end{align}
Although everything is Gaussian here, the nonlinear function $h$ breaks the Gaussianity
of $\pi_{k+1}$ in general. One idea to deal with this might be to run MCMC
and invoke the approximation $v_{k+1}|Y_{k+1} \sim \mathcal{N}(m_{k+1}, C_{k+1})$
with $m_{k+1}$ and $C_{k+1}$ set to their empirical estimates computed from the
MCMC samples. This has the distinct disadvantage of requiring an MCMC run at
every time step.

In order to discover alternative approximation methods, it is useful to recall
the joint Gaussian view of the analysis step, which I discuss in
[this](https://arob5.github.io/blog/2024/02/15/Kalman-Filter/) post. The idea
here was that, in the linear Gaussian setting,
$(v_{k+1}, y_{k+1})|Y_k$ has a joint Gaussian distribution. The filtering
distribution $v_{k+1}|Y_{k+1} = v_{k+1}|y_{k+1}, Y_k$ is then obtained as
a conditional distribution of the joint Gaussian, which is available in closed-form.
In the present
setting of the additive noise model (1) this joint distribution is given by
\begin{align}
\begin{pmatrix} v_{k+1} \newline y_{k+1} \end{pmatrix} \bigg| Y_k
&= \begin{pmatrix} \hat{v} \newline h(\hat{v}) + \epsilon_{k+1} \end{pmatrix},
&& \hat{v} \sim \mathcal{N}(\hat{m}\_{k+1}, \hat{C}\_{k+1})
\end{align}
which is again generally non-Gaussian due to $h$. This perspective points to the
idea of approximating this joint distribution as a Gaussian, so that an
approximation of the filtering distribution then falls out as a conditional.
Notice that we have found ourselves in a very similar situation to the analysis
step, in that we again want to approximate the nonlinear mapping of a Gaussian
with a Gaussian. The problem
is thus to furnish a Gaussian approximation of
\begin{align}
\tilde{h}(\hat{v}) &= \begin{pmatrix} \hat{v} \newline h(\hat{v}) + \epsilon_{k+1} \end{pmatrix},
&& \hat{v} \sim \mathcal{N}(\hat{m}\_{k+1}, \hat{C}\_{k+1}), \epsilon_{k+1} \sim \mathcal{N}(0, R).
\end{align}
In the non-additive error case (2), $h(\hat{v}, \epsilon_{k+1})$ replaces
$h(\hat{v}) + \epsilon_{k+1}$ in the above expression.
{% endkatexmm %}

{% katexmm %}
## The Generic Problem Setting
{% endkatexmm %}


# TODO
Discuss the change of measure formula for the PDF, extended Kalman filter,
statistically linearlized Kalman filter, Fourier-Hermite Kalman filter,
unscented Kalman filter (see Saarka Bayesian estimation of time-varying
  systems notes).
