---
title: Linearly Transforming Gaussian Process Priors
subtitle: I derive how a linear transformation of a Gaussian process prior influences the Gaussian process posterior, and consider some special cases.
layout: default
date: 2024-03-06
keywords: GP
published: false
---

## Background
{% katexmm %}
We consider a Gaussian process (GP) distribution over functions
$f: \mathcal{U} \to \mathbb{R}$ with $\mathcal{U} \subset \mathbb{R}^D$. In
particular, consider a GP $f \sim \mathcal{GP}(\mu, k)$ with mean function
$\mu: \mathcal{U} \to \mathbb{R}$ and positive definite kernel
(i.e., covariance function) $k: \mathcal{U} \times \mathcal{U} \to \mathbb{R}$.
Throughout this post we suppose that we have observed the function evaluations
$(u_n, f(u_n))$, $n = 1, \dots, N$ and seek to perform inference at a set of
new inputs $(\tilde{u}_m)$, $m = 1, \dots, M$. We collect the two sets up
inputs into matrices $U \in \mathbb{R}^{N \times D}$ and
$\tilde{U} \in \mathbb{R}^{M \times D}$, respectively. In general, I like to
override the function notation to vectorize the operations; i.e.,
$f(U) \in \mathbb{R}^N$ and $k(U, \tilde{U}) \in \mathbb{R}^{N \times M}$
where $f(U)_{n} = f(u_n)$ and $k(U, \tilde{U})_{n,m} = k(u_n, u_m)$.
However, since we will only be considering the fixed (but arbitrary) inputs
$U$ and $\tilde{U}$, we will lighten notation by writing $\mathbf{f} = f(U)$,
$\mathbf{\tilde{f}} = f(\tilde{U})$, $\boldsymbol{\mu} = \mu(U)$,
$\boldsymbol{\tilde{\mu}} = \mu(\tilde{U})$, $K = k(U,U)$,
$\tilde{K}=k(\tilde{U}, \tilde{U})$, and $C = k(U, \tilde{U})$ (C for
**c**ross-covariance).

With notation established, we emphasize that the vector $\mathbf{\tilde{f}}$
is unobserved and the goal is to characterize the conditional distribution
$p(\mathbf{\tilde{f}}|\mathbf{f})$. This conditional can be derived by
considering the joint distribution implied by the GP prior
\begin{align}
\begin{pmatrix} \mathbf{\tilde{f}} \newline \mathbf{f} \end{pmatrix}
\sim \mathcal{N}\left(\begin{pmatrix} \boldsymbol{\tilde{\mu}} \newline \boldsymbol{\mu} \end{pmatrix},
\begin{pmatrix} \tilde{K} \quad C^\top \newline C \quad K \end{pmatrix} \right).
\end{align}
The well-known Gaussian conditioning identities imply that the conditional
$\mathbf{\tilde{f}}_N := \mathbf{\tilde{f}}|\mathbf{f}$ is also Gaussian
$\mathbf{\tilde{f}}_N \sim \mathcal{N}(\boldsymbol{\tilde{\mu}}_N, \tilde{K}_N)$
with mean and covariance given by
\begin{align}
\boldsymbol{\tilde{\mu}}_N
&= \mathbb{E}[f(\tilde{U})|f(U)]
= \boldsymbol{\tilde{\mu}} + C^\top K^{-1}(\mathbf{f} - \boldsymbol{\mu}) \newline
\tilde{K}_N
&= \text{Cov}[f(\tilde{U})|f(U)] = \tilde{K} - C^\top K^{-1}C.
\end{align}

{% endkatexmm %}
