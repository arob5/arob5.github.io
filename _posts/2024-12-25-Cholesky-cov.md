---
title: Cholesky Decomposition of a Covariance Matrix
subtitle: Regression interpretation of the Cholesky decomposition, with applications to covariance estimation.
layout: default
date: 2024-12-25
keywords:
published: true
---

In this post we consider the Cholesky decomposition of the covariance
matrix of a Gaussian distribution. The eigendecomposition of covariance
matrices gives rise to the well-known method of
[principal components analysis](https://arob5.github.io/blog/2023/12/15/PCA/).
The Cholesky decomposition is not as widely discussed in this context, but
also has a variety of useful statistical applications.

# Setup and Background
{% katexmm %}

## The Cholesky Decomposition
The Cholesky decomposition of a positive definite matrix $C$ is the
unique factorization of the form
$$
C = LL^\top, \tag{1}
$$
where $L$ is a lower-triangular matrix with positive diagonal
elements (note that constraining the diagonal to be positive is required
for uniqueness).
A positive definite matrix can also be uniquely decomposed as
$$
C = LDL^\top, \tag{2}
$$
where $L$ is lower-triangular with ones on the diagonal, and $D$ is a diagonal
matrix with positive entries on the diagonal. We will refer to this as the
*modified Cholesky decomposition*, but it is also often called the
*LDL decomposition*. Given the modified Cholesky decomposition
$C = \tilde{L}D\tilde{L}^\top$, we can form (1) by setting
$L := \tilde{L}D^{1/2}$. We refer to both $\tilde{L}$ and $L$ as the
*lower Cholesky factor* of $C$; which we are referring to will be clear from
context. $L$ is guaranteed to be invertible, and $L^{-1}$ is itself a
lower-triangular matrix.

## Statistical Setup
Throughout this post we consider a random vector
$$
x := \left(x^{(1)}, \dots, x^{(p)} \right)^\top \in \mathbb{R}^p, \tag{3}
$$
with positive definite covariance $C := \text{Cov}[x]$. We will often assume
that $x$ is Gaussian, but this assumption is not required for some of the
results discussed below. We focus on the (modified) Cholesky decomposition
of $C$, and let $L = \{\ell_{ij}\}$ denote the entries of the lower Cholesky
factor. For the modified decomposition, we denote
$D := \text{diag}(d_1, \dots, d_p)$, where $d_i > 0$.
{% endkatexmm %}

# The Basics
Let $C = LDL^\top$ and define the random variable
$$
\epsilon := L^{-1}x, \tag{4}
$$
which satisfies
$$
\text{Cov}[\epsilon] = L^{-1}CL^{-\top} = L^{-1}LDL^\top L^{-\top} = D. \tag{4}
$$
Thus, the map $x \mapsto L^{-1}x$ outputs a "decorrelated" random vector. The
inverse map $\epsilon \mapsto L\epsilon$ "re-correlates" $\epsilon$, producing
a random vector with covariance $C$. If we add on the assumption that
$x$ is Gaussian, then $\epsilon$ is a Gaussian vector with independent
entries. The transformation $L\epsilon$ is the typical method used in simulating 
draws from a correlated Gaussian vector.
