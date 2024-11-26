---
title: Nonlinear Least Squares
subtitle: Gauss-Newton, Levenberg-Marquardt
layout: default
date: 2024-11-26
keywords: statistics
published: true
---

# Notation
{% katexmm %}
For a positive definite matrix $A$, we use the following notation for
the norm and inner product, weighted by $A$:

$$
\langle u, v \rangle_A := \langle A^{-1}u, v\rangle = u^\top A^{-1}v \newline
$$
$$
\lVert u \rVert^2_{A} := \langle u, u\rangle_{A} = u^\top A^{-1}u,
$$

where $\langle \cdot, \cdot \rangle$ and $\lVert \cdot \rVert$ denote the standard
Euclidean inner product and norm, respectively. For a function
$\fwd: \R^d \to \R^n$ we write $D\fwd(u)$ to denote
the $n \times d$ Jacobian matrix evaluated at $u$, with
$\nabla \fwd(u) := D\fwd(u)^\top$ its transpose. When $n=1$ then
$\nabla \fwd(u)$ is the gradient vector, but note that we extend the
notation beyond this typical case. Let $D_j\fwd(u)$ denote the $1 \times n$
matrix storing the derivative of $\fwd(u)$ with respect to $u_j$, and
$\nabla_j \fwd(u) \Def D_j\fwd(u)^\top$. Similarly, let $D\fwd_i(u)$ denote
the $1 \times d$ matrix storing the derivative of $\fwd(u)$ with respect to
the $i^{\text{th}}$ output $\fwd_i(u)$, and $\nabla \fwd_i(u) \Def D\fwd_i(u)^\top$
Finally, we denote the $d \times d$ Hessian matrix for a scalar-valued
function $g: \R^d \to \R$ by $\nabla^2 g(u)$. Similarly, $\nabla^2_i \fwd(u)$
is the $d \times d$ Hessian of the map $u \mapsto \fwd_i(u)$.
{% endkatexmm %}

# Introduction
{% katexmm %}
In this post we investigate optimization problems of the form

$$
\begin{align}
u_{\star} &\in \argmin_{u \in \R^d} J(u) \newline
J(u) &:= \frac{1}{2}\lVert y - \fwd(u)\rVert^2_{\Sigma} + \frac{1}{2}\lVert u - m\rVert_C^2, \tag{1}
\end{align}
$$

where $u,m \in \R^d$ and $y \in \R^d$, and $\Sigma$ and $C$ are positive
definite matrices. Notice that if $\fwd$ is linear, then (1) is simply
a regularized least squares objective. I discuss this special case in
[this](https://arob5.github.io/blog/2024/07/03/lin-Gauss/) post.
Allowing $\fwd$ to be nonlinear
generalizes linear least squares, and is therefore referred to as nonlinear
least squares (NLS). Unlike the linear case, (1) no longer admits a closed-form
solution in general, so we will consider various numerical algorithms that
seek to minimize $J(u)$. While the usual suspects (gradient descent, Newton's
method) can be applied, the quadratic structure of (1) is conducive to some
bespoke algorithms that tend to work well in this setting. We will discuss the
most well-known, the Gauss-Newton algorithm, in addition to the closely related
Levenberg-Marquardt method.

Note that other treatments of this topic will often leave out the regularization
term $\frac{1}{2}\lVert u - m\rVert_C^2$, and assume $\Sigma$ to be the identity
matrix. Alternatively, sometimes the more generic perspective
$J(u) = \lVert r(u) \rVert^2$ is taken. Setting
$r(u) \Def \Sigma^{-1/2}(y - \fwd(u))$ recovers the first term in our
present formulation in (1).
{% endkatexmm %}

# Examples

# Gradient and Hessian
