---
title: Nonlinear Least Squares
subtitle: Gauss-Newton, Levenberg-Marquardt
layout: default
date: 2024-11-26
keywords: statistics
published: true
---

This post provides an introduction to nonlinear least squares (NLS), which
generalizes the familiar least squares problem by allowing for a nonlinear
forward map. Unlike least squares, the NLS problem does not admit closed-form
solutions, but a great deal of work has been devoted to numerical schemes
tailored to this specific problem. We introduce two of the most popular
algorithms: Gauss-Newton and Levenberg-Marquardt.

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
function $g: \R^d \to \R$ by $\nabla^2 g(u) := D\nabla g(u)$.
Similarly, $\nabla^2_i \fwd(u)$ is the $d \times d$ Hessian of the map
$u \mapsto \fwd_i(u)$.
{% endkatexmm %}

# Introduction
{% katexmm %}
In this post we investigate optimization problems of the form

\begin{align}
u_{\star} &\in \text{argmin}\_{u \in \mathbb{R}^d} J(u) \newline
J(u) &:= \frac{1}{2}\lVert y - \mathcal{G}(u)\rVert^2_{\Sigma} + \frac{1}{2}\lVert u - m\rVert_C^2, \tag{1}
\end{align}

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

## Statistical Perspective
The solution $u_{\star}$ to the optimization problem in (1) can be viewed
as the maximum a posterior (MAP) estimate of the Bayesian model
\begin{align}
y|u &\sim \mathcal{N}(\mathcal{G}(u), \Sigma) \tag{2} \newline
u &\sim \mathcal{N}(m, C).
\end{align}
In particular, the posterior density of (2) satisfies
$$
p(u|y) \propto \exp\left[-J(u)\right], \tag{3}
$$
where $J(u)$ is the nonlinear least squares objective (1).

## Gradient and Hessian
The gradient and Hessian of $J(u)$ play a prominent role in numerical
algorithms that seek to solve (1). We provide the expressions for these
quantities below, with derivations given in the appendix.
<blockquote>
  <p><strong>Gradient and Hessian.</strong> <br>
  The gradient $\nabla J(u)$ and Hessian $\nabla^2 J(u)$ of the cost function
  defined in (1) are given by
  \begin{align}
  \nabla J(u) &= -\mathcal{G}(u)\Sigma^{-1}(y-\mathcal{G}(u)) + C^{-1}(y-m) \tag{4} \newline
  \nabla^2 J(u) &= [\nabla \mathcal{G}(u)]\Sigma^{-1}[\nabla \mathcal{G}(u)]^\top  +
  \sum_{i=1}^{n} \epsilon_i(u) \nabla^2 \mathcal{G}_i(u) + C^{-1} \tag{5},
  \end{align}
  where $\epsilon_i(u)$ is the $i^{\text{th}}$ entry of the vector
  $\epsilon(u) := \Sigma^{-1}(\mathcal{G}(u) - y)$.
  </p>
</blockquote>

In contrast to least squares, the Hessian $\nabla^2 J(u)$ depends on $u$ in
general.
{% endkatexmm %}

# Numerical Optimization
{% katexmm %}
We now discuss numerical schemes that construct a sequence of iterates
$u_1, u_2, \dots$ with the goal of converging to an optimum of $J(u)$.

## Newton's Method
Given a current iterate $u_k$, Newton's method constructs a quadratic
approximation of $J(u)$ centered at the point $u_k$, then minimizes this
quadratic function to select the next iterate $u_{k+1}$. Specifically,
Newton's method considers the second order Taylor expansion
$$
J_k(u) := J(u_k) + DJ(u_k)[u-u_k] + \frac{1}{2}(u-u_k)^\top [\nabla^2 J(u_k)] (u-u_k). \tag{5}
$$
The next iterate $u_{k+1}$ is thus chosen by solving
$$
\nabla J_k(u_{k+1}) = 0, \tag{6}
$$
which yields
$$
\nabla J_k(u_{k_1}) = \nabla J(u_k) + [\nabla^2 J(u_k)] (u_{k+1}-u_k) = 0. \tag{7}
$$
At this point, note that it may be the case that $J_k$ doesn't even have a minimum,
in which case $\nabla^2 J(u_k)$ is not positive definite. We will not get into this
issue here, and instead simply assume that $\nabla^2 J(u_k)$ is invertible. Then
we can solve for $u_{k+1}$ in (7) as
$$
u_{k+1} = u_k - [\nabla^2 J(u_k)]^{-1} \nabla J(u_k). \tag{8}
$$
Working with the Hessian here can be difficult in certain settings; note in
(5) that computing the Hessian of $J$ requires computing the Hessian of
$\mathcal{G}$ for each output dimension. Since $\mathcal{G}$ can be a very
complicated and computationally function, this presents challenges. The below
methods maintain the spirit of the Newton update (8), but replace the Hessian
with something that is easier to compute.

## Gauss-Newton
We introduce the Gauss-Newton update from two different perspectives. The
first simply approximates the Hessian by ignoring the "difficult" terms, while
the second views the update as solving a "local" least squares problem.

### Approximate Hessian
The Gauss-Newton procedure performs the Newton update (8) with an approximate
Hessian constructed by simply dropping the higher-order terms in (5).
Doing so yields the Hessian approximation
$$
\hat{H}_k := G_k^\top \Sigma^{-1}G_k + C^{-1}, \tag{9}
$$
where we have introduced the shorthand for the Jacobian of $\mathcal{G}$
evaluated at $u_k$:
$$
G_k := D\mathcal{G}(u_k) \tag{10}
$$
Notice that (9) no longer requires extracting second order derivative
information from $\mathcal{G}$. The resulting Gauss-Newton update takes the form
$$
u_{k+1} = u_k - \hat{H}_k^{-1} \nabla J(u_k). \tag{11}
$$
Plugging in the expressions for $\hat{H}_k$ and $\nabla J(u_k)$ yields the
explicit update
$$
u_{k+1}
= u_k - \left(G_k^\top \Sigma^{-1}G_k + C^{-1} \right)^{-1} \left[C^{-1}(y-m) - \mathcal{G}(u_k)\Sigma^{-1}(y-\mathcal{G}(u_k)) \right]. \tag{12}
$$

A nice consequence of using $\hat{H}_k$ is that, unlike the Hessian, it is
guaranteed to be positive semidefinite. This follows from the fact that
$G^\top_k \Sigma^{-1}G_k$ is positive semidefinite, $C^{-1}$ is positive definite,
and the sum of a positive semidefinite and positive definite matrix is positive
definite. The invertibility of $\hat{H}_k$ is thus guaranteed, and the local quadratic
function that Gauss-Newton is implicitly minimizing is guaranteed to have a minimum.  

### Local Least Squares Perspective

## Levenberg-Marquardt

{% endkatexmm %}

# References
- Nonlinear Least Squares (neos Guide)
- A multispectrum nonlinear least-squares fitting technique
- What to Do When Your Hessian Is Not Invertible (Gill and King)
