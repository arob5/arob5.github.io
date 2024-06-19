---
title: Linearly Transforming Gaussian Process Priors
subtitle: I derive how a linear transformation of a Gaussian process prior influences the Gaussian process posterior, and consider some special cases.
layout: default
date: 2024-03-06
keywords: GP
published: true
---

The main goal of this post is to apply different types of linear transformations
to Gaussian vectors, and investigate how these transformations impact the
resulting Gaussian conditional distributions. Since many Gaussian process (GP) derivations
reduce to calculations with Gaussian vectors, our investigations of
multivariate Gaussians will immediately lead to results on GPs. From the GP
perspective, I emphasize that we will *not* be considering linear functionals
of GPs (i.e., maps that take a function as input and return a scalar). This is
an interesting topic that is worthy of its own blog post. We will instead be
considering linear transformations applied in a *pointwise* fashion. We start
by considering a generic multivariate Gaussian setup, then translate the
results into GP language. We conclude by discussing applications to
GP regression, multi-output GPs, and linear inverse problems.

## Multivariate Gaussians

{% katexmm %}
### Notation and Review
We start by considering (finite-dimensional) multivariate Gaussians, as many of
the derivations in the Gaussian process setting reduce to computations with
Gaussian vectors. Consider a Gaussian vector
$x \sim \mathcal{N}(\mu, C)$ partitioned as
\begin{align}
x = \begin{bmatrix} x_M \newline x_N \end{bmatrix}
&\sim \mathcal{N}\left(
\begin{bmatrix} \mu_M \newline \mu_N \end{bmatrix},
\begin{bmatrix} C_M & C_{MN} \newline C_{NM} & C_N \end{bmatrix}
\right) \tag{1}
\end{align}
where $x_M \in \mathbb{R}^M$, $x_N \in \mathbb{R}^N$, and
$C_{NM} = C_{MN}^\top$. Throughout this post,
subscripts with capital letters serve as indicators of vector and matrix
dimensions. It is well-known that
the conditional distribution $x_M|x_N$ is again Gaussian. In particular, we
have $x_M|x_N \sim \mathcal{N}(\hat{\mu}_M, \hat{C}_N)$, where
\begin{align}
\hat{\mu}\_M &:= \mu_M + C_{MN} C_{N}^{-1}(x_N - \mu_N) \tag{2} \newline
\hat{C}\_M &:= C_M - C_{MN} C_{N}^{-1} C_{NM}.
\end{align}

### Linear Transformation of Gaussian Vector
We recall that linear transformations of Gaussians preserve Gaussianity.
Therefore, for a matrix $A \in \mathbb{R}^{R \times (M + N)}$, the random
vector $y := Ax$ has distribution

\begin{align}
y \sim \mathcal{N}(A\mu, ACA^\top). \tag{3}
\end{align}
In this post we will be concerned with matrices $A$ with certain structure; the
motivation will become more clear when we start considering GPs.
For now, suppose that $A$ is of the form
\begin{align}
A :=
\begin{bmatrix} A_M & 0 \newline 0 & A_N \end{bmatrix}, \tag{4}
\end{align}
where $A_M \in \mathbb{R}^{R_1 \times M}$ and $A_N \in \mathbb{R}^{R_2 \times N}$.
In this case, the transformed distribution (3) assumes the form
\begin{align}
y \sim \mathcal{N}\left(
\begin{bmatrix} A_M\mu_M \newline A_N \mu_N \end{bmatrix},
\begin{bmatrix}
A_M C_M A_M^\top & A_M C_{MN} A_N^\top \newline
A_N C_{NM}A_M^\top & A_N C_N A_N^\top
\end{bmatrix}
\right). \tag{5}
\end{align}
Having characterized the joint distribution of the transformed vector $Ax$, we
now consider the effect on the conditional distribution. Let
$y_M := A_M x_M$ and $y_N := A_N x_N$. Applying the Gaussian conditioning
identity (2), we obtain
\begin{align}
y_M | y_N \sim \mathcal{N}\left(\hat{\mu}_M^{y}, \hat{C}_M^{y} \right),
\end{align}  
where

\begin{align}
\hat{\mu}\_{M}^{y}
&:= A_M \mu_M + A_M C_{MN} A_N^\top (A_N C_N A_N^\top)^{-1}[A_N x_N - A_N \mu_N] \tag{6} \newline
\hat{C}\_{M}^{y}
&:= A_M C_M A_M^\top - A_M C_{MN} A_N^\top (A_N C_N A_N^\top)^{-1} A_N C_{NM} A_M^\top.
\end{align}

#### Generalization to Affine Maps
The generalization from linear to affine maps is almost immediate. Consider an
affine map of the form
\begin{align}
y := Ax + b
&= \begin{bmatrix} A_M & 0 \newline 0 & A_N \end{bmatrix}
\begin{bmatrix} x_M \newline x_N \end{bmatrix} +
\begin{bmatrix} b_M \newline b_N \end{bmatrix}. \tag{7}
\end{align}
The joint distribution of $y$ is then given by  
\begin{align}
y \sim \mathcal{N}\left(
\begin{bmatrix} A_M\mu_M + b_M \newline A_N \mu_N + b_N \end{bmatrix},
\begin{bmatrix}
A_M C_M A_M^\top & A_M C_{MN} A_N^\top \newline
A_N C_{NM}A_M^\top & A_N C_N A_N^\top.
\end{bmatrix} \right) \tag{8}
\end{align}
Note that the constant terms only affect the mean. Applying the Gaussian
conditioning formulas then gives the conditional distribution
\begin{align}
y_M | y_N \sim \mathcal{N}\left(\hat{\mu}_M^{y}, \hat{C}_M^{y} \right),
\end{align}  
where

\begin{align}
\hat{\mu}\_{M}^{y}
&:= b_M + A_M \mu_M + A_M C_{MN} A_N^\top (A_N C_N A_N^\top)^{-1}[A_N x_N - A_N \mu_N] \tag{9} \newline
\hat{C}\_{M}^{y}
&:= A_M C_M A_M^\top - A_M C_{MN} A_N^\top (A_N C_N A_N^\top)^{-1} A_N C_{NM} A_M^\top.
\end{align}
The only difference with respect to (6) is the addition of $b_M$ in the conditional
mean. Note that the $A_N x_N - A_N \mu_N$ term in the conditional mean is unchanged
due to the cancellation $b_M - b_M$.

Note that $A_M$ and $A_N$ map to $\mathbb{R}^{R_1}$ and $\mathbb{R}^{R_2}$,
respectively. Therefore, the dimension of $Ax$ may differ from that of
$x$. Now, it would be nice to be able to write $\hat{\mu}_{M}^{y}$ and
$\hat{C}_{M}^{y}$ as functions of $\hat{\mu}_M$ and $\hat{C}_M$, respectively.
In the general setting, there is not much we can do given that $A_N$ may
not be invertible; thus, we can't necessarily simplify the term
$(A_N C_N A_N^\top)^{-1}$.

#### Special case: Invertibility
Let's now consider the special case that $A_N$ is invertible; in particular,
this means $A_N \in \mathbb{R}^{N \times N}$. We'll work with the affine map
(7), since the linear result follows as the special case $b = 0$.
With the invertibility assumption we can simplify (9) as
\begin{align}
\hat{\mu}\_{M}^{y}
&= b_M + A_M \mu_M + A_M C_{MN} A_N^\top (A_N^\top)^{-1} C_N^{-1} A_N^{-1} A_N[x_N - \mu_N] \tag{10} \newline
&= b_M + A_M \left(\mu_M + C_{MN} C_N^{-1} [x_N - \mu_N] \right) \newline
&= b_M + A_M \hat{\mu}\_{M} \newline
\hat{C}\_{M}^{y}
&= A_M C_M A_M^\top - A_M C_{MN} A_N^\top (A_N C_N A_N^\top)^{-1} A_N C_{NM} A_M^\top \newline
&= A_M C_M A_M^\top - A_M C_{MN} A_N^\top (A_N^\top)^{-1} C_N^{-1} A_N^{-1} A_N C_{NM} A_M^\top \newline
&= A_M \left(C_M - C_{MN} C_N^{-1} C_{NM}\right) A_M^\top \newline
&= A_M \hat{C}\_{M} A_M^\top.
\end{align}
In words, what we have just shown is that, if $A_N$ is invertible, then
conditioning $y_M|y_N$ is equivalent to conditioning $x_M|x_N$ and then
applying the transformation $A_M$ after the fact. We might write this symbolically
as
\begin{align}
(A_M x_M | A_N x_N) \overset{d}{=} A_M(x_M | x_N). \tag{11}
\end{align}
This result makes intuitive sense; since $A_N$ is a bijection, then conditioning
on $x_N$ is equivalent to conditioning on $A_N x_N$ - they both contain the
same information. Note that no invertibility assumption is required for $A_M$;
what matters here is the variable that is being conditioned on.

{% endkatexmm %}


## Gaussian Process Review
{% katexmm %}
### Gaussian Process Prior
We consider a Gaussian process (GP) distribution over functions
$f: \mathcal{U} \to \mathbb{R}$ with $\mathcal{U} \subset \mathbb{R}^D$. In
particular, consider a GP $f(\cdot) \sim \mathcal{GP}(\mu(\cdot), k(\cdot, \cdot))$
with mean function
$\mu: \mathcal{U} \to \mathbb{R}$ and positive definite kernel
(i.e., covariance function) $k: \mathcal{U} \times \mathcal{U} \to \mathbb{R}$.
Throughout this post we suppose that we have observed the function evaluations
$\{u_n, f(u_n)\}_{n=1}^{N}$ and seek to perform inference at a set of
new inputs $\{\tilde{u}_m\}_{m=1}^{M}$. We will only be considering these two
sets of inputs throughout this post, so we will lighten notation by letting  
$f, \mu \in \mathbb{R}^{N}$ denote the vectors defined by
$f_n := f(u_n)$ and $\mu_n := \mu(u_n)$. We analogously define
$\tilde{f}, \tilde{\mu} \in \mathbb{R}^M$ to be the vectors of the GP and mean
function evaluations at the unobserved inputs $\tilde{u}_1, \dots, \tilde{u}_M$.
Finally, we let $K \in \mathbb{R}^{N \times N}$,
$\tilde{K} \in \mathbb{R}^{M \times M}$, and $C \in \mathbb{R}^{N \times M}$
denote the matrices given by
$K_{n,n^\prime} := k(u_n, u_{n^\prime})$, $\tilde{K}_{m,m^\prime} := k(u_m, u_{m^\prime})$,
and $C_{n, m} := k(u_n, u_m)$.

### Gaussian Process Posterior  
The vector $\tilde{f}$ is unobserved and the goal is to characterize the
conditional distribution $\tilde{f}|f$ (note that we will always be implicitly
conditioning on the inputs). This distribution can be derived by considering the
joint distribution implied by the GP prior:
\begin{align}
\begin{bmatrix} \tilde{f} \newline f \end{bmatrix}
\sim \mathcal{N}\left(\begin{bmatrix} \tilde{\mu} \newline \mu \end{bmatrix},
\begin{bmatrix} \tilde{K} \quad C^\top \newline C \quad K \end{bmatrix} \right).
\end{align}
The well-known Gaussian conditioning identities imply that the conditional
$\tilde{f}_{\star} := \tilde{f}|f$ is also Gaussian
$\tilde{f}_{\star} \sim \mathcal{N}(\tilde{\mu}_{\star}, \tilde{K}_{\star})$
with mean and covariance given by

\begin{aligned}
\tilde{\mu}^{N} &:= \tilde{\mu} + C^\top K^{-1} (f - \mu) \newline
\tilde{K}^{N} &:= \tilde{K} - C^\top K^{-1} C.
\end{aligned}

Let's now consider

{% endkatexmm %}

## Pointwise Transformations of Gaussian Processes


## Applications

### Normalizing the Response
### Multi-Output GPs
### Inverse Problem with Linear Forward Model
### Input Dimension Reduction
Discuss alternate view as defining a new kernel.
