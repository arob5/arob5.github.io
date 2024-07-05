---
title: The Polynomial Kernel
subtitle: Introduction and basic properties, kernel ridge regression, Gaussian processes.
layout: default
date: 2024-06-27
keywords: kernels, Gaussian-Process
published: true
---

In this post we discuss the polynomial kernel, with particular emphasis on
the quadratic special case. We motivate the kernel within a (regularized)
regression context, then discuss basic properties, and applications to
Gaussian processes.

# From Basis Functions to Kernel

## Polynomial Basis
{% katexmm %}
We consider working with the multivariate polynomial space

$$
\mathcal{P}_p(\mathbb{R}^d)
:= \text{span}\left\{x_1^{a_1} \cdots x_d^{a_d} : a_1 + \dots + a_d \leq p \right\}, \tag{1}
$$

with subscripts indexing the $d$ input dimensions.
The dimension of $\mathcal{P}_p(\mathbb{R}^d)$ can be shown to be
$q := \binom{p + d}{d}$ (for a proof, check out chapter 2 of Wendland's
*Scattered Data Approximation*).
Let $\varphi: \mathbb{R}^d \to \mathbb{R}^q$ denote the feature map
corresponding to this polynomial basis, such that
\begin{align}
f(x) &:= \varphi(x)^\top \beta = \sum\_{j=1}^{q} \beta_j \varphi_j(x) \in \mathcal{P}_p(\mathbb{R}^d). \tag{2}
\end{align}
Every polynomial in the space can be represented via a particular choice of the
coefficient vector $\beta \in \mathbb{R}^q$. As a concrete example in the
one-dimensional setting $d = 1$ with polynomial dimension $p = 2$ we have
$\varphi(x) = \begin{bmatrix} 1 & x & x^2 \end{bmatrix}^\top$ (when we draw
connections with the kernel approach, the scaling of these basis elements will
be different, but the same ideas apply).

## Polynomial Kernel
The goal is to find a kernel function $k: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$
that induces the feature map $\varphi: \mathbb{R}^d \to \mathbb{R}^q$. I claim that the
function
\begin{align}
&k(x, z) = \left(\langle x, z \rangle + c \right)^p, &&c \geq 0 \tag{3}
\end{align}
does the trick. We'll show this for the special case $p=2$ below, but check out
these excellent [notes](https://dataminingbook.info/book_html/chap5/book.html)
for a derivation in the general case (which follows from an application of the
multinomial theorem). Proceeding with the $p=2$ case, our claim is that (3) satisfies
\begin{align}
k(x, z) &= \langle \varphi(x), \varphi(z) \rangle, \tag{4}
\end{align}
with $\varphi$ a feature map outputting vectors storing basis functions for
$\mathcal{P}_p(\mathbb{R}^d)$. To show this, we simply multiply out the square to obtain  
\begin{align}
k(x, z)
&= \left(\sum_{j=1}^{d} x_j z_j + c \right)^2 \tag{5} \newline
&= \left(\sum_{j=1}^{d} x_j z_j\right)^2 + 2c \sum_{j=1}^{d} x_j z_j + c^2 \newline
&= \sum_{j=1}^{d} x_j^2 z_j^2 + 2\sum_{j=2}^{d}\sum_{\ell=1}^{j-1} x_jz_j x_{\ell} z_{\ell} +
2c \sum_{j=1}^{d} x_j z_j + c^2 \newline
&= \sum_{j=1}^{d} x_j^2 z_j^2 +
\sum_{j=2}^{d}\sum_{\ell=1}^{j-1} (\sqrt{2}x_j x_{\ell}) (\sqrt{2}z_j z_{\ell}) +
\sum_{j=1}^{d} (\sqrt{2c}x_j) (\sqrt{2c}z_j) + c^2 \newline
&= \langle \varphi(x), \varphi(z) \rangle,
\end{align}
where
\begin{align}
\varphi(x) &:= \begin{bmatrix} x_1^2, \dots, x_d^2, \sqrt{2}x_2 x_1, \dots,
\sqrt{2}x_d x_{d-1}, \sqrt{2c}x_1, \dots, \sqrt{2c}x_d, c \end{bmatrix}^\top. \tag{6}
\end{align}
We have used the well-known formula for expanding the square of a summation.
The feature map (6) that popped out of these derivations does indeed correspond
to a basis for $\mathcal{P}_p(\mathbb{R}^d)$; the scaling of the basis functions
might look a bit weird with the $\sqrt{2}$ terms, but this is just how the
calculation works out. We have established (4) in the special case $p=2$, though
the result holds for any positive integer $p$. This is enough to verify that
(3) is a valid positive semidefinite (PSD) kernel, which avoids working with the
PSD definition directly.

To further appreciate the connection between the kernel and feature map, consider the
expression derived in (5) for $f(x) := k(x, z)$ viewed as a function of $x$ only with $z$
a fixed parameter. We see that $f(x)$ is a polynomial in the space $\mathcal{P}_p(\mathbb{R}^d)$, evaluated at $x$. All of the constants and terms involving
$z$ simply become the coefficients on the basis functions that determine the particular
polynomial. Thus, by fixing the second argument of the kernel at different inputs $z$,
we are able to produce any polynomial in the space. Since the polynomial space is
finite-dimensional, a finite set of such kernels can span the space. In other settings
where the feature space is infinite-dimensional, this will not be the case.  

# Example 1: Ridge Regression
We now compare the explicit basis function and kernel approaches in their application
to a standard regression problem. Much of the discussion here is
not actually specific to polynomials, applying more generally to arbitrary
feature maps and kernels.   

## Polynomial Basis
Let's start by considering a standard linear regression setting with training
data $\{(x^i, y^i)\}_{i=1}^{n}$, with inputs stacked in the matrix
$X \in \mathbb{R}^{n \times d}$ and the responses in
the vector $y \in \mathbb{R}^n$. Note that we are using superscripts to
index distinct vectors, while subscripts index vector dimensions.
We'll assume a regression model that is linear with respect polynomial basis
functions up to degree $p$; in other words, the space of all possible regression
functions is $\mathcal{P}_p(\mathbb{R}^d)$. We let $\varphi: \mathbb{R}^d \to \mathbb{R}^q$
be the associated feature map, scaled to align with (6).
Let $\Phi \in \mathbb{R}^{n \times q}$ denote the *feature matrix* evaluated at
the training inputs $X$; precisely, $\Phi_{ij} := \varphi_j(x^i)$. The linear
model in the polynomial basis is thus given by
\begin{align}
y &= \Phi \beta + \epsilon, &&\epsilon \sim \mathcal{N}(0, \sigma^2 I_n). \tag{7}
\end{align}
We consider the ridge regression setting, whereby the coefficients $\beta$
are estimated by minimizing the residual sum of squares, regularized by an $L_2$ penalty.
This yields
\begin{align}
\hat{\beta}
&:= \text{argmin}_{\beta} \left[\lVert y-\Phi \beta \rVert_2^2 +
\lambda \lVert \beta \rVert_2^2 \right]
&= \left(\lambda I_q + \Phi \Phi^\top \right)^{-1} \Phi^\top y \tag{8}
\end{align}
which is derived by setting the gradient of the loss equal to zero and solving
for $\beta$. To predict at a new set of inputs $\tilde{X} \in \mathbb{R}^{m \times d}$ we
first map $\tilde{X}$ to the feature matrix $\tilde{\Phi} \in \mathbb{R}^{m \times q}$,
then compute
\begin{align}
\hat{y}
&:= \tilde{\Phi} \hat{\beta}
= \tilde{\Phi}\left(\lambda I_q + \Phi^\top \Phi \right)^{-1} \Phi^\top y. \tag{9}
\end{align}
This is just run-of-the-mill ridge regression using polynomial basis functions,
but we emphasize some points that will become relevant when comparing to kernel
methods later on:
1. Computing the $m$ predictions $\hat{y}$ is $\mathcal{O}(q^3 n + qm)$. The
$q^3 n$ stems from the linear solve required to compute $\hat{\beta}$. After
model fitting, the training inputs $X$ are forgotten, the relevant information
having been encoded in the parameters $\hat{\beta}$. Therefore, the prediction
calculations scale as $qm$, independently of $n$.
2. This procedure requires estimating $q = \binom{p + d}{d}$ parameters, which
grows quite quickly in both $d$ and $p$. This can be problematic as the number
of parameters approaches or exceeds the number of observations $n$.
Regression with multivariate polynomials is thus difficult to scale to higher-dimensional
problems. One may deal with this by reducing the number of parameters
(e.g., by excluding some of the interaction terms in the basis) or employing
methods that seek to discover sparsity in $\beta$ (e.g., $L_1$ regularization).
We will shortly discuss how the kernel approach can help alleviate this difficulty.
3. In general, the matrix $\Phi^\top \Phi$ is PSD since
$x^\top \Phi^\top \Phi x = \lVert \Phi x \rVert_2^2 \geq 0$ (one can
guarantee it is positive definite (PD) under certain conditions on the input points
(see Wendland chapter 2 for details). In any case, the addition of
$\lambda I_q$ with $\lambda > 0$ ensures that $\lambda I_q + \Phi^\top \Phi$ is
positive definite (PD), and hence invertible. From a numerical perspective,
this can be useful even when $\Phi^\top \Phi$ is theoretically positive definite,
since its numerical instantiation may fail to be so.

## Polynomial Kernel
We now discuss an alternative route to compute the prediction $\hat{y}$ that eliminates
the need to estimate the $q$ parameters (there will, of course, be a new cost to pay).
The first required observation is that
\begin{align}
&&k(X, X) = \Phi \Phi^\top, &&k(\tilde{X}, X) = \tilde{\Phi} \Phi^\top \tag{10}
\end{align}
where we have vectorized notation by defining the matrices
$k(X, X) \in \mathbb{R}^{n \times n}$ and $k(\tilde{X}, X) \in \mathbb{R}^{m \times n}$
by $k(X, X)_{ij} := k(x^i, x^j)$ and $k(\tilde{X}, X)_{ij} := k(\tilde{x}^i, x^j)$.
The matrix $k(X, X)$ looks a lot like the $\Phi^\top \Phi$ term in (9), but
the order of the terms is reversed. However, we can make these terms align by
rewriting (9) via an application of the [Woodbury identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity). We actually really only need the simpler fact
\begin{align}
(\lambda I_q + UV)^{-1}U = U(\lambda I_n + VU)^{-1}, \tag{11}
\end{align}
which the linked Woodbury Wikipedia page refers to as the *push-through identity*
(I've modified it slightly by incorporating the $\lambda$). We therefore obtain
\begin{align}
\left(\lambda I_q + \Phi^\top \Phi \right)^{-1} \Phi^\top
&= \Phi^\top \left(\lambda I_n + \Phi \Phi^\top \right)^{-1}, \tag{12}
\end{align}
having applied (11) with $U := \Phi^\top$ and $V := \Phi$. Therefore, we can
write (9) as
\begin{align}
\hat{y} &= \tilde{\Phi}\left(\lambda I_q + \Phi^\top \Phi \right)^{-1} \Phi^\top y \newline
&= \tilde{\Phi} \Phi^\top \left(\lambda I_n + \Phi \Phi^\top \right)^{-1} y \newline
&= k(\tilde{X}, X) \left[\lambda I_n + k(X, X) \right]^{-1} y. \tag{13}
\end{align}
Sometimes the term
\begin{align}
\alpha := \left[\lambda I_n + k(X, X) \right]^{-1} y \tag{14}
\end{align}
is referred to as the *dual* weights (whereas $\beta$ might be called the
*primal* weights). To motivate why, let's consider the prediction $\hat{y}_{s}$, which
can be written as

\begin{align}
\hat{y}\_{s} &= k(\tilde{x}^s, X) \alpha = \sum_{i=1}^{n} \alpha_i k(\tilde{x}^s, x^i). \tag{15}
\end{align}
The sum (15) can be viewed as the analog to
\begin{align}
\hat{y}^s &= \varphi(\tilde{x}^s)^\top \hat{\beta} = \sum_{j=1}^{d} \hat{\beta}_j \varphi(\tilde{x}^s)_j. \tag{16}
\end{align}

We see that both methods generate the prediction via a linear combination of basis
functions: in (15) the $q$ basis functions come from the feature map $\varphi$,
and in (16) the $n$ basis functions are $k(\cdot, x^1), \dots, k(\cdot, x^n)$. In
the former case, there are $q$ fixed, *global* basis functions. In the latter,
the basis functions are *local*, defined with respect to each training input
$x^i$; therefore, the number of basis functions grows linearly in the number of
observations.

We close this section by enumerating some properties of the kernel approach,
serving as a comparison to the points listed in the previous section for the
explicit basis function approach.
1. Computing the $m$ predictions $\hat{y}$ is $\mathcal{O}(n^3 + n^2 m)$.
The $n^3$ comes from the linear solve $\left[\lambda I_n + k(X, X)\right]^{-1}y$.
Once this linear solve has already been computed, prediction calculations
scale like $n^2 m$, which now, in contrast to the first approach, depends on $n$.
The poor scaling in $n$ is the cost we pay with the kernel method.   
2. Notice that no parameters are explicitly estimated in this kernel approach.
Instead of compressing the information into a vector of coefficients $\hat{\beta}$,
this method stores the $n \times n$ *kernel matrix* $k(X,X)$ to be used for
prediction. The kernel approach does typically require estimating a few
*hyperparameters* (in this case, $c$), which we discuss in detail later on.
3. While the $q \times q$ matrix $\Phi^\top \Phi$ can be guaranteed to be
PD under some fairly relaxed conditions, the $n \times n$ kernel matrix
$k(X, X) = \Phi \Phi^\top$ is almost certainly not PD. However, it still is
PSD, as we noted above. This can also be seen by considering
$x^\top \Phi \Phi^\top x = \lVert \Phi^\top x \rVert_2^2 \geq 0$. Therefore,
the addition of $\lambda I_n$ with $\lambda > 0$ is crucial in this setting
to ensure that the matrix $\lambda I_n + k(X,X)$ is invertible.
{% endkatexmm %}


# Example 2: The Bayesian Perspective
In the previous section, we motivated the polynomial kernel within a ridge
regression setting. In this section we walk through similar derivations, but
from a Bayesian point of view. In particular, we compare Bayesian linear
regression using polynomial basis functions with its kernelized analog, the
latter giving rise to Gaussian process (GP) regression.

{% katexmm %}
## Polynomial Basis
Consider the following Bayesian polynomial regression model:
\begin{align}
y|\beta &\sim \mathcal{N}(\Phi \beta, \sigma^2 I_n) \newline
\beta &\sim \mathcal{N}\left(0, \frac{\sigma^2}{\lambda} I_q \right) \tag{17}
\end{align}
This is a linear Gaussian model, with posterior given by
\begin{align}
\beta | y &\sim \mathcal{N}\left(\hat{m}, \hat{C} \right), \tag{18}
\end{align}
where
\begin{align}
\hat{m} &= \hat{C} \left(\frac{1}{\sigma^2}\Phi^\top y \right)
= \left(\Phi^\top \Phi + \lambda \sigma^2 I_q \right)^{-1} \Phi^\top y \newline
\hat{C} &= \left(\frac{1}{\sigma^2} \Phi^\top \Phi + \lambda I_q \right)^{-1}
\end{align}

## Polynomial Kernel

{% endkatexmm %}
