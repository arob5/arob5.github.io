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

# Polynomial Regression
{% katexmm %}
## Basis Functions
Let's start by considering a standard linear regression setting with training
data $\{(x_i, y_i)\}_{i=1}^{n}$, stacking the inputs in the matrix
$X \in \mathbb{R}^{n \times d}$ and the responses in
the vector $y \in \mathbb{R}^n$. We'll assume a regression model that is linear w
ith respect polynomial basis functions up to degree $p$; in other words,
we are working with the function space space $\mathcal{P}_p(\mathbb{R}^d)$,
the set of all multivariate polynomials
in $d$ variables up to degree $p$. The dimension of this can be shown to be
$q := \binom{p + d}{d}$ (for a proof, check out chapter 2 of Wendland's
*Scattered Data Approximation*). Let $\varphi: \mathbb{R}^d \to \mathbb{R}^q$
denote the feature map corresponding to this polynomial basis. For example, in
the one-dimensional setting $d = 1$ with $p = 2$ we have
$\varphi(x) = \begin{bmatrix} 1 & x & x^2 \end{bmatrix}^\top$. Finally, let
$\Phi \in \mathbb{R}^{n \times q}$ denote the feature matrix evaluated at inputs
$X$; precisely, $\Phi_{ij} := \varphi(x_i)_j$. With notation out of the way,
we can write the linear model in the polynomial basis as
\begin{align}
y &= \Phi \beta + \epsilon, &&\epsilon \sim \mathcal{N}(0, \sigma^2 I_n). \tag{1}
\end{align}
We consider the ridge regression setting, whereby the coefficients $\beta$
are estimated by maximum likelihood, regularized by an $L_2$ penalty.
This yields
\begin{align}
\hat{\beta}
&:= \text{argmax}_{\beta} \left[\frac{1}{\sigma^2} \lVert y-\Phi \beta \rVert_2^2 +
\lambda \lVert \beta \rVert_2^2 \right]
&= \left(\lambda I_q + \Phi \Phi^\top \right)^{-1} \Phi^\top y, \tag{2}
\end{align}
which is derived by setting the gradient of the loss equal to zero and solving
for $\beta$. To predict at a new set of inputs $\tilde{X} \in \mathbb{R}^{m \times d}$ we
first map $\tilde{X}$ to the feature matrix $\tilde{\Phi} \in \mathbb{R}^{m \times q}$,
then compute
\begin{align}
\hat{y} &:= \tilde{\Phi} \hat{\beta} = \tilde{\Phi}\left(\lambda I_q + \Phi^\top \Phi \right)^{-1} \Phi^\top y. \tag{3}
\end{align}
This is just run-of-the-mill ridge regression using polynomial basis functions,
but we emphasize some points that will become relevant when comparing to kernel
methods later on:
1. Computing the $m$ predictions $\hat{y}$ is $\mathcal{O}(q^3 n + qm)$. The
$q^3 n$ stems from the linear solve required to compute $\hat{\beta}$. After
model fitting, the training inputs $X$ are forgotton, the relevant information
having been encoded in the parameters $\hat{\beta}$. Therefore, the prediction
calculations scale as $qm$, independently of $n$.
2. This procedure requires estimating $q = \binom{p + d}{d}$ parameters, which
grows quite quickly in both $d$ and $p$. This can be problematic as the number
of parameters approaches or exceeds the number of observations $n$.
Regression with multivariate polynomials is thus difficult to scale. One may
deal with this by reducing the number of parameters (e.g., by excluding some
of the interaction terms in the basis) or employing methods that seek to discover
sparsity in $\beta$ (e.g., $L_1$ regularization).
3. In general, the matrix $\Phi^\top \Phi$ is positive semidefinite (PSD) since
$x^\top \Phi^\top \Phi x = \lVert \Phi x \rVert_2^2 \geq 0$ (one can
guarantee it is positive definite (PD) under certain conditions on the input points;
again, see Wendland chapter 2 for details). In any case, the addition of
$\lambda I_q$ with $\lambda > 0$ ensures that $\lambda I_q + \Phi^\top \Phi$ is
positive definite, and hence invertible. From a numerical perspective,
this can be useful even when $\Phi^\top \Phi$ is theoretically positive definite,
since its numerical instantiation may fail to be so.

## Kernelizing Polynomial Regression
We now discuss an alternative route to compute the prediction $\hat{y}$ that eliminates
the need to estimate the $q$ parameters (there will, of course, be a new cost to pay).
The goal is to find a kernel function $k: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$
that induces the feature map $\varphi: \mathbb{R}^d \to \mathbb{R}^q$. I claim that the
function
\begin{align}
&k(x, z) = \left(\langle x, z \rangle + c \right)^p, &&c \geq 0 \tag{4}
\end{align}
does the trick. We'll show this for the special case $p=2$ below, but check out
these excellent [notes](https://dataminingbook.info/book_html/chap5/book.html)
for a derivation in the general case (which follows from an application of the
multinomial theorem). Proceeding with the $p=2$ case, our claim is that (4) satisfies
\begin{align}
k(x, z) &= \langle \varphi(x), \varphi(z) \rangle, \tag{5}
\end{align}
with $\phi$ denoting the multivariate polynomial feature map considered in the previous
section. To show this, we simply multiply out the square to obtain  
\begin{align}
k(x, z)
&= \left(\sum_{j=1}^{d} x_j z_j + c \right)^2 \newline
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
\phi(x) &:= \begin{bmatrix} x_1^2, \dots, x_d^2, \sqrt{2}x_2 x_1, \dots,
\sqrt{2}x_d x_{d-1}, \sqrt{2c}x_1, \dots, \sqrt{2c}x_d, c \end{bmatrix}^\top. \tag{6}
\end{align}
Note that we used the well-known formula for expanding the square of a summation.
The feature map (6) that popped out of these derivations does indeed correspond
to a basis for $\mathcal{P}_p(\mathbb{R}^d)$; the scaling of the basis functions
might look a bit weird with the $\sqrt{2}$ terms, but this is just how the
calculation works out. We have established (5) in the special case $p=2$, though
the result holds for any positive integer $p$. This is enough to verify that
(4) is a valid PSD kernel, which avoids working with the PSD definition directly.

With this established, we now see how this kernel can be used in computing
the regression estimates $\hat{y}$ in (3). The first required observation is
that
\begin{align}
&&k(X, X) = \Phi \Phi^\top, &&k(\tilde{X}, X) = \tilde{\Phi} \Phi^\top \tag{7}
\end{align}
where we have vectorized notation by defining the matrices
$k(X, X) \in \mathbb{R}^{n \times n}$ and $k(\tilde{X}, X) \in \mathbb{R}^{m \times n}$
by $k(X, X)_{ij} := k(x_i, x_j)$ and $k(\tilde{X}, X)_{ij} := k(\tilde{x}_i, x_j)$.
The matrix $k(X, X)$ looks a lot like the $\Phi^\top \Phi$ term in (3), but
the order of the terms is reversed. However, we can make these terms align by
rewriting (3) via an application of the [Woodbury identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity). We actually really only need the simpler fact
\begin{align}
(\lambda I_q + UV)^{-1}U = U(\lambda I_n + VU)^{-1}, \tag{8}
\end{align}
which the linked Woodbury Wikipedia page refers to as the *push-through identity*
(I've modified it slightly by incorporating the $\lambda$). We therefore obtain
\begin{align}
\left(\lambda I_q + \Phi^\top \Phi \right)^{-1} \Phi^\top
&= \Phi^\top \left(\lambda I_n + \Phi \Phi^\top \right)^{-1}, \tag{9}
\end{align}
having applied (8) with $U := \Phi^\top$ and $V := \Phi$. Therefore, we can
write (3) as
\begin{align}
\hat{y} &= \tilde{\Phi}\left(\lambda I_q + \Phi^\top \Phi \right)^{-1} \Phi^\top y \newline
&= \tilde{\Phi} \Phi^\top \left(\lambda I_n + \Phi \Phi^\top \right)^{-1} y \newline
&= k(\tilde{X}, X) \left[\lambda I_n + k(X, X) \right]^{-1} y. \tag{10}
\end{align}
Sometimes the term
\begin{align}
\alpha := \left[\lambda I_n + k(X, X) \right]^{-1} y \tag{11}
\end{align}
is referred to as the *dual* weights (whereas $\beta$ might be called the
*primal* weights). To motivate why, let's consider the prediction $\hat{y}_{s}$, which
can be written as

\begin{align}
\hat{y}\_{s} &= k(\tilde{x}\_{s}, X) \alpha = \sum_{i=1}^{n} \alpha_i k(\tilde{x}_s, x_i). \tag{11}
\end{align}
The sum (11) can be viewed as the analog to
\begin{align}
\hat{y}\_{s} &= \varphi(\tilde{x}\_{s})^\top \hat{\beta} = \sum_{j=1}^{d} \hat{\beta}_j \varphi(\tilde{x}\_{s})_j. \tag{12}
\end{align}

We see that both methods generate the prediction via a linear combination of basis
functions: in (12) the $q$ basis functions coming from the feature map $\varphi$,
and in (11) the $n$ basis functions $k(\cdot, x_1), \dots, k(\cdot, x_n)$. In
the former case, there are $q$ fixed, *global* basis functions. In the latter,
the basis functions are *local*, defined with respect to each training input
$x_i$; therefore, the number of basis functions grows linearly in the number of
observations.

We close this section by enumerating some properties of the kernel approach,
serving as a comparison to the points listed in the previous section.
1. Computing the $m$ predictions $\hat{y}$ is $\mathcal{O}(n^3 + n^2 m)$.
The $n^3$ comes from the linear solve $\left[\lambda I_n + k(X, X)\right]^{-1}y$.
Once this linear solve has already been completed, prediction calculations
scale like $n^2 m$, which now, in contrast to the first approach, depends on $n$.
The poor scaling in $n$ is the cost we pay with this approach.  



## Kernel Regression  
{% endkatexmm %}
