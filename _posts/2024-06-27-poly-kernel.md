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
Let's start by considering a standard linear regression setting with training
data $\{(x_i, y_i)\}_{i=1}^{n}$, stacking the inputs in the matrix
$X \in \mathbb{R}^{n \times d}$ and the responses in
the vector $y \in \mathbb{R}^n$.
We'll assume a regression model that is linear with respect polynomial basis  
functions up to degree $p$; in other words, we are working with the hypothesis
space $\mathcal{P}_p(\mathbb{R}^d)$, the set of all multivariate polynomials
in $d$ variables up to degree $p$. The dimension of this can be shown to be
$q := \binom{p + d}{d}$ (for a proof, check out chapter 2 of Wendland's
*Scattered Data Approximation*). Let $\varphi: \mathbb{R}^d \to \mathbb{R}^q$
denote the feature map corresponding to this polynomial basis. For example, in
the one-dimensional setting $d = 1$ with $p = 2$ we have
$\varphi(x) = \begin{bmatrix} 1 & x & x^2 \end{bmatrix}^\top$. Finally, let
$\Phi \in \mathbb{R}^{n \times q}$ denote the feature matrix evaluated at inputs
$X$; precisely, $\Phi_{ij} := \varphi(x_i)_j$. With notation out of the way,
we consider the linear model in the polynomial basis functions
\begin{align}
y &= \Phi \beta + \epsilon, &&\epsilon \sim \mathcal{N}(0, \sigma^2 I_n). \tag{1}
\end{align}
We consider the standard procedure of estimating $\beta$ via maximum likelihood,
regularized by adding an $L_2$ penalty on the coefficients. This yields
\begin{align}
\hat{\beta}
&:= \text{argmax}_{\beta} \left[\frac{1}{\sigma^2} \lVert y-\Phi \beta \rVert_2^2 +
\lambda \lVert \beta \rVert_2^2 \right]. 
\end{align}




{% endkatexmm %}
