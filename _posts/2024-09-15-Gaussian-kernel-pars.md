---
title: Gaussian Kernel Hyperparameters
subtitle: Interpreting the lengthscale and marginal variance parameters of a
Gaussian covariance function.
layout: default
date: 2024-12-26
keywords:
published: false
---

In this post we discuss the interpretation of the hyperparameters defining
a Gaussian covariance function. We then describe empirical methods for setting
reasonable default values for these hyperparameters.  

# The Gaussian covariance function
{% katexmm %}

## One-Dimensional Input Space
In one dimension, the Gaussian covariance function is a function
$k: \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ defined by
$$
k(x,z) := \alpha^2 \exp\left[-\left(\frac{x-z}{\ell}\right)^2 \right]. \tag{1}
$$
We will refer to the hyperparameters $\alpha^2$ and $\ell$ as the
*marginal variance* and *lengthscale*, respectively. Notice that the range of
$k(\cdot, \cdot)$  is the interval $(0,\alpha^2]$. In interpreting (1), we
will find it useful to think of a function $f: \mathbb{R} \to \mathbb{R}$
satisfying
$$
\text{Cov}[f(x), f(z)] = k(x,z). \tag{2}
$$
That is, $k(x,z)$ describes the covariance between the function output
values as a function of the inputs $x$ and $z$. The relation (2) commonly arises
when assuming that $f$ is a Gaussian process with covariance function
(i.e., kernel) $k(\cdot, \cdot)$. Under this interpretation, note that
$$
\text{Var}[f(x)] = k(x,x) = \alpha^2, \tag{3}
$$
which justifies the terminology "marginal variance". We also note that (1)
is *isotropic*, meaning that $k(x,z)$ depends only on the distance $\abs{x-z}$.
With respect to (2), this means that the covariance between the function
values $f(x)$ and $f(z)$ decays as a function of $\abs{x-z}$, regardless of the
particular values of $x$ and $z$. We will therefore find it useful to
abuse notation by writing $k(d) = k(x,z)$, where $d := \abs{x-z}$. To emphasize,
when $k(\cdot)$ is written with only a single argument, the argument should be
interpreted as a *distance*.

It is often useful to
work with correlations instead of covariances, as they are scale-free. The
Gaussian correlation function is given by
$$
\rho(x,z) := \exp\left[-\left(\frac{x-z}{\ell}\right)^2 \right], \tag{4}
$$
so that $k(x,z) = \alpha^2 \rho(x,z)$. Note also that
$$
\text{Cor}[f(x),f(z)]
= \frac{\text{Cov}[x,z]}{\sqrt{\text{Var}[f(x)]\text{Var}[f(z)]}}
= \frac{k(x,z)}{k(0)}
= \rho(x,z). \tag{5}
$$

## Generalizing to Multiple Dimensions
To generalize (1) to a $d$-dimensional input space, we define
$k: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ as a product of
one-dimensional correlation functions:
$$
k(x,z)
:= \alpha^2 \prod_{j=1}^{d} \rho(x,z;\ell_j) \newline
:= \alpha^2 \prod_{j=1}^{d} \exp\left[-\left(\frac{x_j-z_j}{\ell_j}\right)^2 \right] \newline
= \alpha^2 \exp\left[-\left(\frac{x_j - z_j}{\ell_j}\right) \right]. \tag{5}
$$
The choice of a product form for the covariance implies that the covariance
decays isotropically in each dimension, with the rate of decay in dimension
$j$ controlled by $\ell_j$. There is no interaction across dimensions.
{% endkatexmm %}

# Interpreting the Hyperparameters
We now discuss the interpretation of the marginal variance and lengthscale
hyperparameters. Although we focus on the Gaussian covariance function here,
the marginal variance interpretation applies to any covariance function which
implies a constant variance across the input space. The interpretation of the
lengthscales is also relevant to other covariance functions that have similar
hyperparameters (e.g., the Matérn kernel).

{% katexmm %}
## Lengthscales

## Marginal Variance
{% endkatexmm %}

TODO:
1. TRY USING TWO STAGE APPROACH; FIT QUADRATIC POLYNOMIAL THEN FIT GP TO RESIDUALS.
2. Review hyperparameterization/defaults; understand what I'm currently doing.
3. Write up blog post on default hyperparameter methods (did I already include
this in my GP specification post?)
4. Get the 1d example GP fit looking reasonable. Maybe try using quadratic
mean function instead of kernel.
