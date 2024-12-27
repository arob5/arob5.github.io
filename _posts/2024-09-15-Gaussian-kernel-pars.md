---
title: Gaussian Kernel Hyperparameters
subtitle: Interpreting and optimizing the lengthscale and marginal variance parameters of a Gaussian covariance function.
layout: default
date: 2024-12-26
keywords:
published: true
---

In this post we discuss the interpretation of the hyperparameters defining
a Gaussian covariance function. We then consider some practical considerations
to keep in mind when learning the values of these parameters. In particular,
we discuss empirical methods for setting reasonable bounds and default
values.

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
That is, $k(x,z)$ describes the covariance between the ouputs $f(x)$ and
$f(z)$ as a function of the inputs $x$ and $z$. The relation (2) commonly arises
when assuming that $f$ is a Gaussian process with covariance function
(i.e., kernel) $k(\cdot, \cdot)$. Under this interpretation, note that
$$
\text{Var}[f(x)] = k(x,x) = \alpha^2, \tag{3}
$$
which justifies the terminology "marginal variance". We also note that (1)
is *isotropic*, meaning that $k(x,z)$ depends only on the distance
$\lvert x-z \rvert$.
With respect to (2), this means that the covariance between the function
values $f(x)$ and $f(z)$ decays as a function of $\lvert x-z \rvert$,
regardless of the
particular values of $x$ and $z$. We will therefore find it useful to
abuse notation by writing $k(d) = k(x,z)$, where $\lvert x-z \rvert$. To emphasize,
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
= \frac{\text{Cov}[f(x),f(z)]}{\sqrt{\text{Var}[f(x)]\text{Var}[f(z)]}}
= \frac{k(x,z)}{k(0)}
= \rho(x,z). \tag{5}
$$

## Gaussian Density Interpretation
Another way to think about this is that $\rho(x,z)$ is an
unnormalized Gaussian density centered at $z$ and evaluated at $x$; i.e.,
$$
\rho(x,z) \propto \mathcal{N}(x|z, \ell^2/2). \tag{6}
$$
The reason for the $1/2$
scaling is due to the fact that (1) is missing the typical $1/2$ factor in the
Gaussian probability density (the $1/2$ is sometimes included in the Gaussian
covariance parameterization to align it with the density). Under this
interpretation, we can think of $\rho(\cdot,z)$ as a Gaussian curve centered
at $z$, describing the dependence between $f(z)$ and all other function
values. The fact that (1) only depends on its inputs through
$\lvert x-z\rvert$ means that the shape of this Gaussian is the same for all
$z$; considering different $z$ simply shifts the location of the curve.
We can make this more explicit by consdering the correlation function to be
a function only of distance:
$$
\rho(d) \propto \mathcal{N}(d|0, \ell^2/2). \tag{7}
$$
This is another way of viewing the fact that the rate of correlation decay
is the same across the whole input space, and depends only on the distance
between points.

## Generalizing to Multiple Dimensions
To generalize (1) to a $p$-dimensional input space, we define
$k: \mathbb{R}^p \times \mathbb{R}^p \to \mathbb{R}$ as a product of
one-dimensional correlation functions:
$$
k(x,z)
:= \alpha^2 \prod_{j=1}^{p} \rho(x,z;\ell_j)
:= \alpha^2 \prod_{j=1}^{p} \exp\left[-\left(\frac{x_j-z_j}{\ell_j}\right)^2 \right]
= \alpha^2 \exp\left[-\sum_{j=1}^{p} \left(\frac{x_j - z_j}{\ell_j}\right)^2 \right]. \tag{7}
$$
The choice of a product form for the covariance implies that the covariance
decays isotropically in each dimension, with the rate of decay in dimension
$j$ controlled by $\ell_j$. There is no interaction across dimensions.
In this $p$-dimensional setting, we see that $k(x,z)$ is a function of
$\lvert x_1 - z_1\rvert, \dots, \lvert x_p - z_p\rvert$. We can thus think of
$k$ as a function $k(d_1, \dots, d_p)$ of the distances
$d_j = \lvert x_j - z_j \rvert$ in each dimension.
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
In the introduction, we already described how (1) describes the correlation
between $f(x)$ and $f(z)$ as a function of $\lvert x-z\rvert$, but we have
not yet explicitly described the role of $\ell$ in controlling the rate
of correlation decay.
Notice in (1) that increasing $\ell$ decreases the rate of decay of $k(x,z)$
as $\lvert x-z \rvert$ increases. In other words, when $\ell$ is larger then
the function values $f(x)$ and $f(z)$ will retain higher correlation even
when the inputs $x$ and $z$ are further apart. On the other hand, as
$\ell \to 0$ then $f(x)$ and $f(z)$ will be approximately uncorrelated even
when $x$ and $z$ are close. We will make the notion of "closeness" more precise
shortly. The connection to the Gaussian density in (6) and (7) provides
another perspective. Since $\ell^2/2$ is the variance of the Gaussian,
then increasing $\ell$ implies the Gaussian curve becomes more "spread out",
indicating more dependence across longer distances.

It is important to keep in mind that there are two different distances we're
working with here: (1) the "physical" distance in the input space; and (2)
the correlation distance describing dependence in the output space. The
correlation/covariance functions relate these two notions of distance. The
lengthscale parameter $\ell$ has the same units as the inputs; i.e., it is
defined in the physical space. However, it controls correlation distance as
well. Since this is what we really care about, it is useful to interpret a
specific value of $\ell$ based on its effect on the rate of correlation decay.
To interpret a model with a specific value of $\ell$, one can
simply plot the curve $\rho(d)$ as $d$ is varied; i.e.,
consider a sequence of physical distances $d_1, d_2, \dots, d_k$ and report
the values  
$$
\rho(d_1), \rho(d_2), \dots, \rho(d_k). \tag{8}
$$
In the common setting that the input space is a bounded interval $[a,b]$, then
we can choose $d_1=a$, $d_k=b$ and space the rest of the points uniformly
throughout the interval. Another value that might be worth reporting is the
distance $d$ at which the correlation is essentially zero. The correlation
is never exactly zero, but we might choose a small threshold $\epsilon$ that
we deem to be negligible. Setting
$$
\rho(d) = \exp\left[-\left(\frac{d}{\ell}\right)^2 \right] = \epsilon \tag{8}
$$
and solving for $d$, we obtain
$$
d = \ell \sqrt{\log(1/\epsilon)}, \tag{9}
$$
a value that scales linearly in $\ell$. For example, if we set
$\epsilon = 0.001$, then (9) becomes $d \approx 7\ell$. In words, this says
that the correlation becomes negligible when the physical distance is about
seven times the lengthscale.

To generalize these notions to the product covariance in (7), we can consider
producing a correlation decay plot for each dimension separately; i.e., we
can generate the values $\rho(d_i; \ell_j)$ in (8) for each
$i = 1,\dots,k$ and $j = 1,\dots,p$. While this gives us information about
the correlation decay in each dimension, we must keep in mind that the decay in
$\rho(d)$ is controlled by the *product* of
$\rho(d; \ell_j), \dots, \rho(d; \ell_p)$.

## Marginal Variance
The marginal variance is more straightforward to interpret. As we established
in (3), $\alpha^2$ is the variance of $f(x)$, which is constant across all
values of $x$. Thus, $\alpha^2$ provides information on the scale of the
function. Suppose that $f$ is a Gaussian process with covariance function
$k(\cdot,\cdot)$ and a constant mean $m$. This implies that
$f(x) \sim \mathcal{N}(m,\alpha^2)$ for any $x$. Thus,
$$
\mathbb{P}\left[\lvert f(x)-m\rvert \leq n\alpha \right]
= \Phi(n) - \Phi(-n), \tag{9}
$$
where $\Phi$ is the standard normal distribution function. In particular,
$$
\mathbb{P}\left[\lvert f(x)-m\rvert \leq 2\alpha \right] \approx 0.95, \tag{10}
$$
meaning that values of $f(x)$ falling outside of the interval
$m \pm 2\alpha$ are unlikely.
{% endkatexmm %}

# Bounds and Defaults
{% katexmm %}
We now consider setting reasonable bounds and default values for the lengthscale
and marginal variance. The motivation here is primarily on estimating the
covariance hyperparameters for Gaussian process models via maximum marginal
likelihood. In general, it is often better to regularize the optimization
by specifying prior distributions on the hyperparameters, or to go the full
Bayesian approach and sample from the posterior distribution over the
hyperparameters. We won't go into such topics here. Throughout this section,
suppose that we have observed $n$ input-output pairs
$$
y_i = f(x_i), \qquad i=1, \dots, n \tag{11}
$$
for some function of interest $f$. For now we assume that these observations
are noiseless, and $f$ is modeled as a Gaussian process with covariance
function $k(\cdot, \cdot)$. Our intention here is not to go into depth on
methods for estimating Gaussian process hyperparameters; see
[this](https://arob5.github.io/blog/2024/01/11/GP-specifications/) post for
more details on this topic. Instead, we will describe some practical
considerations for avoiding pathological behavior when estimating the
hyperparameters of the covariance function.

## Lengthscales
The lengthscale parameter $\ell$ is constrained to $(0, \infty)$. However,
note that learning the value of $\ell$ from the training data relies on the
consideration of how $y_i$ varies based on pairwise distances between the
$x_i$. Thus, there is no information in the data to inform lengthscale values
that are below the minimum, or above the maximum, observed pairwise distances.
Let $d_1, \dots, d_m$ denote the set of pairwise distances constructed from
the training inputs $x_1, \dots, x_n$. There are $m = \frac{n(n-1)}{2}$ such
distances, corresponding the the number of entries in the lower triangle
(excluding the diagonal) of an $n \times n$ matrix. 



{% endkatexmm %}
