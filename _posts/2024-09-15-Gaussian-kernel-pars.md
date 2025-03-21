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
to keep in mind when learning the values of these parameters. In particular, we
focus on setting bounds and initialization values to avoid
common pathologies in hyperparameter optimization. The typical application
is hyperparameter estimation for Gaussian process models. It is important
to emphasize that ideally one would define informative priors on hyperparameters
and cast hyperparameter optimization within a Bayesian inferential framework.
Even if one is only optimizing hyperparameters, prior distributions offer a
principled means of regularizing the optimization problem. In this post, we
focus only on bound constraints for hyperparameters, which can be thought
of equivalently as uniform priors. In addition, we consider the setting where
little prior knowledge is available, so that they main purpose of the bounds
is to avoid pathological behavior in the optimization. We discuss heuristic
methods for empirically deriving reasonable default bounds based on the
training data. Deriving the bounds from the training data also implies that the
process can be automated, which may be useful in certain contexts.
In general, I would typically recommend considering priors
other than uniform distributions. However, I think this is still a useful exercise
to think through. Moreover, many of the more classically-oriented Gaussian process
packages (e.g., kergp and hetGP in `R`) only offer hyperparameter regularization
via bound constraints. For a discussion on non-uniform priors and full Bayesian
inference for Gaussian process hyperparameters, see Michael Betencourt's
excellent posts [here](https://betanalpha.github.io/assets/case_studies/gp_part2/part2.html)
and [here](https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html).

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
$k: \mathbb{R}^p \times \mathbb{R}^p \to \mathbb{R}$ using a product of
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
d = d(\ell) = \ell \sqrt{\log(1/\epsilon)}, \tag{9}
$$
a value that scales linearly in $\ell$. For example, if we set
$\epsilon = 0.001$, then (9) becomes $d \approx 7\ell$. In words, this says
that the correlation becomes negligible when the physical distance is about
seven times the lengthscale.

To generalize these notions to the product covariance in (7), we can consider
producing a correlation decay plot for each dimension separately; i.e., we
can generate the values $\rho(d_i; \ell_j)$ in (8) for each
$i = 1,\dots,k$ and $j = 1,\dots,p$. While this gives us information about
the correlation decay in each dimension, we must keep in mind that the decay
in $\rho(d_1, \dots, d_p)$ is controlled by the *product* of
$\rho(d_1; \ell_1), \dots, \rho(d_p; \ell_p)$.

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
y^{(i)} = f(x^{(i)}), \qquad i=1, \dots, n \tag{11}
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

### One Dimensional
We start by considering bounds
$$
\ell \in [\ell_{\text{min}}, \ell_{\text{max}}], \tag{11}
$$
for the lengthscale parameter. I should start by noting that these bounds should
be informed by domain knowledge, if it is available. We will consider the
case where there is no prior knowledge, and instead consider empirical
approaches based on the training data.
In general, $\ell$ is constrained
to the interval $(0, \infty)$. However,
note that learning the value of $\ell$ from the training data relies on the
consideration of how $y_i$ varies based on pairwise distances between the
$x^{(i)}$. Thus, there is no information in the data to inform lengthscale values
that are below the minimum, or above the maximum, observed pairwise distances.
Let $d^{(1)}, \dots, d^{(m)}$ denote the set of pairwise distances constructed
from the training inputs $x_1, \dots, x_n$. There are $m = \frac{n(n-1)}{2}$
such distances, corresponding the the number of entries in the lower triangle
(excluding the diagonal) of an $n \times n$ matrix. Thus, a reasonable first
step is to enforce the constraint
$$
\ell_{\text{min}} := d_{\text{min}}, \qquad \ell_{\text{max}} := d_{\text{max}}, \tag{12}
$$
where $d_{\text{min}}$ and $d_{\text{max}}$ denote the minimum and maximum of
$\{d^{(1)}, \dots, d^{(m)}\}$, respectively. Without this constraint, an
optimizer can sometimes get stuck in local minima at very small or large
lengthscales, which can lead to pathological Gaussian process fits. As
mentioned previously, it is typically best to think in terms of correlations,
so let's consider the implications of the choice (12) from this viewpoint.
If $\ell = \ell_{\text{min}}$, the correlation at the minimum
observed pairwise distance is given by
$$
\rho(d_{\text{min}}; \ell_{\text{min}})
= \rho(d_{\text{min}}; d_{\text{min}})
= \exp\left[-\left(\frac{d_{\text{min}}}{d_{\text{min}}}\right)^2 \right]
= \exp(-1) \approx 0.37.
$$
Thus, the lower bound in (12) implies that we are constraining the lengthscale
to satisfy
$$
\rho(d_{\text{min}}; \ell) \geq 0.37. \tag{13}
$$
We can generalize this idea and instead set $\ell_{\text{min}}$ to be the
value that achieves the constraint
$$
\rho(d_{\text{min}};\ell_{\text{min}}) = \rho_{\text{min}}, \tag{14}
$$
for some value $\rho_{\text{min}}$ we are free to specify. Solving (14) for
$\ell_{\text{min}}$, we obtain
$$
\ell_{\text{min}}
= \frac{d_{\text{min}}}{\sqrt{\log\left(1/\rho_{\text{min}}\right)}}. \tag{15}
$$
Note that setting $\rho_{\text{min}} < 0.37$ loosens the bound, relative to
(12), and vice versa; that is,
\begin{align}
&\rho_{\text{min}} < 0.37 \implies \ell_{\text{min}} < d_{\text{min}} \newline
&\rho_{\text{min}} > 0.37 \implies \ell_{\text{min}} > d_{\text{min}}
\end{align}
We can generalize this even further by replacing $d_{\text{min}}$ and
$d_{\text{max}}$ by empirical quantiles of $\{d^{(1)}, \dots, d^\text{(m)}\}$.
For example, we might consider enforcing (14) where the minimum
$d_{\text{min}}$ is replaced by the $5^{\text{th}}$ percentile of the
observed pairwise distances. Making this replacement without changing
$\rho_{\text{min}}$ will result in a more restrictive bound; i.e., a larger
value for $\ell_{\text{min}}$.

### Multiple Input Dimensions
We consider two different approaches to generalize the above procedure to
$p$-dimensional input space.

#### Dimension-by-Dimension
In considering bounds for the lengthscales $\ell_1, \dots, \ell_p$ in the
product correlation (7), one approach is to simply repeat the
procedure independently for each input dimension. In other words, we consider
the pairwise distances $d^{(1)}_{j}, \dots, d^{(n)}_{j}$ in the $j^{\text{th}}$
dimension, where each distance is of the form
$\lvert x_j^{(i)} - x_j^{(l)} \rvert$ for some $i \neq l$. The bounds for the
parameter $\ell_j$ can then be constructed as described above with respect to
these pairwise distances. Let $d_{\text{min},j}$ be the minimum (or some
other quantile) of $d^{(1)}_{j}, \dots, d^{(n)}_{j}$. This procedure then
implies the dimension-by-dimension constraint
$$
\rho(d_{\text{min},j}; \ell_j) \geq \rho_{\text{min}}, \qquad j=1, \dots, p \tag{16}
$$
which in turn implies
$$
\rho(d_{\text{min},1}, \dots, d_{\text{min},p}) \geq \rho_{\text{min}}^{p}. \tag{17}
$$
For example, if we choose $\rho_{\text{min}} = 0.37$ in $p=5$ dimensions, then
$\rho_{\text{min}}^5 \approx 0.007$. It is important to note that (16) does not
provide a direct constraint on the minimum correlation at the distance
$D_\text{min} := \min_{i \neq l} \lVert x^{(i)} - x^{(l)} \rVert_2$.
There is a distinction between the pairs of points that are closest in
each input dimension separately, and the pair that is closest in $\mathbb{R}^p$.
The distances $d_{\text{min},j}$ in (16) may be coming from different pairs of
inputs, not the particular pair of inputs that is closest in $\mathbb{R}^p$.
For this portion of the post, we will use capital "D" for distances in the
multivariate space, and lower case "d_j" for univariate distance with respect
to the $j^{\text{th}}$ dimension.

#### Multivariate Constraint
We might instead choose to constrain the correlation at the distance
$D_\text{min} = \min_{i \neq l} \lVert x^{(i)} - x^{(l)} \rVert_2$,
which means we are now considering Euclidean distance in $\mathbb{R}^p$,
rather than approaching the problem one dimension at a time. For the lower bound,
this means enforcing the constraint
$$
\rho(x_{\star}, x_{\star}^\prime; \ell_{\text{min},1}, \dots, \ell_{\text{min},p}) = \rho_{\text{min}}, \tag{18}
$$
where $(x_{\star}, x_{\star}^\prime)$ are a pair of training inputs
satisfying $\lVert x_{\star} - x_{\star}^\prime \rVert_2 = D_{\text{min}}$; i.e., the
two closest points. In order to choose the lower bounds $\ell_{\text{min},j}$,
a reasonable approach is to assume the $\ell_{\text{min},j}$ scale linearly
with the scale of the respective input dimension. Equivalently, we can
consider scaling the inputs so that $x \in [0,1]^p$.
In particular, for each dimension $j$ we re-scale as
$$
\tilde{x}_j := \frac{x_j - m_j}{M_j - m_j}, \tag{19}
$$
where we have defined
\begin{align}
&m_j := \min_{i} x^{(i)}\_j, &&M_j := \max_{i} x^{(i)}_j. \tag{20}
\end{align}
We'll use tildes to denote quantities pertaining to the scaled space, and
also stop writing the "min" subscript in $\ell_{\text{min},j}$ for succinctness.
Note that this scaling implies that univariate distances scale as
$$
\tilde{d}_j
:= \tilde{x}_j - \tilde{x}_j^\prime
= \frac{x_j - m_j}{M_j - m_j} - \frac{x^\prime_j - m_j}{M_j - m_j}
= \frac{x_j - x_j^\prime}{M_j - m_j}
= \frac{d_j}{M_j - m_j}. \tag{21}
$$
Since lengthscales correspond to distances, we have
$$
\tilde{\ell}_j := \frac{\ell_j}{M_j - m_j}, \tag{22}
$$
Note that these scalings have no effect on the correlations, since
$$
\rho(\tilde{x},\tilde{x}^\prime; \tilde{\ell}_1, \dots, \tilde{\ell}_p)
= \exp\left[-\sum_{j=1}^{p} \left(\frac{\tilde{d}_j}{\tilde{\ell}_j}\right)^2 \right]
= \exp\left[-\sum_{j=1}^{p} \left(\frac{d_j}{\ell_j}\right)^2 \right]
= \rho(x,x^\prime; \ell_1, \dots, \ell_p), \tag{23}
$$
given that the $M_j - m_j$ factors cancel in the ratio. If we consider an
isotropic model in scaled space
(i.e., $\tilde{\ell} \equiv \tilde{\ell}_j \ \forall j$), then this is equivalent
to assuming that the lengthscale bounds scale linearly as
$$
\ell_j = (M_j - m_j)\tilde{\ell}. \tag{24}
$$
If we enforce the constraint
$$
\rho(\tilde{x}_{\star}, \tilde{x}^\prime_{\star}; \tilde{\ell}, \dots, \tilde{\ell}) = \rho_{\text{min}} \tag{25}
$$
in scaled space, then we see from (23) that this encodes the constraint
(18), as desired.

## Marginal Variance
We now consider the definition of reasonable default bounds for the marginal
variance parameter $\alpha^2$. Throughout this section, we assume that $f(x)$
is a zero-mean Gaussian process with a Gaussian covariance function.
A reasonable approach here is to cap $\alpha^2$
so that most of the prior probability falls within the observed range of the
data. As always, such a heuristic could present challenges in generalizing
beyond the training data, and domain knowledge should take precedent in
constraining the value of $\alpha^2$.

To make precise the notion of placing most prior mass over the observed
data range, let $y_{\text{max}} := \max_i \lvert y_i \rvert$. Then we can define
the upper bound $\alpha^2_{\text{max}}$ as the value of $\alpha$ implying
$$
\mathbb{P}\left[\lvert f(x)\rvert \leq y_{\text{max}} \right] = p, \tag{26}
$$
for some (large) probability $p$ (e.g., $p=0.99$). This constraint excludes
values $\alpha > \alpha_{\text{max}}$, which would lead to priors that place
more probability on values of $\lvert f(x) \rvert$ that exceed $y_{\text{max}}$.
The use of the value $y_{\text{max}}$ might lead to some concerns around
robustness, since a single extreme value $y_i$ could exert significant
influence on the bound $\alpha_{\text{max}}$. On the flip side, we also don't
want to simply ignore the larger values and set $\alpha_{\text{max}}$
restrictively low. Similar to the lengthscale bound approach, one option
is to replace $y_{\text{max}}$ with a different empirical quantile, and
pair this with a slightly lower value of $p$.

In any case, for a chosen quantile $y_{\text{max}}$ and probability $p$,
we can solve the expression (26) for $\alpha$ to obtain an expression
for the bound $\alpha_{\text{max}}$. Noting that
$f(x) \sim \mathcal{N}(0, \alpha^2)$, we see that (26) can equivalently
be written as
$$
\mathbb{P}\left[-y_{\text{max}} \leq \alpha Z \leq y_{\text{max}} \right] = p, \tag{27}
$$
where $Z \sim \mathcal{N}(0,1)$. Since Gaussians are symmetric, we can write
(27) as
$$
1 - 2 \mathbb{P}\left[Z \geq y_{\text{max}}/\alpha \right] = p. \tag{28}
$$
Solving (28) for $\alpha$ then gives
$$
\alpha = \frac{y_{\text{max}}}{\Phi^{-1}\left(\frac{1+p}{2}\right)}, \tag{29}
$$
recalling that $\Phi(t) := \mathbb{P}[Z \leq t]$ denotes the standard normal
distribution function. 

{% endkatexmm %}
