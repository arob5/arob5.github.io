---
title: Comparing Approaches for Specifying and Estimating Gaussian Process Parameters
subtitle: I review and derive various formulas that come in handy when sequentially adding data to a Gaussian process model.
layout: default
date: 2024-01-11
keywords: Gaussian-Process
published: true
---

Gaussian processes (GP) are widely utilized across various fields, each with
their own preferences, terminology, and conventions. Some notable domains that
make significant use of GPs include
- Spatial statistics (kriging)
- Design and analysis of computer experiments (emulator/surrogate modeling)
- Bayesian optimization
- Machine learning

Even if you're a GP expert in one of these domains,
these differences can make navigating the
GP literature in other domains a bit tricky. The goal of this post is to
summarize common approaches for specifying GP distributions, and emphasize
conventions and assumptions that tend to differ across fields. By
"specifying GP distributions", what I am really talking about here is
parameterizing the mean and covariance functions that define the GP. While
GPs are non-parametric models in a certain sense, specifying and
learning the *hyperparameters* making up the mean and covariance functions
is a crucial step to successful GP applications. I will discuss popular
parameterizations for these functions, and different algorithms for learning
these parameter values from data. In the spirit of drawing connections across
different domains, I will try my best to borrow terminology from different fields,
and will draw attention to synonymous terms by using boldface.  

## Background
{% katexmm %}

### Gaussian Processes
Gaussian processes (GPs) define a probability distribution over a space of
functions in such a way that they can be viewed as a generalization of
Gaussian random vectors. Just as Gaussian vectors are defined by their
mean vector and covariance matrix, GPs are defined by a mean and covariance
*function*. We will interchangeably refer to the latter as either the
**covariance function** or **kernel**.

We will consider GPs defined over a space of functions of the form
$f: \mathcal{X} \to \mathbb{R}$, where $\mathcal{X} \subseteq \mathbb{R}^d$.
We will refer to elements $x \in \mathcal{X}$ as **inputs** or
**locations** and the images $f(x) \in \mathbb{R}$ as **outputs** or
**responses**. If the use of the word "locations" seems odd, note that in
spatial statistical settings, the inputs $x$ are often geographic coordinates.
We will denote the mean and covariance function defining the
GP by $\mu: \mathcal{X} \to \mathbb{R}$
and $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$, respectively. The mean
function is essentially unrestricted, but the covariance function $k(\cdot, \cdot)$
must be a valid positive definite kernel. If $f(\cdot)$ is a GP with
mean function $\mu(\cdot)$ and kernel $k(\cdot, \cdot)$ we will denote this by
\begin{align}
f \sim \mathcal{GP}(\mu, k). \tag{1}
\end{align}

The defining property of GPs is that their finite-dimensional distributions
are Gaussian; that is, for an arbitrary finite set of $n$ inputs
$X := \{x_1, \dots, x_N\} \subset \mathcal{X}$,
the vector $f(X) \in \mathbb{R}^n$ is distributed as
\begin{align}
f(X) \sim \mathcal{N}(\mu(X), k(X, X)). \tag{2}
\end{align}
We are vectorizing notation here so that $[f(X)]_i := f(x_i)$,
$[\mu(X)]_i := \mu(x_i)$, and $[k(X, X)]_{i,j} := k(x_i, x_j)$. When the
two input sets to the kernel are equal, we lighten notation by writing
In particular, suppose we have two sets of inputs
$X$ and $\tilde{X}$, containing $n$ and
$m$ inputs, respectively. The defining property (2) then implies

\begin{align}
\begin{bmatrix} f(\tilde{X}) \newline f(X) \end{bmatrix}
&\sim \mathcal{N}\left(
  \begin{bmatrix} \mu(\tilde{X}) \newline \mu(X) \end{bmatrix},
  \begin{bmatrix}
  k(\tilde{X}) & k(\tilde{X}, X) \newline
  k(X, \tilde{X}) & k(X)
  \end{bmatrix}
\right). \tag{3}
\end{align}

The Gaussian joint distribution (3) implies that the conditional distributions
are also Gaussian. In particular, the distribution of $f(\tilde{X})|f(X)$ can be
obtained by applying the well-known Gaussian conditioning identities:

\begin{align}
f(\tilde{X})|f(X) &\sim \mathcal{N}(\hat{\mu}(\tilde{X}), \hat{k}(\tilde{X})), \tag{4} \newline
\hat{\mu}(\tilde{X}) &:= \mu(\tilde{X}) + k(\tilde{X}, X)k(X)^{-1} [f(X) - \mu(X)] \newline
\hat{k}(\tilde{X}) &:= k(\tilde{X}) - k(\tilde{X}, X)k(X)^{-1} k(X, \tilde{X}).
\end{align}
The fact that the result (4) holds for arbitrary finite sets of inputs $\tilde{X}$
implies that the conditional $f | f(X)$ is also a GP, with mean and covariance
functions $\hat{\mu}(\cdot)$ and $\hat{k}(\cdot, \cdot)$ defined by (4).
On a terminology note, the $n \times n$ matrix $k(X)$ is often called the
**kernel matrix**. This is the matrix containing the kernel evaluations at the
set of $n$ *observed* locations.

### Regression with GPs
One common application of GPs is their use as a flexible nonlinear regression
model. Let's consider the basic regression setup with observed data pairs
$(x_1, y_1), \dots, (x_n, y_n)$. We assume that the $y_i$ are noisy observations
of some underlying latent function output $f(x_i)$. The GP regression model arises
by placing a GP prior distribution on the latent function $f$. We thus consider
the regression model
\begin{align}
y(x) &= f(x) + \epsilon(x) \tag{5} \newline
f &\sim \mathcal{GP}(\mu, k) \newline
\epsilon &\overset{iid}{\sim} \mathcal{N}(0, \sigma^2),
\end{align}
where we have assumed a simple additive Gaussian noise model. This assumption is
quite common in the GP regression setting due to the fact that it results in
closed-form conditional distributions, similar to (4). We will assume the
error model (5) throughout this post, but note that there are many other possibilities
if one is willing to abandon closed-form posterior inference.

The solution of the regression problem is given by the distribution of
$f(\cdot)|y(X)$ or $y(\cdot)|y(X)$, where $y(X)$ is the $n$-dimensional vector
of observed responses. The first distribution is the posterior on the latent
function $f$, while the second incorporates the observation noise as well.
Both distributions can be derived in the same way, so we focus on the second.
Letting $\tilde{X}$ denote a set of $m$ inputs at which we would like to predict the
response, consider the joint distribution
\begin{align}
\begin{bmatrix} y(\tilde{X}) \newline y(X) \end{bmatrix}
&\sim \mathcal{N}\left(
  \begin{bmatrix} \mu(\tilde{X}) \newline \mu(X) \end{bmatrix},
  \begin{bmatrix}
  k(\tilde{X}) + \sigma^2 I_m & k(\tilde{X}, X) \newline
  k(X, \tilde{X}) & k(X) + \sigma^2 I_n
  \end{bmatrix}
\right). \tag{6}
\end{align}
This is quite similar to (3), but now takes into account the noise term
$\epsilon$. This does not affect the mean vector since $\epsilon$ is mean-zero;
nor does it affect the off-diagonal elements of the covariance matrix since
$\epsilon$ and $f$ were assumed independent. Applying the Gaussian conditioning
identities (4) yields the posterior distribution
\begin{align}
y(\tilde{X})|y(X) &\sim \mathcal{N}(\hat{\mu}(\tilde{X}), \hat{k}(\tilde{X})), \tag{7} \newline
\hat{\mu}(\tilde{X}) &:= \mu(\tilde{X}) + k(\tilde{X}, X)[k(X) + \sigma^2 I_n]^{-1} [f(X) - \mu(X)] \newline
\hat{k}(\tilde{X}) &:= \sigma^2 I_m + k(\tilde{X}) - k(\tilde{X}, X)[k(X) + \sigma^2 I_n]^{-1} k(X, \tilde{X}).
\end{align}
We will refer to (7) as the GP **posterior**, **predictive**, or generically
**conditional**, distribution.
We observe that these equations are identical to (4), modulo the appearance
of $\sigma^2$ in the predictive mean and covariance equations. The distribution
$f(\tilde{X})|y(X)$ is identical to (7), except that the $\sigma^2 I_m$ is removed in
the predictive covariance. Again, this reflects the subtle distinction between
doing inference on the latent function $f$ vs. on the observation process $y$.

### Noise, Nuggets, and Jitters
Observe that this whole regression procedure is only slightly different from the
noiseless GP setting explored in the previous section (thanks to the Gaussian
likelihood assumption). Indeed, the conditional distribution of
$f(\tilde{X})|y(X)$ is derived from $f(\tilde{X})|f(X)$ by simply replacing
$k(X)$ with $k(X) + \sigma^2 I_n$ (obtaining the distribution
$y(\tilde{X})|y(X)$ requires the one additional step of adding $\sigma^2 I_m$
to the predictive covariance). In other words, we have simply applied standard
GP conditioning using the modified kernel matrix
\begin{align}
C(X) := k(X) + \sigma^2 I_n. \tag{8}
\end{align}
We thus might reasonably wonder if the model (5) admits an alternative
equivalent representation by defining a GP directly on the observation process $y$.
Defining such a model would require
defining a kernel $c: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ that
is consistent with (8). This route is fraught with difficulties and subtleties, which I will
do my best to describe clearly here. At first glance, it seems like the right
choice is
\begin{align}
c(x, x^\prime) := k(x, x^\prime) + \sigma^2 \delta(x, x^\prime), \tag{9}
\end{align}
where $\delta(x, x^\prime) := 1[x = x^\prime]$ is sometimes called the
**stationary white noise kernel**. Why isn't this quite right? Notice in (9)
that $\sigma^2$ is added whenever the inputs $x = x^\prime$ are equal. However,
suppose we observe multiple independent realizations of the process at the
same inputs $X$. In the regression model (9) the errors $\epsilon(x)$ are
independent across these realizations, *even at the same locations*. However,
this will not hold true in the model under (9), since $\delta(x, x^\prime)$ only
sees the values of the inputs, and has no sense of distinction across realizations.
We might try to fix this by writing something like
\begin{align}
c(x_i, x_j) := k(x_i, x_j) + \sigma^2 \delta_{ij}, \tag{10}
\end{align}
where the Delta function now depends on the labels $i, j$ instead of the values
of the inputs. In the spatial statistics literature,
it is not uncommon to see a covariance function defined like (10), but this is
basically a notational hack. A kernel is a function of two inputs from
$\mathcal{X}$ - we can't have it also depending on some side information like
the labels $i, j$. At the end of the day, (9) and (10) are attempts to
incorporate some concept of **white noise** inside the kernel itself, rather
than via a hierarchical model like (5). I would just stick with the hierarchical
model, which is easily rigorously defined and much more intuitive.

Nonetheless, one should not be surprised if expressions like (10) pop up,
especially in the spatial statistics literature. Spatial statisticians refer
to the noise term $\epsilon(x)$ as the **nugget**, and $\sigma^2$ the
**nugget variance** (sometimes these terms are conflated). In this context,
instead of representing observation noise, $\sigma^2$ is often thought of
as representing some unresolved small-scale randomness in the spatial field
itself. If you imagine sampling a field to determine the concentration of some
mineral across space, then you would hope that repeated measurements (taken around
the same time) would yield the same values. Naturally, they may not, and the
introduction of the nugget is one way to account for this.

While this discussion may seem to be needlessly abstract, we recall that the
effect of incorporating the noise term (however you want to interpret it) is to
simply replace the kernel matrix $k(X)$ with the new matrix $c(X) = k(X) + \sigma^2 I_n$.
Confusingly, there is one more reason (having nothing to do with observation error
or nuggets) that people use a matrix of the form $c(X)$ in place of $k(X)$:
numerical stability. Indeed, even though $k(X)$ is theoretically positive definite,
in practice its numerical instantiation may fail to have this property. A simple
approach to deal with this is to add a small, fixed constant $\sigma^2$ to the
diagonal of the kernel matrix. In this context, $\sigma^2$ is often called the
**jitter**. While computationally its effect is the same as the nugget, note that
its introduction is motivated very differently. The jitter is not stemming from
some sort of random white noise; it is purely a computational hack to improve
the conditioning of the kernel matrix. Check out
[this](https://discourse.mc-stan.org/t/adding-gaussian-process-covariance-functions/237/67)
thread for some entertaining debates on the use of the nugget and jitter concepts.

### Parameterized Means and Kernels
Everything we have discussed this far assumes fixed mean and covariance
functions. In practice, suitable choices for these quantities are not typically
known. Thus, the usual approach is to specify some parametric families
$\mu = \mu_{\psi}$ and $k = k_{\phi}$ and learn their parameters from data.
The parameters $\psi$ and $\phi$ are often referred to as **hyperparameters**,
since they are not the primary parameters of interest in the GP regression model.
Recalling from (5) that the GP acts as a prior distribution on the latent
function, we see that $\psi$ and $\phi$ control the specification of this
prior distribution. In addition to $\psi$ and $\phi$, the parameter
$\sigma^2$ is also typically not known. I will not wade back into the previous
section's debate in arguing whether this should be classified as
a "hyperparameter" or not. In any case, let's let
$\theta := \{\psi, \phi, \sigma^2 \}$ denote the full set of
(hyper)parameters that must be learned from data.

#### Mean Functions
The machine learning community commonly uses the simplest possible form for the
mean function: $\mu(x) \equiv 0$. This zero-mean assumption is less restrictive
than it seems, since GPs mainly derive their expressivity from the kernel.
A slight generalization is to allow a constant, non-zero mean
$\mu(x) \equiv \beta_0$, where $\beta_0 \in \mathbb{R}$.
However, constant (including zero-mean) GP priors can have some undesirable properties;
e.g., in the context of extrapolation. Sometimes one wants more flexibility, and
in these cases it is quite common to consider some sort of linear regression
model
\begin{align}
\mu(x) = h(x)^\top \beta, \tag{11}
\end{align}
where $h: \mathcal{X} \to \mathbb{R}^p$ is some feature map and $\beta \in \mathbb{R}^p$
the associated coefficient vector. For example, $h(x) = [1, x^\top]^\top$
would yield a standard linear model, and
$h(x) = [1, x_1, \dots, x_d, x_1^2, \dots, x_d^2]$ would allow for a quadratic
trend.

#### Kernels
The positive definite restriction makes defining valid covariance functions
much more difficult than defining mean functions. Thus, one typically falls back
on one of a few popular choices of known parametric kernel families (though
note that kernels can be combined in various ways to give a large variety of
options). While the goal of this post is not to explore specific kernels, in order to have
a concrete example in mind consider the following parameterization:
\begin{align}
k(x, \tilde{x}) = \alpha^2 \sum_{j=1}^{d} \left(-\frac{\lvert x^{j} - \tilde{x}^j \rvert}{\ell^j}\right)^2.
\tag{12}
\end{align}
Note that I'm using superscripts to index vector entries here.
This kernel goes by many names, including **exponentiated quadratic**,
**squared exponential**, **Gaussian**, **radial basis function**, and
**automatic relevance determination**. The parameter $\alpha^2$ is sometimes called
the **marginal variance**, or just the **scale parameter**. The parameters
$\ell^1, \dots, \ell^d$ are often called **lengthscale**, **smoothness**, or
**range** parameters, since they control the smoothness of the GP realizations
along each coordinate direction. Other popular kernels (e.g., Matérn) have
analogous parameters controlling similar features. Note that in this example
we have $\phi = \{\alpha^2, \ell^1, \dots, \ell^d \}$.

It is quite common in the
spatial statistics (and sometimes the computer experiments) literature to see
kernels written like $\alpha^2 k(\cdot, \cdot)$; in these cases $k(\cdot, \cdot)$
typically represents a *correlation* function, which becomes the covariance function
after multiplying by the marginal variance $\alpha^2$. There is an advantage in
decomposing the kernel this way when it comes to estimating the hyperparameters,
which we will discuss shortly.

### The GP (Marginal) Likelihood Function
Let's first recall the GP regression model (5)
\begin{align}
y(x) &= f(x) + \epsilon(x) \newline
f &\sim \mathcal{GP}(\mu_{\psi}, k_{\phi}) \newline
\epsilon &\overset{iid}{\sim} \mathcal{N}(0, \sigma^2),
\end{align}
where we have now explicitly added the dependence on $\psi$ and $\phi$.
This model is defined for any $x \in \mathcal{X}$. However, when estimating
hyperparameters, we will naturally be restricting the model to $X$, the finite
set of locations at which we actually have observations. Restricting to
$X$ reduces the above model to the standard (finite-dimensional)
Bayesian regression model
\begin{align}
y(X)|f(X), \theta &\sim \mathcal{N}(f(X), \sigma^2 I_n) \tag{13} \newline
f(X)|\theta &\sim \mathcal{N}(\mu_{\psi}(X), k_{\phi}(X)).
\end{align}
We could consider completing the Bayesian specification by defining a prior
on $\theta$, but we'll hold off on this for now.
Notice that the model (13) defines a joint distribution over
$[y(X), f(X)] | \theta$, with $y(X)|f(X), \theta$ representing the
likelihood of the observations at the observed input locations $X$. At present
everything is conditional on a fixed $\theta$.
Now, if we marginalize the likelihood $y(X)|f(X), \theta$ with
respect to $f(X)$ then we obtain the distribution $y(X) | \theta$. This is often
called the **marginal likelihood**, due to the fact that $f(X)$ was marginalized
out. Thanks to all the Gaussian assumptions here, the marginal likelihood
is available in closed-form. One could approach the derivation using (13) as
the starting point, but it's much easier to consider the model written out using
random variables,
\begin{align}
y(X) &= f(X) + \epsilon(X).
\end{align}
Since $f(X)$ and $\epsilon(X)$ are independent Gaussians, then their sum is also
Gaussian with mean and covariance given by
\begin{align}
\mathbb{E}[y(X)|\theta]
&= \mathbb{E}[f(X)|\theta] + \mathbb{E}[\epsilon(X)|\theta] = \mu_{\psi}(X) \newline
\text{Cov}[y(X)|\theta]
&= \text{Cov}[f(X)|\theta] + \text{Cov}[\epsilon(X)|\theta]
= k_{\phi}(X) + \sigma^2 I_n.
\end{align}
We have thus found that  
\begin{align}
y(X)|\theta \sim \mathcal{N}\left(\mu_{\psi}(X), C_{\phi, \sigma^2}(X)\right), \tag{14}
\end{align}
recalling the definition $C_{\phi, \sigma^2}(X) := k_{\phi}(X) + \sigma^2 I_n$.
We will let $\mathcal{L}(\theta)$ denote the log density of this Gaussian
distribution; i.e. the log **marginal likelihood**:
\begin{align}
\mathcal{L}(\theta)
&:= -\frac{1}{2} \log \text{det}\left(2\pi C_{\phi, \sigma^2}(X) \right) -
\frac{1}{2} (y(X) - \mu_{\psi}(X))^\top C_{\phi, \sigma^2}(X)^{-1} (y(X) - \mu_{\psi}(X)) \tag{15}
\end{align}
The function $\mathcal{L}(\theta)$ plays a central role in the typical
to hyperparameter optimization, as we will explore below. Also note that
the above derivations also apply to the noiseless setting
(i.e., $y(X) = f(X)$) by setting $\sigma^2 = 0$. In this case, the marginal
likelihood is simply the GP distribution restricted to the inputs $X$.

I have henceforth been a bit verbose with the notation in (15) to make very explicit
the dependence on the inputs $X$ and the hyperparameters. To lighten notation a
bit, we define $y_n := y(X)$, $\mu_{\psi} := \mu_{\psi}(X)$, and
$C_{\phi, \sigma^2} := C_{\phi, \sigma^2}(X)$, allowing us to rewrite (15) as
\begin{align}
\mathcal{L}(\theta)
&:= -\frac{1}{2} \log \text{det}\left(2\pi C_{\phi, \sigma^2} \right) -
\frac{1}{2} (y_n - \mu_{\psi})^\top C_{\phi, \sigma^2}^{-1} (y_n - \mu_{\psi}). \tag{16}
\end{align}
We have simply suppressed the explicit dependence on $X$ in the notation.
{% endkatexmm %}

# Hyperparameter Optimization
{% katexmm %}
We now begin to turn out attention to methods for learning the values of the
hyperparameters from data. This section starts with the most popular approach:
optimizing the marginal likelihood.  

## Maximum Marginal Likelihood, or Empirical Bayes
Recall that (16) gives the expression for the log marginal likelihood $\mathcal{L}(\theta)$, which is just the log density of $y(X)|\theta$ viewed as a function of $\theta$.
A natural approach is to set the hyperparameters $\theta$ to their values
that maximize $\mathcal{L}(\theta)$:
\begin{align}
\hat{\theta} := \text{argmax} \mathcal{L}(\theta). \tag{17}
\end{align}
At first glance, the Gaussian form of $\mathcal{L}(\theta)$ might look quite
friendly to closed-form optimization.
After all, maximum likelihood estimates of the mean and covariance of Gaussian
vectors are indeed available in closed-form. However, upon closer inspection notice
that the covariance is not being directly optimized; we are optimizing $\phi$, and
the covariance $C_{\phi, \sigma^2}$ is a *nonlinear* function of this
parameter. Thus, in general some sort of iterative numerical scheme is
is used for the optimization. Typically, gradient-based approaches are preferred,
meaning we must be able to calculate quantities like
$\frac{\partial}{\partial \phi} C_{\phi, \sigma^2}$.
The exact gradient calculations will thus depend on the choice of kernel; specifics
on kernels and optimization schemes are not the focus of this post. We will instead
focus on the high level ideas here. The general approach to GP regression
that we have outlined so far can be summarized as:
1. Solve the optimization problem (17) and fix the hyperparameters at their
optimized values $\hat{\theta}$. The hyperparameters will be fixed from
this point onward.
2. Use the GP predictive equations (7) to perform inference at a set of locations
of interest $\tilde{X}$.

One might object to the fact that we are estimating the hyperparameters from
data, and then neglecting the uncertainty in $\hat{\theta}$ during the
prediction step. It is true that this uncertainty is being ignored, but it is
also very computationally convenient to do so.
We will discuss alternatives later
on, but I would argue that this simple approach is the most commonly used
in practice today. One way to think about this strategy is in an
**empirical Bayes** context; that is, we can view this approach as an approximation
to a fully Bayesian hierarchical model, which would involve equipping the
hyperparameters with their own priors. Instead of marginalizing the hyperparameters,
we instead fix their values at their most likely values with respect to the
observed data. We are using the data to "fine tune" the GP prior distribution.
In the literature you will see this general hyperparameter optimization strategy
referred to as either **empirical Bayes**, **maximum marginal likelihood**, or
even just **maximum likelihood**.

## Special Case Closed-Form Solutions: Mean Function
As mentioned above, in general the maximization of $\mathcal{L}(\theta)$ requires
numerical methods. However, in certain cases elements of $\theta$ can be optimized
in closed-form, meaning that numerical optimization may only be required for
a subset of the hyperparameters. We start by considering closed form optimizers
for the parameters defining the mean functions.

### Constant Mean
With the choice of constant mean $\mu_{\psi}(x) \equiv \beta_0$ the log marginal
likelihood becomes

\begin{align}
\mathcal{L}(\theta)
&:= -\frac{1}{2} \log \text{det}\left(2\pi C\_{\phi, \sigma^2} \right) -
\frac{1}{2} (y_n - \beta_0 1_n)^\top C\_{\phi, \sigma^2}(X)^{-1} (y_n - \beta_0 1_n),
\end{align}

with $1_n \in \mathbb{R}^n$ denoting a vector of ones. We now consider optimizing
$\mathcal{L}(\theta)$ as a function of $\beta_0$ only. The partial derivative
with respect to the constant mean equals
\begin{align}
\frac{\partial \mathcal{L}(\theta)}{\partial \beta_0}
&= y_n^\top C\_{\phi, \sigma^2}^{-1}1_n - \beta_0 1_n^\top C\_{\phi, \sigma^2}^{-1} 1_n.    \tag{18}
\end{align}
Setting (18) equal to zero and solving for $\beta_0$ gives the optimum
\begin{align}
\hat{\beta}_0 = \frac{y_n^\top C\_{\phi, \sigma^2}^{-1} 1_n}{1_n C\_{\phi, \sigma^2}^{-1} 1_n}. \tag{18}
\end{align}
Notice that $\hat{\beta}_0$ depends on the values of the other hyperparameters
$\phi$ and $\sigma^2$. Therefore, while this does not give us the outright value
for the mean, we can plug $\hat{\beta}_0$ in place of $\beta_0$ in the marginal
likelihood. This yields the **profile likelihood**, which is no longer a function
of $\beta_0$ and hence the dimensionality of the subsequent numerical optimization
problem has been reduced.

### Linear Model Coefficients
Let's try to do the same thing with the mean function
$\mu_{\psi}(x) = h(x)^\top x$. We will find that it doesn't work out as nicely
in this case.



## Bias Corrections

{% endkatexmm %}

# Bayesian Approaches

# TODOs
- MLE vs. GLM estimate of the coefs $\beta$

# References
- Surrogates (Gramacy)
- Statistics or geostatistics? Sampling error or nugget effect? (Clark)
