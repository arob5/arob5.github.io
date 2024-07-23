---
title: Gaussian Process Priors, Specification and Parameter Estimation
subtitle: A deep dive into hyperparameter specifications for GP mean and covariance functions, including both frequentist and Bayesian methods for hyperparameter estimation.
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
$k(X) := k(X, X)$.
Now suppose we have two sets of inputs
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
\epsilon(x) &\overset{iid}{\sim} \mathcal{N}(0, \sigma^2),
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
doing inference on the latent function $f$ versus on the observation process $y$.

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

While this discussion may seem needlessly abstract, we recall that the
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
$h(x) = [1, x_1, \dots, x_d, x_1^2, \dots, x_d^2]^\top$ would allow for a quadratic
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
we have $\phi = \{\alpha^2, \ell^1, \dots, \ell^d \}$. Also note that people
choose to parameterize the Gaussian kernel in many different ways; for example,
it's not uncommon to see a $1/2$ factor included inside the exponential to make
the kernel align with the typical parameterization of the Gaussian probability
density function. Knowing which parameterization you're working with is important
for interpreting the hyperparameters, specifying bounds, defining priors, etc.
  
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
The function $\mathcal{L}(\theta)$ plays a central role in the typical approach
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
on, but this simple approach is probably the most commonly used
in practice today. One way to think about this strategy is in an
**empirical Bayes** context; that is, we can view this approach as an approximation
to a fully Bayesian hierarchical model, which would involve equipping the
hyperparameters with their own priors. Instead of marginalizing the hyperparameters,
we instead fix them at their most likely values with respect to the
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

### Constant Mean: Plug-In MLE
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
\hat{\beta}_0(\phi, \sigma^2) = \frac{y_n^\top C\_{\phi, \sigma^2}^{-1} 1_n}{1_n C\_{\phi, \sigma^2}^{-1} 1_n}. \tag{19}
\end{align}
Notice that $\hat{\beta}_0(\phi, \sigma^2)$ depends on the values of the other hyperparameters
$\phi$ and $\sigma^2$. Therefore, while this does not give us the outright value
for the mean, we can plug $\hat{\beta}_0(\phi, \sigma^2)$ in place of $\beta_0$ in the marginal
likelihood. This yields the **profile likelihood** (aka the **concentrated likelihood**),
which is no longer a function of $\beta_0$ and hence the dimensionality of the subsequent numerical optimization problem has been reduced.

### Linear Model Coefficients: Plug-In MLE
Let's try to do the same thing with the mean function $\mu_{\psi}(x) = h(x)^\top \beta$.
The constant mean function is actually just a special case of this more general
setting, but its common enough that it warranted its own section.
If we denote by $H \in \mathbb{R}^{n \times p}$ the feature matrix with rows equal
to $h(x_i)^\top$, $i = 1, \dots, n$ then the marginal likelihood becomes
\begin{align}
\mathcal{L}(\theta)
&:= -\frac{1}{2} \log \text{det}\left(2\pi C_{\phi, \sigma^2} \right) -
\frac{1}{2} (y_n - H\beta)^\top C_{\phi, \sigma^2}^{-1} (y_n - H\beta), \tag{20}
\end{align}
with gradient
\begin{align}
\nabla_{\beta} \mathcal{L}(\theta)
&= H^\top C_{\phi, \sigma^2}^{-1}y_n - (H^\top C_{\phi, \sigma^2}^{-1} H)\beta. \tag{21}
\end{align}
Setting the gradient equal to zero and solving for $\beta$ yields the optimality
condition
\begin{align}
\left(H^\top C_{\phi, \sigma^2}^{-1} H\right)\hat{\beta} &= H^\top C_{\phi, \sigma^2}^{-1}y_n. \tag{22}
\end{align}
A unique solution for $\hat{\beta}$ thus exists when $H^\top C_{\phi, \sigma^2}^{-1} H$
is invertible. When does this happen? First note that this matrix is positive
semidefinite, since
\begin{align}
\beta^\top \left(H^\top C_{\phi, \sigma^2}^{-1} H\right) \beta
&= \beta^\top (H^\top [LL^\top]^{-1} H) \beta
= \lVert L^{-1} H\beta \rVert_2^2 \geq 0,
\end{align}
where we have used the fact that $C_{\phi, \sigma^2}$ is positive definite and
hence admits a decomposition $LL^\top$. The matrix $H^\top C_{\phi, \sigma^2}^{-1} H$
is thus positive definite when $L^{-1}H$ has linearly independent columns; i.e., when
it is full rank. We already know that $L^{-1}$ is full rank. If we assume that
$H$ is also full rank and $n \geq p$ then we can conclude that $L^{-1}H$ is
full rank; see [this](https://math.stackexchange.com/questions/272049/rank-of-matrix-ab-when-a-and-b-have-full-rank) post for a quick proof. Thus, under these assumptions we conclude
that $H^\top C_{\phi, \sigma^2}^{-1} H$ is invertible and so
\begin{align}
\hat{\beta}(\phi, \sigma^2) &= \left(H^\top C_{\phi, \sigma^2}^{-1} H\right)^{-1} H^\top C_{\phi, \sigma^2}^{-1}y_n.
\tag{23}
\end{align}
Notice that (23) is simply a [generalized least squares](https://en.wikipedia.org/wiki/Generalized_least_squares) estimator. As with the constant mean, we can plug
$\hat{\beta}(\phi, \sigma^2)$ into the marginal likelihood to concentrate out
the parameter $\beta$. The resulting concentrated likelihood can then be numerically
optimized as a function of the remaining hyperparameters.

### Linear Model Coefficients: Closed-Form Marginalization
The above section showed that, conditional on fixed kernel hyperparameters,
the coefficients of a linear mean function can be optimized in closed form.
We now show a similar result: if the mean coefficients are assigned a Gaussian
prior then, conditional on fixed kernel hyperparameters, the coefficients can
be marginalized in closed form. To this end, we consider the same linear mean
function as above, but now equip the coefficients with a Gaussian prior:
\begin{align}
\mu_{\psi}(x) &= h(x)^\top \beta, &&\beta \sim \mathcal{N}(b, B).
\end{align}
Restricted to the model inputs $X$, the model is thus
\begin{align}
y_n|\beta &\sim \mathcal{N}\left(H\beta, C_{\phi, \sigma^2} \right) \newline
\beta &\sim \mathcal{N}(b, B).
\end{align}
Our goal here is derive the marginal distribution of $y_n$. We could resort to
computing the required integral by hand, but an easier approach is to notice
that under the above model $[y_n, \beta]$ is joint Gaussian distributed.
Therefore, the marginal distribution of $y_n$ must also be Gaussian. It thus
remains to identify the mean and covariance of this distribution. We obtain  
\begin{align}
\mathbb{E}[y_n]
&= \mathbb{E}\mathbb{E}[y_n|\beta] = \mathbb{E}[H\beta] = Hb \newline
\text{Cov}[y_n]
&= \mathbb{E}[y_n y_n^\top] - \mathbb{E}[y_n]\mathbb{E}[y_n]^\top \newline
&= \mathbb{E} \mathbb{E}\left[y_n y_n^\top | \beta\right] - (Hb)(Hb)^\top \newline
&= \mathbb{E}\left[\text{Cov}[y_n|\beta] + \mathbb{E}[y_n|\beta] \mathbb{E}[y_n|\beta]^\top \right] - Hbb^\top H^\top \newline
&= \mathbb{E}\left[C_{\phi, \sigma^2} + (H\beta)(H\beta)^\top \right] - Hbb^\top H^\top \newline
&= C_{\phi, \sigma^2} + H\mathbb{E}\left[\beta \beta^\top \right]H^\top - Hbb^\top H^\top \newline
&= C_{\phi, \sigma^2} + H\left[B + bb^\top \right]H^\top - Hbb^\top H^\top \newline
&= C_{\phi, \sigma^2} + HBH^\top,
\end{align}
where we have used the law of total expectation and the various equivalent
definitions for the covariance matrix. To summarize, we have found that the
above hierarchical model implies the marginal distribution
\begin{align}
y_n &\sim \mathcal{N}\left(Hb, C_{\phi, \sigma^2} + HBH^\top \right).
\end{align}
Since this holds for any set of inputs, we obtain the analogous result for the
GP prior:
\begin{align}
y(x) &= f(x) + \epsilon(x) \newline
f &\sim \mathcal{GP}\left(\mu^\prime, k^\prime \right) \newline
\epsilon(x) &\overset{iid}{\sim} \mathcal{N}(0, \sigma^2),
\end{align}
where
\begin{align}
\mu^\prime(x) &= h(x)^\top b \newline
k^\prime(x_1, x_2) &= k(x_1, x_2) + h(x_1)^\top B h(x_2).
\end{align}
After marginalizing, we again end up with a mean function that is linear in the
basis functions $h(\cdot)$. The basis function coefficients are now given by
the prior mean $b$. The mean $b$ is something that we can prescribe, or we could
again entertain an empirical Bayes approach to set its value. Note that we have
descended another step in the hierarchical ladder. The kernel that appears from
the marginalization is now a sum of two kernels: the original kernel $k$ and
the kernel $h(x_1)^\top B h(x_2)$. The latter can be viewed as a linear kernel
in the transformed inputs $h(x_1)$, $h(x_2)$ and weighted by the positive
definite matrix $B$. It serves to account for the uncertainty in the coefficients
of the mean function.

## Special Case Closed-Form Solutions: Marginal Variance
We now consider a closed-form plug-in estimate for the marginal variance
$\alpha^2$, as mentioned in (12). The takeaway from this section will be that
a closed-form estimate is only available when the covariance matrix appearing
in the marginal likelihood (16) is of the form
\begin{align}
C_{\phi} &= \alpha^2 C. \tag{24}
\end{align}
This holds for any kernel of the form $\alpha^2 k(\cdot, \cdot)$ provided
that $\sigma^2 = 0$. For example, the exponentiated quadratic kernel in
(12) satisfies this requirement.
With this assumption, the marginal likelihood is given by
\begin{align}
\mathcal{L}(\theta)
&= -\frac{n}{2} \log\left(2\pi \alpha^2 \right) - \frac{1}{2}\log\text{det}(C) -
\frac{1}{2\alpha^2} (y_n - \mu_{\psi})^\top C^{-1} (y_n - \mu_{\psi}). \tag{25}
\end{align}
The analytical derivations given below go through for a log marginal likelihood
of this form. However, this doesn't work for the common setting with an observation
variance $\sigma^2 > 0$, since in this case the covariance assumes
the form
\begin{align}
C &= \left(\alpha^2 k(X) + \sigma^2 I_n \right).
\end{align}
This can be addressed via the simple reparameterization
\begin{align}
\tilde{\alpha}^2 C &:= \tilde{\alpha}^2\left(k(X) + \tilde{\sigma}^2 I_n \right).
\end{align}
This gives the required form of the covariance, and maintains the same number
of parameters as before. The one downside is that we lose the straightforward
interpretation of the noise variance; the observation noise is now given by
the product $\tilde{\alpha}^2 \tilde{\sigma}^2$ instead of being encoded in
the single parameter $\sigma^2$. This
reparameterization is utilized in the R package [hetGP](https://cran.r-project.org/package=hetGP).


### Plug-In MLE
Let's consider optimizing the log marginal likelihood with respect to $\alpha^2$.
The partial derivative of (25) with respect to $\alpha^2$ is given by
\begin{align}
\frac{\partial \mathcal{L}(\theta)}{\partial \alpha^2}
&= -\frac{n}{2}\frac{2\pi}{2\pi \alpha^2} - \frac{(y_n - \mu_{\psi})^\top C^{-1} (y_n - \mu_{\psi})}{2\alpha^4} \newline
&= -\frac{n}{2\alpha^2} - \frac{(y_n - \mu_{\psi})^\top C^{-1} (y_n - \mu_{\psi})}{2\alpha^4}.
\end{align}
Setting this expression equal to zero and solving for $\alpha^2$ yields
\begin{align}
\hat{\alpha}^2 &= \frac{(y_n - \mu_{\psi})^\top C^{-1} (y_n - \mu_{\psi})}{n}.
\end{align}
Following the same procedure as before, the estimate $\hat{\alpha}^2$ can be
subbed in for $\alpha^2$ in $\mathcal{L}(\theta)$ to obtain the
concentrated log marginal likelihood.

### Closed-Form Marginalization


## Bias Corrections

{% endkatexmm %}

# Bayesian Approaches

# Computational Considerations
## Log Marginal Likelihood
We start by considering the computation of the log marginal likelihood (16),
\begin{align}
\mathcal{L}(\theta)
&= -\frac{n}{2} \log(2\pi) -\frac{1}{2} \log \text{det}\left(C \right) -
\frac{1}{2} (y - \mu)^\top C^{-1} (y - \mu),
\end{align}
where we now suppress all dependence on hyperparameters in the notation for
succinctness.
Since $C = k(X) + \sigma^2 I_n$ is positive definite, we may Cholesky decompose it as
$C = L L^\top$. Plugging this decomposition into the log marginal likelihood yields
\begin{align}
\mathcal{L}(\theta)
&= -\frac{n}{2} \log(2\pi) - \frac{1}{2} \log\text{det}\left(C\right) -
\frac{1}{2} (y_n - \mu)^\top \left(LL^\top \right)^{-1} (y_n - \mu).
\end{align}
The log determinant and the quadratic term can both be conveniently written in terms
of the Cholesky factor. These terms are given respectively by
\begin{align}
\log\text{det}\left(LL^\top\right)
&= \log\text{det}\left(L\right)^2
= 2 \log \prod_{i=1}^{n} L_{ii}
= 2 \sum_{i=1}^{n} \log\left(L_{ii} \right),
\end{align}
and
\begin{align}
(y_n - \mu)^\top \left(LL^\top \right)^{-1} (y_n - \mu)
&= (y_n - \mu)^\top \left(L^{-1}\right)^\top L^{-1} (y_n - \mu)
= \lVert L^{-1}(y - \mu)\rVert_2^2.
\end{align}
The linear solve $L^{-1}(y - \mu)$ can be computed in $\mathcal{O}(n^2)$ by
exploiting the fact that the linear system has lower triangular structure.
Plugging these terms back into the log marginal
likelihood gives
\begin{align}
\mathcal{L}(\theta)
&= -\frac{n}{2} \log(2\pi) - \sum_{i=1}^{n} \log\left(L_{ii}\right) -
\frac{1}{2} \lVert L^{-1}(y - \mu)\rVert_2^2.
\end{align}
Note that the Cholesky factor $L$ is a function of $\phi$ and $\sigma^2$ and hence
must be re-computed whenever the kernel hyperparameters or noise variances
change.

## Profile Log Marginal Likelihood with Linear Mean Function
We now consider computation of the concentrated marginal log-likelihood under
a mean function of the form (11), $\mu(x) = h(x)^\top \beta$, where the generalized
least squares (GLS) estimator $\hat{\beta} = \left(H^\top C^{-1} H\right)^{-1} H^\top C^{-1}y$
(see (23)) is inserted in place of $\beta$. We are thus considering the profile log
marginal likelihood
\begin{align}
\mathcal{L}(\theta)
&= -\frac{n}{2} \log(2\pi) -\frac{1}{2} \log \text{det}\left(C \right) -
\frac{1}{2} (y - H\hat{\beta})^\top C^{-1} (y - H\hat{\beta}).
\end{align}
We will derive a numerically stable implementation of this expression in two steps,
first applying a Cholesky decomposition (as in the previous section), and then
leveraging a QR decomposition as in a typical ordinary least squares (OLS)
computation. We first write $\hat{\beta}$ in terms of the Cholesky factor $L$,
where $C = LL^\top$:
\begin{align}
\hat{\beta}
&= \left(H^\top C^{-1} H\right)^{-1} H^\top C^{-1}y \newline
&= \left(H^\top \left[LL^\top\right]^{-1} H\right)^{-1} H^\top \left[LL^\top\right]^{-1}y \newline
&= \left(\left[L^{-1}H \right]^\top \left[L^{-1}H \right] \right)^{-1} \left[L^{-1}H \right]^\top
\left[L^{-1}y\right].
\end{align}
Notice that the GLS computation boils down to two lower-triangular linear solves:
$L^{-1}H$ and $L^{-1}y$. However, the above expression still requires one non-triangular
linear solve that we will now address via the QR decomposition. The above expression
for $\hat{\beta}$ can be viewed as a standard OLS estimator with design matrix
$L^{-1}H$ and response vector $L^{-1}y$. At this point, we could adopt a standard
OLS technique of taking the QR decomposition of the design matrix $L^{-1}H$.
This was my original thought, but I found a nice alternative looking through the
code in the R [kergp](https://github.com/cran/kergp/blob/master/R/logLikFuns.R)
package (see the function `.logLikFun0` in the file `kergp/R/logLikFuns.R`). The approach
is to compute the QR decomposition
\begin{align}
\begin{bmatrix} L^{-1}H & L^{-1}y \end{bmatrix} &= QR = Q \begin{bmatrix} \tilde{R} & r \end{bmatrix}.
\end{align}
That is, we compute QR on the matrix formed by concatenating $L^{-1}y$ as an additional
column on the design matrix $L^{-1}H$. We have written the upper triangular matrix
$R \in \mathbb{R}^{(p+1) \times (p+1)}$ as the concatenation of
$\tilde{R} \in \mathbb{R}^{(p+1) \times p}$ and the vector
$r \in \mathbb{R}^{p+1}$ so that $L^{-1}H = Q\tilde{R}$ and $L^{-1}y = Qr$.
We recall the basic properties of the QR decomposition: $R$ is upper triangular
and invertible, and $Q$ has orthonormal columns with span equal to the column space
of $\begin{bmatrix} L^{-1}H & L^{-1}y \end{bmatrix}$. Taking the QR decomposition
of this concatenated matrix leads to a very nice expression for the quadratic
form term of the profile log marginal likelihood. But first let's rewrite $\hat{\beta}$
in terms of these QR factors:
\begin{align}
\hat{\beta}
&= \left(\left[L^{-1}H \right]^\top \left[L^{-1}H \right] \right)^{-1} \left[L^{-1}H \right]^\top
\left[L^{-1}y\right] \newline
&= \left(\left[Q\tilde{R} \right]^\top \left[Q\tilde{R} \right] \right)^{-1} \left[Q\tilde{R} \right]^\top \left[Qr\right] \newline
&= \left(\tilde{R}^\top Q^\top Q\tilde{R} \right)^{-1} \tilde{R}^\top Q^\top Qr \newline
&= \left(\tilde{R}^\top \tilde{R} \right)^{-1} \tilde{R}^\top r,
\end{align}
where we have used the fact that $Q^\top Q$ is the identity since $Q$ is orthogonal.
Plugging this into the quadratic form term of the log likelihood gives
\begin{align}
(y - H\hat{\beta})^\top C^{-1} (y - H\hat{\beta})
&= (y - H\hat{\beta})^\top \left[LL^\top \right]^{-1} (y - H\hat{\beta}) \newline
&= \lVert L^{-1}(y - H\hat{\beta}) \rVert_2^2 \newline
&= \lVert L^{-1}y - L^{-1}H\hat{\beta} \rVert_2^2 \newline
&= \lVert Qr - Q\tilde{R} \hat{\beta} \rVert_2^2 \newline
&= \left\lVert Qr - Q\tilde{R} \left(\tilde{R}^\top \tilde{R} \right)^{-1} \tilde{R}^\top r \right\rVert_2^2 \newline
&= \left\lVert Q\left[r - \tilde{R} \left(\tilde{R}^\top \tilde{R} \right)^{-1} \tilde{R}^\top r \right] \right\rVert_2^2 \newline
&= \left\lVert r - \tilde{R} \left(\tilde{R}^\top \tilde{R} \right)^{-1} \tilde{R}^\top \right\rVert_2^2 \newline
&= \left\lVert \left[I - \tilde{R} \left(\tilde{R}^\top \tilde{R} \right)^{-1} \tilde{R}^\top \right]r \right\rVert_2^2,
\end{align}
where the penultimate line follows from the fact that $Q$ is orthogonal, and hence an
isometry. At this point, notice that the matrix
$P := \tilde{R} \left(\tilde{R}^\top \tilde{R} \right)^{-1} \tilde{R}^\top$ is the standard OLS projection matrix (i.e., hat matrix)
constructed with the design matrix $\tilde{R}$. Also, take care to notice that
$\tilde{R}$ is not invertible (it is not even square). Using standard properties
of the projection matrix, we know that $P$ has rank $p$, since $\tilde{R}$ has rank $p$.
Also, since $R$ is upper triangular, then the last row of $\tilde{R}$ contains all zeros.
Letting, $e_j$ denote the $j^{\text{th}}$ standard basis vector of $\mathbb{R}^{p+1}$,
this means that
\begin{align}
\mathcal{R}(P) \perp \text{span}(e_{p+1}),
\end{align}
where $\mathcal{R}(P)$ denotes the range (i.e., column space) of $P$.
The only subspace of $\mathbb{R}^{p+1}$ with rank $p$ and satisfying this property
is $\text{span}(e_1, \dots, e_p)$. The conclusion is that $P$ projects onto $\text{span}(e_1, \dots, e_p)$, and thus the annihilator $I - P$ projects onto
the orthogonal complement $\text{span}(e_{p+1})$. We thus conclude,
\begin{align}
\left\lVert \left[I - P\right]r \right\rVert_2^2
&= \lVert \langle r, e\_{p+1} \rangle e\_{p+1} \rVert_2^2 \newline
&= \lVert r\_{p+1} e\_{p+1} \rVert_2^2 \newline
&= r\_{p+1}^2,
\end{align}
where $r_{p+1}$ is the last entry of $r$; i.e., the bottom right entry of $R$.
We finally arrive at the expression for the concentrated log marginal likelihood
\begin{align}
\mathcal{L}(\theta)
&= -\frac{n}{2} \log(2\pi) - \sum_{i=1}^{n} \log\left(L_{ii}\right) -
\frac{1}{2} r_{p+1}^2.
\end{align}

# References
- Surrogates (Gramacy)
- Statistics or geostatistics? Sampling error or nugget effect? (Clark)
- Michael Betencourt's very nice [post](https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html#4_adding_an_informative_prior_for_the_length_scale) on setting
priors on GP hyperparameters.
