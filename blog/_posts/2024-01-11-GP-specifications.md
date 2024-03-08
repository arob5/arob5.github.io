---
title: Comparing Approaches for Specifying and Estimating Gaussian Process Parameters
subtitle: I review and derive various formulas that come in handy when sequentially adding data to a Gaussian process model.
layout: default
date: 2024-01-11
keywords: Gaussian-Process
published: false
---


## Setup and notation
{% katexmm %}

### The Model
We consider the common observation model in which observations of a GP are
directly observed, after potentially being polluted by some additive iid
Gaussian noise,
\begin{align}
y(\mathbf{x}) &= f(\mathbf{x}) + \epsilon(\mathbf{x}) \newline \tag{1}
f(\cdot) &\sim \mathcal{GP}(\mu(\cdot; \theta\_{\mu}), k(\cdot, \cdot; \theta_k)) \newline
\epsilon(\mathbf{x}) &\overset{\text{iid}}{\sim} \mathcal{N}(0, \eta^2)
\end{align}

with $f$ and $\epsilon$ assumed independent. Here, the inputs $\mathbf{x}$ belong
to some space $\mathcal{X} \subseteq \mathbb{R}^D$ and the response is
univariate; i.e., $y(\mathbf{x}) \in \mathbb{R}$. The prior mean
$\mu: \mathcal{X} \to \mathbb{R}$ and covariance function (i.e. kernel)
$k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ are parameterized by
some sets of parameters $\theta_{\mu}$ and $\theta_k$, respectively. Often
we suppress writing the dependence on these *hyperparameters*, but since the
point of this post is to explore how these parameters are specified and estimated,
we draw explicit attention to them above. When convenient, we will collectively
denote these parameters as $\theta := \{\theta_{\mu}, \theta_k, \eta^2 \}$,
where we have also tossed in the observation variance $\eta^2$, which is also
typically unknown and hence must also be estimated. This variance parameter is
sometimes also called the *nugget*.

Before proceeding, I note that the regression
model considered here is only the tip of the iceberg when it comes to GPs,
contrary to the way they are sometimes presented in machine learning texts.
GPs with non-Gaussian likelihoods and non-linear mappings of GPs are routinely
considered in such fields as Bayesian inverse problems. This post does not
touch on these more complicated settings (the nice closed-form posterior in
the next section doesn't even apply anymore).

### GP Closed-Form Posterior, Assuming Fixed Hyperparameters
We now consider the typical regression setting, in which a dataset consisting
of $N$ input-output pairs $\{\mathbf{x}_n, y_n\}_{n=1}^{N}$ has been observed.
Let $\mathbf{X}$ denote the $N \times D$ design matrix with $n^{\text{th}}$ row
equal to $\mathbf{x}_n$. Similarly, let $\mathbf{y}$ be the $N \times 1$ vector
with $n^{\text{th}}$ entry set to $y_n$. Suppose we are interested in predicting
the response at some unsampled locations
$\mathbf{\tilde{x}}_1, \dots, \mathbf{\tilde{x}}_M$, stacked into the rows of
the $M \times D$ matrix $\mathbf{\tilde{X}}$. Denote the response at these
locations by $\mathbf{\tilde{y}} := y(\mathbf{\tilde{X}}) \in \mathbb{R}^M$.
The GP predictive (i.e. conditional or posterior) distribution
$\mathbf{\tilde{y}}|\mathbf{X}, \mathbf{y}, \mathbf{\tilde{X}}$ is found by
considering the joint distribution across the observed and unobserved locations
and then applying the well-known Gaussian conditional formulas.
\begin{align}
\begin{bmatrix} \mathbf{y} \\\ \mathbf{\tilde{y}} \end{bmatrix} \bigg|
\mathbf{X}, \mathbf{\tilde{X}}
\sim \mathcal{N}\left(
\begin{bmatrix} \mu(\mathbf{X}) \\\ \mu(\mathbf{\tilde{X}})\end{bmatrix},
\begin{bmatrix} \mathbf{K} & k(\mathbf{X}, \mathbf{\tilde{X}})
\\\ k(\mathbf{\tilde{X}}, \mathbf{X}) & k(\mathbf{\tilde{X}}) \end{bmatrix}
\right)
\end{align}
where we have defined $\mathbf{K} := k(\mathbf{X})$. We see that the GP
predictive distribution is indeed simply the conditional distribution of this
multivariate Gaussian. Applying the formula for a Gaussian conditional yields
\begin{align}
\mathbf{\tilde{y}}|\mathbf{X}, \mathbf{y}, \mathbf{\tilde{X}}
\sim \mathcal{N}\left(\hat{\mu}(\mathbf{\tilde{X}}), \hat{k}(\mathbf{\tilde{X}}) \right)
\end{align}
{% endkatexmm %}

## Generic Hyperparameter Estimation Approaches
### Maximum Likelihood
### Bayesian Modeling
