---
title: Predictive Checks in Bayesian Models
subtitle: Prior and posterior predictive checks.
layout: default
date: 2024-11-21
keywords: UQ
published: true
---

{% katexmm %}
Consider the typical statistical setting whereby one wants to relate a latent
parameter $\theta$ to observed data $y_{\text{obs}}$. The Bayesian approach
models $\theta$ as a random variable, and also views $y_{\text{obs}}$ as a
realization of a data random variable $y$. The data and parameter are then
related through the construction of a joint probability distribution over
$(\theta, y)$. Inference on the parameter $\theta$ proceeds by characterizing
the conditional distribution $\theta|[y = y_{\text{obs}}]$;
i.e., the posterior distribution. We emphasize that the "model" in a  
Bayesian context is the joint distribution we define on $(\theta, y)$. By
conditioning on $y = y_{\text{obs}}$ we are assuming that the observed data
$y_{\text{obs}}$ can reasonably be viewed as a realization of the random
variable $y$ under the assumed model. It is therefore important to check that
the model is well-specified; or, more realistically, that the model is not
too badly misspecified. In this post we will discuss *predictive checks*,
a popular class of methods to diagnose such model inadequacy.
{% endkatexmm %}

# Setup and Background
{% katexmm %}
We start by establishing a bit of notation. Assume the joint probability
distribution on $(\theta,y)$ admits a density $p(\theta,y)$ that can be decomposed as
as
$$
p(\theta,y) = \pi_0(\theta)p(y|\theta), \tag{1}
$$
where $\pi_0(\theta)$ is the *prior* density describing the marginal distribution
of $\theta$. For each fixed value of $\theta$, $p(\cdot|\theta)$ is a density
function over the data space. Viewed as a function of $\theta$ with $y$ fixed, we refer
to the map $\theta \mapsto p(y|\theta)$ as the likelihood. The marginal
distribution of $y$ under model (1) is referred to as the *prior predictive distribution*, and its density is obtained by marginalizing $\theta$ with
respect to its prior:
$$
p(y)
= \int p(\theta,y) d\theta
= \int \pi_0(\theta)p(y|\theta) d\theta. \tag{2}
$$
As noted above, the conditional distribution $\theta|[y = y_{\text{obs}}]$ is
called the *posterior distribution*, and its density is given by Bayes' rule
$$
\pi(\theta)
:= p(\theta|y = y_{\text{obs}}) \propto \pi_0(\theta) p(y_{\text{obs}}|\theta) \tag{3}
$$
We can also introduce a new random variable $\tilde{y}$ representing another
independent realization of the data. We can therefore consider the joint
distribution over $(\theta, y, \tilde{y})$ under the assumed conditional
independence
$$
p(\tilde{y},y|\theta) = p(\tilde{y}|\theta) p(y|\theta). \tag{4}
$$

The conditional distribution $\tilde{y} | [y = y_{\text{obs}}]$ is referred
to as the *posterior predictive* distribution, and its density is found
by marginalizing the parameter with respect to its posterior distribution

\begin{align}
p(\tilde{y}|y = y\_{\text{obs}})
&:= \int p(\theta,\tilde{y} | y = y\_{\text{obs}}) d\theta \tag{5} \newline
&= \int p(\tilde{y} | \theta, y = y\_{\text{obs}}) p(\theta | y = y\_{\text{obs}}) d\theta \newline
&= \int p(\tilde{y} | \theta) p(\theta | y = y\_{\text{obs}}) d\theta \newline
&= \int p(\tilde{y}|\theta) \pi(\theta) d\theta,
\end{align}

where the final two equalities use (4) and (3), respectively.
{% endkatexmm %}

# Predictive Checks
{% katexmm %}
Predictive checks target model inadequacy in certain quantities of interest
specified by the modeler, which is encoded by some function of the data
(i.e., statistic) $T(y_{\text{obs}})$. Thus, instead of tackling the daunting
problem of assessing the entire probabilistic model, we interrogate the model
adequacy in the "direction" given by $T$. The idea of a Bayesian predictive
check is to compare $T(y_{\text{obs}})$ to the distribution of
$T(y_{\text{rep}})$, where $y_{\text{rep}} \sim q$ for some
*reference distribution* $q$. The subscript here stands for "replicated",
since $y_{\text{rep}}$ can be thought of as replicated, or simulated, data;
it is the probabilistic model's representation of the data. If this
representation deviates wildly from the observed quantity $T(y_{\text{obs}})$,
then this provides evidence of model misspecification. This may imply
misspecification in the prior, likelihood, or both. The following section
summarizes popular choices for the reference distribution $q$. We then
make precise what we mean by "comparing" $T(y_{\text{obs}})$ and
$T(y_{\text{rep}})$.

## Reference Distributions
### Prior Predictive Checks
A *prior predictive check* is defined by choosing the reference distribution
$q$ to be the marginal distribution of $y$ under model (1); i.e.,
$$
q(y_{\text{rep}})
:= \int p(\theta,y_{\text{rep}}) d\theta
= \int \pi_0(\theta)p(y_{\text{rep}}|\theta) d\theta, \tag{6}
$$
which is precisely the prior predictive distribution given in (2).

### Posterior Predictive Checks
A *posterior predictive check* instead chooses the reference distribution
$q$ to be the marginal distribution of $\tilde{y}$ under the joint
model $(\theta, y, \tilde{y})$, conditional on $y = y_{\text{obs}}$;
i.e.,
$$
q(y_{\text{rep}})
:= p(y_{\text{rep}}|y = y_{\text{obs}})
= \int p(y_{\text{rep}}|\theta) \pi(\theta) d\theta, \tag{7}
$$
which is precisely the posterior predictive distribution given in (5).
Note a potential concern with this approach: the data $y_{\text{obs}}$
is used twice. It is first used to construct the posterior distribution
$\pi(\theta)$, then again when comparing to $T(y_\text{obs})$. We will
discuss the consequences of such "double-dipping" below in more depth.
For the time being, let's intuitively consider the potential concerns.
Since the posterior is influenced by the observed data, then it would
seem that by design $T(y_{\text{obs}})$ ought to be close to
$T(y_{\text{rep}})$. Thus, we expect posterior predictive checks to be
overly optimistic, potentially failing to diagnose cases of model
misspecification.

On the other hand, there is still reason to think
that such checks may be of some value, especially given a well-chosen
test statistic $T$. For example, suppose our model assumes a simple
Gaussian likelihood $y|\theta \sim \Gaussian(\theta, \sigma^2 I)$. In this
case, the posterior essentially balances the model fit term
$\lVert y_{\text{obs}} - \theta \rVert^2$ with the prior $\pi_0(\theta)$.
Thus, the posterior only incorporates the information in the data
via the quadratic error between $y_{\text{obs}}$ and $\theta$. It would
therefore be unsurprising that a posterior predictive check with respect
to $T(y) := \lVert y_{\text{obs}} - \theta \rVert^2$ would look quite
optimistic. However, we might consider choosing $T$ to capture some other
aspect of the model fit.

TODO: issue is that $T$ as defined above depends on $T(y,\theta)$.

## Comparing to the Reference
### Simulating Data
### Graphical Checks
### Bayesian p-values  
{% endkatexmm %}

# Frequentist Interpretation: Calibration

References to add:
- Posterior predictive checks (David M. Blei, Princeton)
- Bayesian Data Analysis book
- Bayesian posterior predictive checks for complex models (Lynch)
- Split predictive checks
- Holdout predictive checks for Bayesian model criticism (Moran et al)
