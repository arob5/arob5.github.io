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
$\pi(\theta)$, then again when comparing to $T(y_\text{rep})$ to
$T(y_\text{obs})$. We will
discuss the consequences of such "double-dipping" below in more depth.
For the time being, let's intuitively consider the potential concerns.
Since the posterior is influenced by the observed data, then it would
seem that by design $T(y_{\text{obs}})$ ought to be close to
$T(y_{\text{rep}})$. Thus, we might expect posterior predictive checks to be
overly optimistic, potentially failing to diagnose cases of model
misspecification.

On the other hand, there is still reason to think
that such checks may be of some value, especially given a well-chosen
test statistic $T$. For example, suppose our model assumes a simple
Gaussian likelihood $y|\theta \sim \mathcal{N}(\theta, \sigma^2 I)$. In this
case, the posterior essentially balances the model fit term
$\lVert y_{\text{obs}} - \theta \rVert^2$ with the prior $\pi_0(\theta)$.
Therefore, it would be unsurprising that a posterior predictive check with
respect to $T(y) := y$ or $T(y) := yy^\top$ would look quite optimistic.
However, the Gaussian likelihood does not at all take into account higher
moments. It therefore might be helpful to choose $T$ to target higher moments
of the data in order to interrogate the assumed Gaussian data generating
process.  

## Comparing to the Reference
### Simulating Data
### Graphical Checks
### Bayesian p-values
$\bar{p}(y_{\text{obs}}) := \mathbb{P}[T(y_{\text{rep}}) \geq T(y_{\text{obs}})]$.

{% endkatexmm %}

# Frequentist Interpretation: Calibration
{% katexmm %}
It is important to note that, thus far, we have been viewing the event
$\{T(y_{\text{rep}}) \geq T(y_{\text{obs}})\}$ as random only as
a function of $y_{\text{rep}} \sim q$, where the distribution
$q$ is associated with the Bayesian probability model. In particular,
we will consider $q$ to be the posterior predictive distribution
(7) for now. We interpret the data $y_{\text{obs}}$ as
fixed (non-random), which
implies that $\bar{p}(y_{\text{obs}})$ is likewise non-random.
In thinking about how to interpret the quantity
$\bar{p}(y_{\text{obs}})$, we naturally might wonder about
how sensitive its value is to $y_{\text{obs}}$. After all,
in practice many factors contribute the construction of the
dataset, and if things had progressed slightly differently
we might have ended up with different data. The Bayesian
model doesn't really let us tackle this question;
$\bar{p}(y_{\text{obs}})$ lives entirely in the world
of the Bayesian probability model. One approach to this
question is to assume that the data $y_{\text{obs}}$
is generated from some true distribution
$$
y_{\text{obs}} \sim p_{\star}.
$$
If the likelihood $p(y|\theta)$ is well-specified then
$p_{\star}$ takes the form $p_{\star}(y) = p(y|\theta_{\star})$
for some true parameter value $\theta_{\star}$. We will
denote probabilities with respect to
$y_{\text{obs}} \sim p_{\star}$ using the notation
$\mathbb{P}_{\star}[\cdot]$.

We have
now adopted a classical perspective on the data-generating
process, placing a frequentist perspective on top of the
Bayesian model. With this additional assumption, we now
view $\bar{p}(y_{\text{obs}})$ as a random variable,
with the randomness stemming from
$y_{\text{obs}} \sim p_{\star}$. We expect the value
of $\bar{p}(y_{\text{obs}})$ to vary with different
realizations of $y_{\text{obs}}$. For example, even if the
Bayesian model is reasonable, it might be that certain
realizations of $y_{\text{obs}}$ lead to extreme
values of $T(y_{\text{obs}})$ with corresponding large
values of $\bar{p}(y_{\text{obs}})$. Ideally, we would hope
that $\bar{p}(y_{\text{obs}})$ "tracks" with the distribution
$p_{\star}$. Precisely, for any fixed $y$, we would like the
following to hold:
$$
\bar{p}(y) = \mathbb{P}\left[T(y_{\text{rep}}) \geq T(y) \right]
= \mathbb{P}_{\star}\left[T(y_{\text{obs}}) \geq T(y) \right].
$$
The first probability is with respect to $y_{\text{rep}} \sim q$,
while the second is with respect to $y_{\text{obs}} \sim p_{\star}$.
This equality provides a connection between the Bayesian
and frequentist perspectives. It says that the Bayesian
p-value (which has nothing to do with $p_{\star}$) can
alternatively be interpreted as a probability with respect
to the true data-generating process. Note that $y_{\text{rep}}$
also implicitly depends on $p_{\star}$ through the posterior
distribution

{% endkatexmm %}

References to add:
- Posterior predictive checks (David M. Blei, Princeton)
- Bayesian Data Analysis book
- Bayesian posterior predictive checks for complex models (Lynch)
- Split predictive checks
- Holdout predictive checks for Bayesian model criticism (Moran et al)
- Bayesian predictive assessment of model fitness via realized discrepancies (Gelman et al)
- Bayesian checking for topic models (Mimno et al)
- Checking for prior-data conflict (Evans and Moshonov)
- Comment: Posterior predictive assessment for data subsets in hierarchical models via MCMC.
