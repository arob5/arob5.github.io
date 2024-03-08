---
title: An Introduction to Bayesian Filtering and Smoothing
subtitle: I provide an overview of the general framework for statistical filtering, viewed from the Bayesian perspective.
layout: default
date: 2024-01-29
keywords: Bayes, Filtering, Data-Assim
published: true
---

In this post I provide an overview of the filtering problem for discrete stochastic
dynamical systems, which is nicely cast as a problem of Bayesian inference.
In short, the problem we are trying to solve is as follows: suppose we are trying
to predict the state of a dynamical process which evolves over time. For example,
the process could be the global climate system and the state could be the
temperature at some fixed locations. Suppose we have a model for the system of
interest, which is stochastic in nature; the randomness might be inherant to the
system under study or might come from uncertainties stemming from our lack of
understanding of the dynamics. We could thus step our model forward in time to
try to predict the true state of the system. However, suppose we have a second
source of information: measurements of the process. Just like the dynamics, the
observations are subject to noise, which in this case typically represents some
sort of measurement error. The filtering problem seeks to combine these two
imperfect sources of information to produce an estimate of the true state, as
well as an estimate of the uncertainty. This will be made precise below.

## The Probabilistic State Space Model
{% katexmm %}
The stochastic dynamics and observational process are defined below, formulated
as a Markov state space model (i.e., a hidden Markov model),
\begin{align}
v_{k+1} &\sim G(v_k, \cdot) \newline
y_{k+1} &\sim H(v_{k+1}, \cdot) \newline
v_0 &\sim \mu_0
\end{align}
where $G$ and $H$ are probability kernels defining the stochastic dynamics and
observation process, respectively, and $v_0$ is the initial condition distributed
according to distribution $\mu_0$. We also require the crucial independence
assumptions:
1. For all $j,k$, the distributions $H(v_{k}, \cdot)$ and
$G(v_j, \cdot)$ are independent.
2. For all $j \neq k$, the distributions $H(v_{k}, \cdot)$ and $H(v_{j}, \cdot)$
are independent. Similarly, $G(v_{k}, \cdot)$ and $G(v_{j}, \cdot)$ are
independent.
3. The initial condition $v_0$ is independent of all of these distributions
as well.

In words, the first assumption is requiring that the dynamical noise is
independent of the observation noise. The second assumption assumes a classical
iid noise model for the observation process, as well as a time-homogenous Markov
model for the dynamics. This state space model is quite general; it is
typical to consider the restricted setting of an additive noise model, in which  
the dynamics $G(v_k, \cdot)$ are given by some deterministic
update $g(v_k)$ plus some noise $\eta_{k+1}$, and similarly for
the observation process. These assumptions
yield
\begin{align}
v_{k+1} &= g(v_k) + \eta_{k+1}  \newline
y_{k+1} &= h(v_{k+1}) + \epsilon_{k+1}, \newline
v_0 &\sim \mu_0
\end{align}
{% endkatexmm %}
where we emphasize that in general the non-random functions $g$ and $h$ may be
nonlinear and the random variables $\{\eta_k\}$, $\{\epsilon_k\}$ may be
non-Gaussian. The independence assumptions are easier to state in this special
case, simply requiring that $\{\eta_k\}$, $\{\epsilon_k\}$, and $v_0$ are all
pairwise independent. While this additive noise model is quite common, all
of the calculations later in this post go through under the more general setting.

To wrap up establishing notation, we suppose the states $v_k$
and observations $y_k$ are $d$-dimensional and $n$-dimensional vectors,
respectively. Also, let $Y_k := \\{y_1, \dots, y_k\\}$ collect all observations
up through time $k$ and similarly for the states $V_k := \\{v_1, \dots, v_k\\}$.

### A Pure Probabilistic Modeling Perspective
Defining the probability kernels $G$ and $H$ is a convenient way to encode
the probabalistic structure of the state space model. We can alternatively take
a very generic probabalistic perspective and simply view the model as defining
a joint distribution over all random quantities; supposing that we're
considering a fixed time window $0 \leq k \leq K$, the joint distribution is
\begin{align}
p(V_K, Y_K) = p(v_1, \dots, v_K, y_1, \dots, y_K).
\end{align}
In general, we could define this distribution however we want and make it
quite complicated. The Markovian and independence assumptions vastly simplify
things, making the distribution much easier to work with. These assumptions
are as follows:
1. The Markov property for state transitions.
\begin{align}
p(v_k|V_{k-1}, Y_{k-1}) &= p(v_k|v_{k-1}) \newline
p(v_{k-1}|V_{k:K}, Y_{k:K}) &= p(v_{k-1}|v_k)
\end{align}
2. Conditional independence for observations.
\begin{align}
p(y_k|V_k, Y_{k-1}) &= p(y_k|v_k)
\end{align}

View this as simply defining a joint distribution on all quantities, subject
to some assumptions that simplify the joint distributions (see Saarka pg 52).

## The Filtering and Smoothing Problems
We are now ready to define the two primary problems of interest in this s
setting--filtering and smoothing--plus a bonus which we will encounter as
an intermediate task in addressing the first two.
1. The **filtering problem** is to characterize the conditional
distribution $\mu_k := \mathcal{L}(v_k|Y_k)$ ($\mathcal{L}$ for *law*);
in Bayesian parlance, this is the posterior distribution
of the current state $v_k$ given all of the data observed up through time $k$.
We call $\mu_k$ the **filtering distribution** at time $k$, which is assumed
to have a Lebesgue density $\pi_k$.
2. The **smoothing problem** is to characterize the conditional distribution
$\mathcal{L}(v_k|Y_K)$ with $K > k$; that is, future observations are used to
inform inference regarding past states. The distribution $\mathcal{L}(v_k|Y_K)$
is referred to as the **smoothing distribution**.
3. The **prediction problem** is to characterize the distribution
$\mathcal{L}(v_{k+l}|Y_k)$; that is, to peform inference about a future state
using observations which are not fully up-to-date. We call
$\mathcal{L}(v_{k+l}|Y_k)$ the *prediction*, or **forecast distribution**.

These definitions convey the sequential nature of these problems; the goal will
not only be limited to performing inference for a fixed $k$, but developing
algorithms that are easily updated when new data arrive at future times. We
can also view this from a more traditional static inference perspective by
considering the distribution $\mathcal{L}(V_K|Y_K)$. This joint distribution
encodes both the filtering and smoothing distributions. Indeed, its marginal
distribution in the last slot recovers the filtering distribution at
time $K$. Moreover, for any $k < K$, the marginal consisting of the first
$k$ slots is a smoothing distribution. Note that some authors choose to define
the filtering and smoothing distributions slightly differently. For example,
Sans-Alonso, Stuart, and Taeb actually call the joint distribution
$\mathcal{L}(V_K|Y_K)$ the smoothing distribution, while Saarka reserves this
term for the case where the time index of the state under consideration is
strictly less than that of the time index of the available data. The former
approach serves to emphasize the key conceptual difference between
sequentially updating $\mathcal{L}(v_k|Y_k)$ and performing one-shot inference
for $\mathcal{L}(V_K|Y_K)$.

Finally, a few last notational notes. In most of the subsequent exposition,
I will assume that all of the distributions
in question admit Lebesgue densities, and abuse notation by writing $p(\cdot)$
for most densities, with the special notation $\pi_k(\cdot)$ reserved for the
density for the filtering distribution $\mu_k$.

## The Bayesian Perspective
{% katexmm %}
Let's consider a fixed time horizon $1 \leq k \leq K$ here, but note that our
primary interest will ultimately center on how to update the current filtering
distribution at later times: $K+1$, $K+2$, etc.
A Bayesian analysis assumes all of the quantities under consideration are
governed by probability distributions. In this case, the quantities are the
states $V_K$ and the observations $Y_K$. In this context, the states $V_K$
are the unobserved quantities of interest, the role typically assumed by the
*parameters* in a Bayesian statistical model. The stochastic dynamics model
provides the *prior distribution* $p(V_K)$ over these unknown quantities, while
the observation process supplies the *likelihood* which connects the
observed data $Y_K$ to the unobserved states $V_K$.
The state space model defined above provides the information needed to define
the joint distribution over all the variables
$\{V_K, Y_K\} = \{v_1, \dots, v_K, y_1, \dots, y_K\}$. This joint
distribution assumes a very simple form due to the Markovian assumption
in the dynamics and the independence assumptions stated above,
\begin{align}
p(V_K, Y_K)
&= p(Y_K|V_K)p(V_K) \newline
&= \left[\prod_{k=1}^{K} p(y_k|v_k)\right] \cdot
\left[\prod_{k=1}^{K} p(v_k|v_{k-1}) \right] \cdot p(v_0) \newline
&= p(v_0) \prod_{k=1}^{K} p(v_k|v_{k-1}) p(y_k|v_k).
\end{align}
The first product in the second line follows from the second independence
assumption, while the second uses the Markovian structure in the dynamics.
Note that the conditional distributions $p(V_k|Y_k)$ and $p(v_k|Y_k)$ (which
are the primary objects of interest) are both proportional to $p(V_K, Y_K)$.
We will use this fact below when considering how to sample from these
distributions.
{% endkatexmm %}

## Can we just do MCMC?
{% katexmm %}
Since we're keeping things quite general in this post, non-linearities and
non-Gaussianity in the state space equations will unsurprisingly lead to
filtering distributions which cannot be computed in closed-form. In this section
we consider how we might go about sampling from $\mu_k$ using
Markov chain Monte Carlo (MCMC) and in doing so highlight some limitations of
this approach. Let's consider designing an MCMC sampler targeting
$p(V_K|Y_K)$, recalling that this would yield both the filtering distribution
at time $K$, in addition to all of the smoothing distributions up through
time $K$. The structure of this problem lends itself to an MCMC scheme that
updates each conditional distribution one at a time; that is, either a Gibbs
or Metropolis-within-Gibbs algorithm depending on whether or not it is possible
to directly sample the conditionals $p(v_k|V_{-k},Y_k)$ (I use the notation
$V_{-k} := \{v_0, \dots, v_{k-1}, v_{k+1}, \dots, v_k\}$). Recalling the form
of the joint distribution, these conditionals are given by
\begin{align}
p(v_k|V_{-k},Y_k) &\propto p(y_k|v_k)p(v_k|v_{k-1})p(v_{k+1}|v_k).
\end{align}
Aside from the issue that the MCMC sampler might fail to mix well when the
state dimension $d$ or number of time steps $K$ is large, this approach as
one other glaring issue. If we acquire a single new data point $y_{K+1}$ then
the sampler must be run from scratch to sample the new posterior
$p(V_{K+1}|Y_{K+1})$. Since the dimensionality of the state vector grows as
$K$ increases, each subsequent MCMC run will become more challenging and
ultimately computationally infeasible. This
motivates approaches which are amenable to fast updates when new data comes
online. The form of the filtering distribution makes it particularly
suitable for such recursive (i.e. sequential) inference schemes.  
{% endkatexmm %}

## Sequential Inference for the Filtering Problem
{% katexmm %}
Motivated by the previous section, we seek solutions which allow an estimate
of the filtering distribution to be updated sequentially as time progresses.
We begin by considering what exact inference would entail in this setting; i.e.,
the computations required for updating the filtering density in closed-form.
Think about what is required to map $\mu_k$ to $\mu_{k+1}$; first, the dynamics
model generates a prediction for $v_{k+1}$ and then we receive a new observation
$y_{k+1}$. These two sources of information are combined to produce the new
filtering distribution $\mu_{k+1}$. Thus, it is natural to break the update
from $\mu_k$ to $\mu_{k+1}$ into two steps: *prediction* and *analysis*.
Assuming we already know $\mu_k$, we now characterize these two steps, the
composition of which results in the map $\mu_k \mapsto \mu_{k+1}$.

### Forecast
The first step concerns the map $\mu_k \mapsto \mathcal{L}(v_{k+1}|Y_k)$,
which represents the state of knowledge about $v_{k+1}$ given the all
model predictions through time $k+1$, but only considering the observations
up through time $k$. This is the *forecast distribution* defined above,
with time lag $l=1$. We will denote this special case of the forecast
distribution by $\hat{\mu}_{k+1}$, with
density $\hat{\pi}_{k+1}(\cdot) = p(\cdot|Y_k)$. The forecast distribution may
be derived by first considering the joint distribution
\begin{align}
p(v_k, v_{k+1}|Y_k)
&= p(v_{k+1}|v_k, Y_k) p(v_k|Y_k) = p(v_{k+1}|v_k)p(v_k|Y_k)
= p(v_{k+1}|v_k)\pi_k(v_k),
\end{align}
where the final equality uses the Markov assumption. The forecast distribution
then follows by marginalizing $v_k$,
\begin{align}
\hat{\pi}\_{k+1}(v_{k+1})
&= p(v_{k+1}|Y_k) = \int p(v_k, v_{k+1}|Y_k) dv_k =
\int p(v_{k+1}|v_k)\pi_k(v_k) dv_k.
\end{align}
This is a version of the *Chapman-Kolmogorov equation* applied to the Markovian
dynamics model, but simply conditional on $Y_k$. We observe that the current
filtering density $\pi_k(v_k)$ is transformed into the forecast density
$\hat{\pi}\_{k+1}(v_{k+1})$ by means of the above integral, which defines the
map $\mu_k \mapsto \hat{\mu}_{k+1}$. Note that this is alternatively referred
to as the *predict* step, and the associated distribution the
*predictive distribution*.

### Analysis
We have decomposed the map $\mu_k \mapsto \mu_{k+1}$ as the composition
$\mu_k \mapsto \hat{\mu}_{k+1} \mapsto \mu_{k+1}$. The *analysis* step is
defined to be the second map in this composition. It involves combining the
forecast distribution with the new observation $y_{k+1}$ and thus simply
requires an application of Bayes' theorem
\begin{align}
\pi_{k+1}(v_{k+1})
= p(v_{k+1}|Y_{k+1}) &= p(v_{k+1}|Y_{k}, y_{k+1}) \newline
&= \frac{p(y_{k+1}|v_{k+1},Y_k)p(v_{k+1}|Y_k)}{p(y_{k+1}|Y_k)} \newline
&= \frac{p(y_{k+1}|v_{k+1})\hat{\pi}\_{k+1}(v_{k+1})}{p(y_{k+1}|Y_k)},
\end{align}
where the final line uses the independence assumptions and the definition of the
forecast distribution.
Note that Bayes' theorem is being applied with respect to the data $y_{k+1}$
and everything is conditional on $Y_k$. Note that in the context of the analysis
step, the forecast distribution assumes the role of prior distribution on the
state $v_{k+1}$. Alternative names for this step include the *update* or
*correction* step.
{% endkatexmm %}

## Viewing the Filtering Update as an Operator on Probability Distributions   
{% katexmm %}
In the previous section we viewed the map
$\mu_k \mapsto \hat{\mu}_{k+1} \mapsto \mu_{k+1}$ through the manner in which
it transforms probability density functions. Here we take a step back and
appreciate the structure of these maps from a more generic perspective.

### Prediction
Abstractly, we can view the prediction step as the result of feeding the current
distribution $\mu_{k}$ through the one-step Markov kernel $G$,
\begin{align}
\hat{\mu}_{k+1}(\cdot) &= (\mu_k G)(\cdot) := \int G(v_k, \cdot) \mu_k(dv_k).
\end{align}
Viewing the integral as an operation on $\mu_k$, observe that this map is linear,
even though the dynamic model may apply a non-linear map to the states.

### Analysis
The operator mapping $\hat{\mu}_{k+1}$ to $\mu_{k+1}$ arises from Bayes' rule,
and thus can be thought of as the composition of a likelihood operator followed
by a normalization operator. Assuming that the observation process admits
probability densities, Bayes' rule can be abstractly stated as
\begin{align}
\frac{d\mu_{k+1}}{d\hat{\mu}\_{k+1}}(v_{k+1})
&= \frac{p(y_{k+1}|v_{k+1})}{\int p(y_{k+1}|v_{k+1}) \hat{\mu}_{k+1}(dv\_{k+1})}
\end{align}
which expresses the idea that the posterior is a re-weighted version of the
prior, with the weights provided by the likelihood $p(y_{k+1}|v_{k+1})$; the
denominator is simply a normalization constant. By definition of the
Radon-Nikodym derivative, this yields

\begin{align}
\mu_{k+1}(A)
&= \frac{\int_A p(y_{k+1}|v_{k+1}) \hat{\mu}\_{k+1}(dv_{k+1})}{\int p(y_{k+1}|v_{k+1}) \hat{\mu}\_{k+1}(dv_{k+1})}
\end{align}

where $A$ is a measurable set. Viewing this as an operation on $\hat{\mu}_{k+1}$,
observe that the numerator represents a linear transformation of the probability
distribution. However, the operation defined by the denominator is non-linear,
and hence the map $\hat{\mu}_{k+1}$ to $\mu_{k+1}$ is non-linear. Denoting this
map by $\mathcal{A}$ (for *analysis*) we can write
$\mu_{k+1} = \mathcal{A}(\hat{\mu}_{k+1})$.

In this section we have taken an alternative view of the filtering problem. Instead
of considering maps acting on states, we view the state space model as defining
maps which act on probability measures. We showed that updates the filtering
distribution from $\mu_k$ to $\mu_{k+1}$ can be decomposed as
$\mu_k \mapsto \hat{\mu}_{k+1} \mapsto \mu_{k+1}$, where the first
*prediction* operator is linear and Markovian, while the second *analysis*
operator is a non-linear likelihood map. Thus, the complete map between the
subsequent filtering distributions is given by
$$
\mu_{k+1} = \mathcal{A}(\mu_k G).
$$
Many filtering algorithms can be viewed as providing approximations for these
two component operators.
{% endkatexmm %}

## Addressing the Smoothing Problem  
{% katexmm %}
We now turn from filtering to smoothing, where data up through the current
time step is used to inform estimates of past states; that is, the target
distribution here is $p(v_k|Y_K)$ with $K > k$. Given this backward
looking nature, this problem is also referred to as **reanalysis**. We will
see that the smoothing problem is strictly more difficult than the filtering
problem; in fact, smoothing algorithm typically first require solving the
filtering problem as an initial step. In the filtering section, we established
the recursion $p(v_k|Y_k) \to p(v_{k+1}|Y_{k+1})$, which naturally breaks into
the forecast and analysis steps. In seeking a similar result
for the smoothing problem we will instead find the relationship
\begin{align}
p(v_k|Y_K)
&= \pi_k(v_k) \int \frac{p(v_{k+1}|v_k)}{p(v_{k+1}|Y_k)} p(v_{k+1}|Y_K) dv_{k+1}
\end{align}
which establishes a *backward* recursion $p(v_{k+1}|Y_K) \to p(v_k|Y_K)$.
Before deriving this, let's spend a minute thinking about why it makes sense.
The smoothing distribution at time $k$ depends on a product of two terms;
the first is the filtering distribution at time $k$, which is capturing the
information up through the current time. The information coming from the future
is encoded in the second term, which can be viewed as a weigted average of the
smoothing distribution at time $k+1$, $p(v_{k+1}|Y_K)$. The weights in this
average are $\frac{p(v_{k+1}|v_k)}{p(v_{k+1}|Y_k)}$, the ratio of the
distribution predicted by the dynamics alone over the prediction distribution.
As an example, consider the extreme case that this ratio is always equal to
$1$, which causes the integral to evaluate to $1$ and hence $p(v_k|Y_K)$ just
equals the filtering distribution $p(v_k|Y_k)$. Does this make sense? Well, if
the weights in the integral are always one then this is saying that,
conditional on $Y_k$, knowing $v_k$ doesn't help predict $v_{k+1}$. So given
$Y_k$, the states $v_k$ and $v_{k+1}$ are essentially unrelated so the link
between $v_k$ and any useful information from the future has been severed. All
of the useful information is already contained in the filtering distribution.

Before deriving the recursion, I emphasize again that (1) the formula defines a
*backwards* recursion; and (2) the formula depends on the filtering distribution.
The consequences of this will become clear in future posts where we see that
many smoothing algorithms adopt a two-stage "forward filter, backward sample"
structure. Without further ado, the backward recursion is derived below.
\begin{align}
p(v_k|Y_K)
&= \int p(v_k, v_{k+1}|Y_K) dv_{k+1} \newline
&= \int p(v_k|v_{k+1}, Y_k)p(v_{k+1}|Y_K) dv_{k+1} \newline
&= \int p(v_k|v_{k+1}, Y_k)p(v_{k+1}|Y_K) dv_{k+1} \newline
&= \int \frac{p(v_{k+1}|v_k, Y_k)p(v_k|Y_k)}{p(v_{k+1}|Y_k)} p(v_{k+1}|Y_K) dv_{k+1} \newline
&= \int \frac{p(v_{k+1}|v_k)\pi_k(v_k)}{p(v_{k+1}|Y_k)} p(v_{k+1}|Y_K) dv_{k+1} \newline
&= \pi_k(v_k) \int \frac{p(v_{k+1}|v_k)}{p(v_{k+1}|Y_k)} p(v_{k+1}|Y_K) dv_{k+1}
\end{align}
The third and fifth lines follow from the conditional independence assumptions,
while the fourth applies Bayes' rule.

{% endkatexmm %}


## TODOs
* Add independence assumptions in state space model.
* Add graphical plot of state space model.
* Why MCMC is inefficient in this setting.
