---
title: Probabilistic Forecasting and Calibration
subtitle:
layout: default
date: 2024-08-15
keywords: probability, statistics
published: true
---

In this post, I provide an overview of a theory of probabilistic forecasting.
While many of the foundational ideas in this area date back to the mid-twentieth
century, the general formulation of this theory has undergone most of its
development in the past couple decades and is still an active area of research.  
The underlying idea here is that, in order to quantify uncertainties,
predictions of unknown quantities ought to take the form of probability
distributions. This immediately leads to many challenging questions; perhaps
the most obvious being *how do you evaluate the quality of a probabilistic forecast?*
If I predict that there is a 90% chance it will rain tomorrow but it doesn't rain,
did I do a bad job? Or was my probabilistic forecast reasonable but the
realized event just happened to fall in that 10% "no rain" tail? As we will see,
trying to answer questions like these by intuition alone can lead to sub-optimal
strategies and counterintuitive results. A mathematical theory of forecasting
provides the foundation to rigorously study such questions.

We begin this post by defining the theoretical framework that will be used to
study questions related to probabilistic forecasting. We will then discuss the
notion of what it means for a forecast to be "calibrated", before detailing
methods of forecast evaluation using so-called *scoring rules*. Those in the
machine learning community are probably most familiar with the idea of
probabilistic calibration through its application to problems of supervised
[classification](https://scikit-learn.org/stable/modules/calibration.html)
(calibration curves, relaibility diagrams, etc.). We will discuss how these
ideas fall out as a special case of the more general theory presented here.

A long list of references is provided at the end of this post.
[Tillman Gneiting](https://www.h-its.org/people/prof-dr-tilmann-gneiting/)
has been instrumental in developing the modern theory of probabilistic
forecasting, and his papers cited below are one of the major sources for
the information summarized here. I also recommend
Ryan Tibshirani's [lecture notes](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/)
on scoring and calibration for a nice accessible overview.

## Theoretical Framework
### The Prediction Space

{% katexmm %}
In general, probabilistic forecasts are issued in order to predict some
unknown quantity. We will model this target quantity as a random variable
$Y$, taking values in some space $\mathcal{Y}$, and distributed according to
some probability measure $\nu$. Throughout we will write
$\mathcal{L}(Y) = \nu$ to mean that the random variable $Y$ has distribution
(i.e., law) $\nu$. In typical settings, the distribution $\nu$ is unknown to
the forecaster, who only observes a realization $y \in \mathcal{Y}$ of the
random variable $Y$. Prior to this observation, we assume the forecaster issues a
probabilistic forecast $\mu$ in seeking to predict $Y$, where $\mu$ is some
probability measure encoding the forecaster's prediction. In providing a
theoretical model for real-world forecasting problems, notice that the two
main quantities of interest here are the forecast $\mu$ and the observation
$Y$: the forecaster issues a forecast and then observes the realization.

While modeling $Y$ as a random quantity is a straightforward choice, we will
also make the, perhaps less obvious, decision to treat the forecast $\mu$
itself as a random element. That is, we model the forecast as a
*random (probability) measure*. Think about how forecasts might be obtained
in practice, via a combination of data, models, assumptions, expert options, etc.
It therefore seems reasonable to model the process by which all of this
information is transformed into a forecast as random. In a parallel universe
where things unfolded slightly differently, a different forecast might be
produced. Murphy and Winkler (1987) take this insight a step further, proposing
that probabilistic forecasts ought to be studied on the basis of the *joint*
distribution over $(\mu, Y)$. It is reasonable to think that the processes by
which the observation $y$ is generated might also influence the generation of
the forecast $\mu$. For example, some meteorological processes that lead to
a temperature observation $Y$ might contribute to other sources of observed
data on which the forecast $\mu$ is based. The main takeaway here is
that a theory of probabilistic forecasting should be based around the joint
distribution between the forecast and the observation. We will call this joint
probability space the **prediction space**. The below definition
summarizes this idea, and fills in some of the measure-theoretical details.

<blockquote>
  <p><strong>Definition (Prediction Space).</strong>
  Let $(\mathcal{Y}, \mathcal{B})$ be some measurable space of possible outcomes,
  and $\mathcal{M}$ a space of probability measures over $(\mathcal{Y}, \mathcal{B})$.   
  A prediction space is a probability space $(\Omega, \mathcal{A}, \mathbb{Q})$,
  along with a sub-$\sigma$-algebra $\mathcal{A}_1 \subseteq \mathcal{A}$,
  where $\Omega = \mathcal{M} \times \mathcal{Y}$.
  </p>
</blockquote>

The sub-$\sigma$-algebra $\mathcal{A}_1$ can be interpreted as the knowledge,
or information basis, available to the forecaster. The condition
$\mathcal{A}_1 \subseteq \mathcal{A}$ thus encodes the natural assumption that
"nature" has a larger information basis than the forecaster. While the above
definition focuses on defining a suitable measure space, it will typically be
convenient to work on the level of random variables (elements).
Note that I reserve the phrase *random variable* for random quantities assuming
a finite number of values. Since, in general, $\mu$ lives in an infinite-dimensional
space of measures I use the term *random element* or *random quantity*.

<blockquote>
  <p><strong>Prediction Spaces and Random Elements.</strong>
  Given a prediction space $(\Omega, \mathcal{A}, \mathbb{Q})$, as above, we
  will write $(\mu, Y)$ to denote a random element taking values in the
  measurable space $(\Omega, \mathcal{A})$, and distributed according to
  $\mathbb{Q}$. Formally, this random element is defined with respect to
  some implicit underlying probability space
  $(\Omega^\prime, \mathcal{B}^\prime, \mathbb{P}^\prime)$ such that
  $(\mu, Y)(\cdot): \Omega^\prime \to \Omega$. The measure $\mathbb{Q}$ is
  thus the distribution of the random element $(\mu, Y)$ given by the
  push-forward $\mathbb{Q} = \mathbb{P}^\prime \circ (\mu, Y)^{-1}$.
  </p>
</blockquote>

The above definition of the random element interprets random outcomes
$(\mu, Y)$ from the prediction space as a random element that inherits its
randomness from some underlying probability space. To be clear on notation,
$\mathcal{L}(\mu, Y) = \mathbb{Q}$ and $(\mu, Y) \sim \mathbb{Q}$ are two
equivalent ways to express that the random element $(\mu, Y)$ is distributed
according to $\mathbb{Q}$. As mentioned above, we interpret $\mathcal{A}_1$
as the knowledge available to the forecaster; hence, we should require that
the random measure $\mu$ be measurable with respect to $\mathcal{A}_1$. It need
not be measurable with respect to nature's knowledge $\mathcal{A}$, as the
forecasters knowledge base may be more limited.

<blockquote>
  <p><strong>Assumption.</strong>
  The random element $\mu$ is measurable with respect to the sub-$\sigma$-algebra
  $\mathcal{A}_1$; that is,
  $$
  \mu^{-1}(M) \in \mathcal{B}^\prime, \text{ for all } M \in \mathcal{A}_1.
  $$
  Note the slight abuse of notation in writing $\mathcal{B}^\prime$ and
  $\mathcal{A}_1$ even though we are restricting to the measure portion of the
  joint space over $(\mu, Y)$.
  </p>
</blockquote>

The notation here can get a bit thorny. We have formalized $\mu$ as a random
element: a measurable map from $\Omega^\prime$ to $\mathcal{M}$. This means
that the image $\mu(\omega^\prime) \in \mathcal{M}$ is a probability measure for
each $\omega^\prime \in \Omega^\prime$; i.e.,
$$
\mu(\omega^\prime)(\cdot): \mathcal{B} \to [0, 1].
$$
The two arguments on the lefthand side are notationally a bit of a pain, so we
will proceed by treating $\mu$ as a measure representing the image
of the above measurable map, meaning that the $\omega^\prime$ will be suppressed.
We will thus write $\mu(B)$ for $B \in \mathcal{B}$ to mean the random measure
$B$ evaluated at the set $B \subset \mathcal{Y}$.

### Special Case: Real-Valued Quantities
The majority of forecasting literature has focused on the setting
$\mathcal{Y} = \mathbb{R}$; i.e., the quantity of interest $Y$ assumes
scalar values in the real line. Generalization to multivariate settings is an
active research area and beyond the scope of this post. We now restrict the
previous definition to this special case.

<blockquote>
  <p><strong>Definition (Prediction Space in the Scalar Setting).</strong>
  Let $\mathcal{M}$ denote a space of probability measures over
  $(\mathbb{R}, \mathcal{B})$, where $\mathbb{B}$ is the Borel $\sigma$-algebra
  on the real line. In this setting, a prediction space is a probability space
  $(\Omega, \mathcal{A}, \mathbb{Q})$, where
  $\Omega = \mathcal{M} \times \mathbb{R}$, along with a sub-$\sigma$-algebra
  $\mathcal{A}_1$. As before, we will consider $\mathbb{Q}$ as the law
  of random element $(\mu, Y) \in \Omega$. We will also identify the measures
  $\mu \in \mathcal{M}$ with their cumulative distribution functions (CDFs)
  $F(y) := \mu([-\infty, y])$ and slightly abuse notation by writing
  $F \in \mathcal{M}$.
  </p>
</blockquote>

<blockquote>
  <p><strong>Measurability Condition in the Scalar Setting.</strong>
  TODO
  </p>
</blockquote>


### Generalization: Multiple Forecasts  


{% endkatexmm %}







## References
1. Predicting good probabilities with supervised learning.
2. https://scikit-learn.org/stable/modules/calibration.html
3. A General Framework for Forecast Verification (Murphy and Winkler, 1987)
4. Coherant Combination of Experts' Opinions (Dawid, 1995)
5. CS269I: Incentives in Computer Science Lecture #17: Scoring Rules and Peer Prediction (Incentivizing Honest Forecasts and Feedback)
