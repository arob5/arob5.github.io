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
on scoring and calibration for a nice overview.

## Theoretical Framework
{% katexmm %}
In general, probabilistic forecasts are issued in order to predict some
unknown quantity. We will model this target quantity as a random variable
$Y$, taking values in some space $\mathcal{Y}$, and distributed according to
some probability measure $\nu$. Throughout we will write
$\mathcal{L}(Y) = \nu$ to mean that the random variable $Y$ has distribution
(i.e., law) $\nu$. In typical settings, the distribution $\nu$ is unknown to
the forecaster, who only observes a realization $y \in \mathcal{Y}$ of the
random variable $Y$. Prior to this, we assume the forecaster issues a
probabilistic forecast $\mu$ in seeking to predict $Y$, where $\mu$ is some
probability measure encoding the forecaster's prediction. In providing a
theoretical model for real-world forecasting problems, notice that the two
main quantities of interest here are the forecast $\mu$ and the observation
$Y$: the forecaster issues a forecast and then observes the observations.

While modeling $Y$ as a random quantity is a straightforward choice, we will
also make the, perhaps less obvious, decision to treat the forecast $\mu$
itself as a random element. That is, we model the forecast as a
*random (probability) measure*. Think about how forecasts might be obtained
in practice, via a combination of data, models, assumptions, expert options, etc.
It therefore seems reasonable to model the process by which all of this
information is transformed into a forecast as random. In a parallel universe
where things unfolded slightly differently, a different forecast might be
produced. Murphy and Winkler (1987) take this insight a step further, in proposing
that probabilistic forecasts ought to be studied on the basis of the *joint*
distribution over $(\mu, Y)$. It is reasonable to think that the processes by
which the observation $y$ is generated might also influence the generation of
the forecast $\mu$. For example, some meteorological processes that lead to
a temperature observation $y$ might contribute to other sources of observed
data on which the forecast $\mu$ is based. The main takeaway here is
that a theory of probabilistic forecasting should be based around the joint
distribution between the forecast and the observation. We will call this joint
probability space the **prediction space**. The below definition
summarizes this idea, and fills in some of the measure-theoretical details.

<blockquote>
  <p><strong>Definition.</strong>
  Let $Y: \Omega^\prime \to (\mathcal{Y}, \mathcal{B})$ be a random variable
  mapping from some underlying probability space
  $(\Omega^\prime, \mathcal{B}^\prime, \mathbb{P})$.
  A prediction space is a probability space $(\Omega, \mathcal{A}, \mathbb{Q})$,
  along with a sub-$\sigma$-algebra $\mathcal{A}_1 \subseteq \mathcal{A}$,
  where $\Omega = \mathcal{M} \times \mathcal{Y}$, where $\mathcal{M}$ is a space
  of probability measures.
  </p>
</blockquote>

{% endkatexmm %}







## References
1. Predicting good probabilities with supervised learning.
2. https://scikit-learn.org/stable/modules/calibration.html
3. A General Framework for Forecast Verification (Murphy and Winkler, 1987)
4. Coherant Combination of Experts' Opinions (Dawid, 1995)
5. CS269I: Incentives in Computer Science Lecture #17: Scoring Rules and Peer Prediction (Incentivizing Honest Forecasts and Feedback)
