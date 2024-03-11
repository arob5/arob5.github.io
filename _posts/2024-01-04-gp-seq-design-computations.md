---
title: Some Useful Computations for Sequential Design and Optimization with Gaussian Processes
subtitle: I review and derive various formulas that come in handy when sequentially adding data to a Gaussian process model.
layout: default
date: 2023-12-31
keywords: Gaussian-Process, Bayesian-Optimization, Sequential-Design
published: false
---

{% katexmm %}
In a typical regression analysis, it is common to view the *design matrix*
$X$ as fixed, with the observed values of the independent variables out of your
control; the dataset is handed to you and the rest of your analysis proceeds
conditional on the fixed $X$. However, in certain settings the choice of inputs
$\mathbf{x}$ comprising the design $X$ takes center stage. Think, for example,
of an environmental setting where $X$ corresponds to physical locations where
some sort of sample is taken or a sensor is placed. Alternatively, $\mathbf{x}$
might represent the input to an expensive computer simulator, with the task  
being to quantify the sensitivity of the simulation outputs to variations in  
$\mathbf{x}$. What both scenarios have in common is that
1. the experimenter has control over the independent variable $\mathbf{x}$.
2. observing an output corresponding to an input $x$ is costly (in time,
  computation, or money).

In such settings of *experimental design*, great care is thus taken in the
choice of $X$. Given the importance of each *design point*, a common approach is
to construct $X$ sequentially: carefully choose a new point $\mathbf{x}$, add it
to the model and see how things change, then given the updated information choose
the next point. Such a *sequential design* procedure is often employed in
Gaussian process (GP) models, which are ubiquitous in the fields of
black-box optimization, design of computer experiments, and Bayesian quadrature.
While the objectives may be different depending on the application, similar
sequential strategies are used across all of these fields. In this post, I
briefly describe the general setting and highlight some useful formulas and
computations that often come up in these contexts.
{% endkatexmm %}
